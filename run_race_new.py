# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import time
import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
import numpy as np
import torch
from apex import amp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, PreTrainedBertModel, BertModel, BertConfig, BertPooler
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.utils import is_main_process

from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)



class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label



## paths is a list containing all paths
def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id = filename+'-'+str(i),
                            context_sentence = article,
                            start_ending = question,

                            ending_0 = options[0],
                            ending_1 = options[1],
                            ending_2 = options[2],
                            ending_3 = options[3],
                            label = truth))
                
    return examples 

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

from apex.multi_tensor_apply import multi_tensor_applier
class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """
    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)

class SSingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SSingleMatchNet, self).__init__()
        self.map_linear = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop_module = nn.Dropout(2*config.hidden_dropout_prob)
        self.rank_module = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2) )
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.trans_linear(att_vec))
        return output

class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear(q)
        lp = self.linear(p)
        mid = nn.Sigmoid()(lq+lp)
        output = p * mid + q * (1-mid)
        return output

def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i]+1]
        doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
        ques_option_seq_output[i, :ques_len[i]+option_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 1]
        option_seq_output[i, :option_len[i]] = sequence_output[i,
                                                 doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                   i] + 2]
    return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


class BertForMultipleChoiceWithMatch(PreTrainedBertModel):
    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoiceWithMatch, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(2*config.hidden_size, 1)
        self.classifier3 = nn.Linear(3*config.hidden_size, 1)
        self.classifier4 = nn.Linear(4*config.hidden_size, 1)
        self.classifier6 = nn.Linear(6*config.hidden_size, 1)
        self.ssmatch = SSingleMatchNet(config)
        self.pooler = BertPooler(config)
        self.fuse = FuseNet(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None, option_len=None, labels=None, is_3 = False):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        sequence_output, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(sequence_output, doc_len, ques_len, option_len)
       
        pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len+1])
        ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len+1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len+1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len+1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len+1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len+1])

        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse= self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse= self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse= self.fuse([qa_output_pool, aq_output_pool])

        cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier3(output_pool)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss
        else:
            return match_reshaped_logits

def main():
    ete_start = time.time()
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="vocab_file path")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if is_main_process():
        logger.info("device: {} ({}), n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, torch.cuda.get_device_name(0), n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_dir = os.path.join(args.data_dir, 'train')
        train_examples = read_race_examples([train_dir+'/high', train_dir+'/middle'])
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
#    model = BertForMultipleChoice.from_pretrained(args.bert_model,
#        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
#        num_choices=4)
    model = BertForMultipleChoiceWithMatch.from_pretrained(args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
        num_choices=4)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)


    global_step = 0
    train_start = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir, "ascxx"))
    if args.do_train:
        train_features = convert_examples_to_features_new(
                train_examples, tokenizer, args.max_seq_length, True)
        if is_main_process():
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        all_doc_len = torch.tensor(select_field(train_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(train_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(train_features, 'option_len'), dtype=torch.long)
        #train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                                   all_option_len)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        gradClipper = GradientClipper(max_grad_norm=1.0)
        for ep in range(int(args.num_train_epochs)):
            tr_loss = 0
            train_iter = tqdm(train_dataloader, disable=False) if is_main_process() else train_dataloader
            if is_main_process():
                train_iter.set_description("Trianing Epoch: {}/{}".format(ep+1, int(args.num_train_epochs)))
            for step, batch in enumerate(train_iter):
                batch = tuple(t.to(device) for t in batch)
                #input_ids, input_mask, segment_ids, label_ids = batch
                #loss = model(input_ids, segment_ids, input_mask, label_ids)
                input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len = batch
                loss = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                loss.backward()
                gradClipper.step(amp.master_params(optimizer))
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if is_main_process():
                    train_iter.set_postfix(loss=loss.item())
                writer.add_scalar('loss', loss.item(), global_step=global_step)


    finish_time = time.time()
    writer.close()
    # Save a trained model
    if is_main_process():
        logger.info("ete_time: {}, training_time: {}".format(finish_time-ete_start, finish_time-train_start))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model_new.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
