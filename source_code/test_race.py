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

import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
import apex

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForMultipleChoice
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.utils import is_main_process
from pytorch_pretrained_bert import tokenization_albert
from pytorch_pretrained_bert.modeling_albert import AlbertForMultipleChoice, AlbertConfig

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
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label



## paths is a list containing all paths
def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path+"/*json")
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

## paths is a list containing all paths
def read_race_example(filename):
    example=[]
    with open(filename, 'r', encoding='utf-8') as fpr:
        data_raw = json.load(fpr)
        article = data_raw['article']
        ## for each qn
        for i in range(len(data_raw['answers'])):
            truth = ord(data_raw['answers'][i]) - ord('A')
            question = data_raw['questions'][i]
            options = data_raw['options'][i]
            example.append(
                RaceExample(
                    race_id = filename+'-'+str(i),
                    context_sentence = article,
                    start_ending = question,

                    ending_0 = options[0],
                    ending_1 = options[1],
                    ending_2 = options[2],
                    ending_3 = options[3],
                    label = truth))

    return example

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option 
    # 
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    examples_iter = tqdm(examples, disable=False) if is_main_process() else examples
    for example_index, example in enumerate(examples_iter):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        ## display some example
        # if example_index < 1:
        #     logger.info("*** Example ***")
        #     logger.info(f"race_id: {example.race_id}")
        #     for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
        #         logger.info(f"choice: {choice_idx}")
        #         logger.info(f"tokens: {' '.join(tokens)}")
        #         logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
        #         logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
        #         logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
        #     if is_training:
        #         logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

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

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="vocab_file path")
    parser.add_argument("--spm_model_file", default=None, type=str, required=True,
                        help="spm_model_file path")
    parser.add_argument("--config_file", default=None, type=str, required=True,
                        help="config_file path")
    parser.add_argument("--pretrained_model", default=None, type=str, required=True,
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")

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

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization_albert.FullTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, spm_model_file=args.spm_model_file)


    config = AlbertConfig.from_pretrained(args.config_file, num_labels=4)
    model = AlbertForMultipleChoice.from_pretrained(args.bert_model, config=config)    

    model.to(device)


    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_dir = os.path.join(args.data_dir, 'test')
        results = {}
        for fid in range(500):
            s = str(fid).zfill(5)
            fname = test_dir+'/asc'+s+'.txt'
            eval_examples = read_race_example(fname)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
            model.eval()
            answers = []
            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    outputs = model(input_ids, input_mask, segment_ids, label_ids)
                    tmp_eval_loss = outputs[0]
                    outputs = model(input_ids, input_mask, segment_ids)
                    logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                result = np.argmax(logits, axis=1)
                answers.append(chr(ord('A')+result))
            results[s]=answers

        with open(args.output_dir+"/asc_result.json","w") as f:
            json.dump(results, f, indent=1)

        # eval_loss = high_eval_loss / high_nb_eval_steps
        # eval_accuracy = high_eval_accuracy / high_nb_eval_examples

        # result = {'asc_eval_loss': eval_loss,
        #           'asc_eval_accuracy': eval_accuracy}

        # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # with open(output_eval_file, "a+") as writer:
        #     logger.info("***** Eval results *****")
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
