python run_race.py \
    --data_dir=./RACE \
    --bert_model=./model/bert-large-uncased.tar.gz \
    --vocab_file=./model/bert-large-uncased-vocab.txt \
    --output_dir=large_models \
    --max_seq_length=320 --do_train --do_lower_case --train_batch_size=16 --eval_batch_size=12 --learning_rate=1e-5 --num_train_epochs=2 --gradient_accumulation_steps=1 --fp16 --loss_scale=128



