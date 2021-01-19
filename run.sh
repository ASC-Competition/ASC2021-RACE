CUDA_VISIBLE_DEVICES=0 python run_race.py --data_dir=/dev/shm/RACE --bert_model=/dev/shm/bert-large-uncased.tar.gz --vocab_file=/workspace/model/bert-large-uncased-vocab.txt --output_dir=large_models --max_seq_length=320 --do_train --do_lower_case --train_batch_size=6 --eval_batch_size=12 --learning_rate=1e-5 --num_train_epochs=2 --gradient_accumulation_steps=1 --fp16 --loss_scale=128



