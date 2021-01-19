# BERT for RACE

By: ASC commitee

### Implementation
This work is based on Pytorch implementation of BERT. We adapted the original BERT model to work on multiple choice machine comprehension.

### Environment:
The code is tested with nvcr.io/nvidia/pytorch:20.06-py3

### Usage
1. Download the dataset and unzip it. The default dataset directory is ./RACE.
2. Download the pretrained model file `bert-large-uncased.tar.gz` and `bert-large-uncased-vocab.txt`
3. Runing the following script launches fine-tuning for reading comprehension with RACE dataset  ```bash ./run.sh```
4. Inference can be perfomed with the ```bash ./eval.sh```

You can modify the parameters and the code in the script and the baseline code according to your own needs. However, you must use the pre-training model and training dataset, which we provided.

1. Number of training Epochs must be set to specified valued, which will be provided on the spot of the final
2. You need to dump the fine tuning log to `result/race.log`.
3. You need to dump the loss curve and time stamp using TensorboardX, and output to the directory `result/ascxxx/`, where `ascxxx` is your team number
4. The results of inference with test dataset need to be output to `result/answers.json`.
5. Finally, you need to submit the script/code, fine-tuned model, and `result` folder to us.







