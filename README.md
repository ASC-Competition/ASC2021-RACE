# BERT for RACE

By: ASC commitee

## Environment:
The code is tested with nvcr.io/nvidia/pytorch:20.12-py3

## Usage
1. Extrace the compressed dataset file (train_LE.tar.gz, dev_LE.tar.gz and test_LE.tar.gz). The default dataset directory is .source_code/RACE.
2. Extrace the pretrained model file and config files to the folder `source_code/pretrained_model_asc`
3. Runing the following script launches fine-tuning for reading comprehension with RACE dataset  ```cd source_code && bash ./run.sh```
4. Inference can be perfomed with the ```cd source_code && bash ./eval.sh```

You can modify the parameters and the code in the script and the source code according to your own needs. However, you must use the pre-training model and training dataset, which we provided. It's not allowed to modified the backbone model structure or use extra fine-tuning dataset.

1. You need to train no more than 3 epochs.
2. You need to dump the fine tuning log to `log/race.log`.
3. You need to dump the loss curve and time stamp using TensorboardX, and output to the directory `log/ascxxx/`, where `ascxxx` is your team number
4. The results of inference with test dataset need to be output to 'result.json'.
5. Finally, you need to submit the source_code, fine-tuned model, log folder and 'result.json' file to us.



## For Teams using AWS

while you use AWS env, you can run LE task with follow steps:

1. Download data: `aws s3 sync s3://<S3BucketName>/LE/* ~/LE/`, Extract the tar.gz files to the specified directory
2. Build Docker image: `cd source_code && docker build -t asc/le .`
3. submit task `srun -N 1 -C gpu sudo docker run --gpus all -v ~/LE/source_code:/workspace/source_code --ipc=host asc/le:latest bash run.sh`
4. submit task `srun -N 1 -C gpu sudo docker run --gpus all -v ~/LE/source_code:/workspace/source_code --ipc=host asc/le:latest bash eval.sh`