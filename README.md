# BERT for Joint Intent Classification and Slot Filling
PyTorch Lightning implementation of : [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

# Dataset
* atis
* snips

# Configuration
* config/train_config.yaml
```yaml
task: atis
log_path: logs
bert_model: bert-base-uncased
data_path: data
max_len: 64
train_batch_size: 128
eval_batch_size: 32
dropout_rate: 0.1
gpus: 8
distributed_backend: ddp
```
```yaml
task: atis
log_path: logs
bert_model: bert-base-uncased
data_path: data
ckpt_path: checkpoints/atis/epoch=9_val_loss=0.000.ckpt
max_len: 64
train_batch_size: 128
eval_batch_size: 32
dropout_rate: 0.1
gpus: 2
distributed_backend: ddp
```
# Training
```python
python train.py
```

# Evaluation
```python
python eval.py
```


# Reference 
* https://arxiv.org/abs/1902.10909
