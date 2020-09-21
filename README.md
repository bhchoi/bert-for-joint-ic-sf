# BERT for Joint Intent Classification and Slot Filling
PyTorch Lightning implementation of : [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

# Dataset
* atis
* snips

# Config
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

# Training
```python
python train.py
```

# Reference 
* https://arxiv.org/abs/1902.10909
