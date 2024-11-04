import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .train import train_model
from .transformer import BERTForSequenceClassification
from .preprocess import TimeSeriesDataset


# 假设 `sequences` 和 `labels` 分别是时序数据和标签
# 初始化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 64
batch_size = 16

# 创建 DataLoader
train_dataset = TimeSeriesDataset(sequences=train_sequences, labels=train_labels, tokenizer=tokenizer, max_len=max_len)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TimeSeriesDataset(sequences=val_sequences, labels=val_labels, tokenizer=tokenizer, max_len=max_len)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

# 初始化模型和设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTForSequenceClassification(n_classes=num_classes)
model = model.to(device)

# 训练模型
train_model(model, train_data_loader, val_data_loader, device)
