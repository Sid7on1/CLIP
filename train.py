import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from clip_model import CLIPModel
from dataset import ImageTextDataset
import yaml
import os
import json

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Sample data: replace with your own loader
data = [("images/dog.jpg", "A cute dog"), ("images/cat.jpg", "A sleepy cat")]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = ImageTextDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel(embed_dim=config['embed_dim']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
loss_fn = nn.CrossEntropyLoss()

for epoch in range(config['epochs']):
    model.train()
    for batch in dataloader:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(image, input_ids, attention_mask)
        labels = torch.arange(len(logits)).to(device)

        loss_i = loss_fn(logits, labels)
        loss_t = loss_fn(logits.T, labels)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
