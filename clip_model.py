import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class CLIPModel(nn.Module):
    def __init__(self, embed_dim=512):
        super(CLIPModel, self).__init__()

        # Vision Encoder (ResNet-50 backbone)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove last fc layer
        self.vision_encoder = nn.Sequential(*modules)
        self.vision_proj = nn.Linear(resnet.fc.in_features, embed_dim)

        # Text Encoder (BERT base)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, image, input_ids, attention_mask):
        # Vision path
        vision_feat = self.vision_encoder(image).squeeze(-1).squeeze(-1)
        vision_embed = self.vision_proj(vision_feat)
        vision_embed = vision_embed / vision_embed.norm(dim=1, keepdim=True)

        # Text path
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_embed = self.text_proj(text_feat)
        text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

        # Similarity
        logits = torch.matmul(vision_embed, text_embed.T) / self.temperature
        return logits
