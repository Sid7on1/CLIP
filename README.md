# ***CLIP***
CLIP From Scratch is a clean PyTorch implementation of OpenAI’s CLIP model. It uses ResNet-50 as a vision encoder and BERT-base for text encoding to align image and text pairs using contrastive loss. Includes training, inference, and modular design for easy integration and experimentation.

# CLIP From Scratch

This project is a clean, modular PyTorch implementation of OpenAI's CLIP model (Contrastive Language-Image Pretraining). It learns to align images and text in a shared embedding space.

## Features
- Vision encoder: ResNet-50
- Text encoder: BERT-base
- Contrastive loss between image and text embeddings

## Usage
1. Prepare your dataset as a list of (image_path, caption) pairs.
2. Set hyperparameters in `config.yaml`.
3. Run training:
```bash
python train.py
```

## TODO
- Add ViT backbone
- Add pretrained weight loader
- Integrate with SLAM multimodal

## Example
See `examples/inference.ipynb` for testing the model.
