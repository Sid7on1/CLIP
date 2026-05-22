# CLIP (Contrastive Language-Image Pre-Training)

## Description
A PyTorch implementation of CLIP-style contrastive learning for vision-language alignment. Uses ResNet-50 as the vision encoder and BERT as the text encoder, with contrastive loss for training.

## Version
v1.0

## Status
**Complete** - Functional Implementation

## Assessment
A complete, functional CLIP implementation with:
- `clip_model.py`: Full CLIPModel class with ResNet-50 vision encoder and BERT text encoder
- `train.py`: Complete training loop with contrastive loss
- `dataset.py`: ImageTextDataset class
- `utils.py`: Utility functions
- `inference.ipynb`: Jupyter notebook for inference
- `config.yaml`: Configuration for training parameters
- `dataset.py`: Data loading utilities

The code is well-structured and follows standard CLIP training practices (symmetric contrastive loss, temperature parameter, normalization). Missing: preprocessed datasets, but sample data loading code is provided.

## File Structure
```
/
├── clip_model.py     # CLIP model (ResNet-50 + BERT)
├── train.py          # Training script with contrastive loss
├── dataset.py        # ImageTextDataset class
├── utils.py          # Utility functions
├── config.yaml       # Training configuration
├── inference.ipynb   # Jupyter notebook for inference
├── LICENSE
└── README.md
```
