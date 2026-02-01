- `unetArchitecture.py`: Loads & preprocesses DSB-style data → builds/trains U-Net → predicts on train/val/test → visualizes masks.

- `ResUnet.py`: Implements a Residual U-Net (ResUNet) model with encoder–decoder structure (residual blocks + skip connections) for image segmentation; includes model construction and summary (and can be extended for training/inference).