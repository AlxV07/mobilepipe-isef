import argparse
import sys

import torch
from torch import nn
from torchvision import models

from experiments.cnn.utils.loader import load
from mobilepipe.train.trainer import DefaultTrainer, TrainingArguments


# -------- Config --------
parser = argparse.ArgumentParser(description='Default Training Script')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32, must be divisible by 4)')
args = parser.parse_args()

if args.batch_size % 4 != 0:
    print(f"Error: Batch size ({args.batch_size}) must be divisible by 4")
    sys.exit(1)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = 1e-3 * (BATCH_SIZE // 32)

# -------- Dataset --------
dataset = load()


# -------- Model --------
def freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False


num_labels = dataset['train'].features['label'].num_classes
model = models.resnet34(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_labels)
model.to(torch.device('cpu'))
model.apply(freeze_bn)

# -------- Training --------
training_args = TrainingArguments(learning_rate=LR,
                                  epochs=EPOCHS,
                                  batch_size=BATCH_SIZE)

trainer = DefaultTrainer(model=model,
                         args=training_args,
                         train_dataset=dataset['train'])

trainer.train()

# -------- Save --------
trainer.save_model('resnet34_default.pt')

