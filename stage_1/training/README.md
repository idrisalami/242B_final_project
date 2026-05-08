# Training Module

## Overview

Handles model training with contrastive learning objectives.

## Files

- `train.py`: Main training entry point
- `trainer.py`: Training loop and checkpointing

## Key Classes

### `InBatchInfoNCELoss`

Contrastive loss using in-batch negatives.

**Why in-batch negatives?**
- All items in batch serve as negatives for positive pairs
- More efficient than explicit negative sampling
- Larger batch = stronger training signal
- No additional memory overhead

**Formula:**
$$\text{Loss} = -\log \frac{\exp(\text{score}(u, i^+) / \tau)}{\sum_{j} \exp(\text{score}(u, i_j) / \tau)}$$

**Usage:**
```python
loss_fn = InBatchInfoNCELoss(temperature=1.0)
loss = loss_fn(user_embeddings, positive_embeddings, negative_embeddings)
```

### `TwoTowerTrainer`

Complete training pipeline.

**Features:**
- Automatic checkpointing
- Training history logging
- Validation loop
- Gradient clipping
- Learning rate scheduling

**Key methods:**
- `train_epoch()`: Single epoch training
- `validate()`: Validation loop
- `train()`: Full training loop
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load from checkpoint

**Usage:**
```python
trainer = TwoTowerTrainer(
    model=model,
    device="cuda",
    learning_rate=0.001,
    temperature=1.0,
    checkpoint_dir="./checkpoints/"
)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10
)
```

## Training Process

### 1. Setup

```python
python training/train.py config/config.yaml
```

### 2. Training Loop

For each epoch:
1. **Forward pass**: Get user/item embeddings
2. **Loss computation**: In-batch InfoNCE loss
3. **Backward pass**: Gradient computation
4. **Optimization**: Adam optimizer update
5. **Validation**: Evaluate on validation set
6. **Checkpointing**: Save if best

### 3. Checkpointing

Models saved:
- After each epoch (if enabled)
- Best validation loss (default)
- Final model after training

**Files:**
- `checkpoints/model_epoch_1.pt`: Weights per epoch
- `checkpoints/training_history.json`: Loss curves
- `checkpoints/config.yaml`: Configuration used

## Configuration

From `config.yaml`:

```yaml
training:
  batch_size: 256
  learning_rate: 0.001
  num_epochs: 10
  warmup_steps: 500
  optimizer: "adam"
  loss_function: "in_batch_softmax"
  device: "cuda"
  num_workers: 4
```

## Training Tips

### Batch Size
- **Larger batches**: More in-batch negatives, more diverse training
- **Typical**: 256-512
- **Constraint**: GPU memory

### Learning Rate
- **Starting point**: 1e-3
- **Schedule**: Cosine annealing with warmup
- **Monitoring**: Check training curves

### Epochs
- **Typical**: 5-10 for convergence
- **Early stopping**: Available via validation loss

### Hardware
- **GPU**: NVIDIA RTX 3090 or better recommended
- **Memory**: 24GB+ for large batches
- **Training time**: 2-4 hours for 10 epochs on 100k playlists

## Monitoring Training

### Training History

Saved to `checkpoints/training_history.json`:

```json
{
  "train_loss": [0.523, 0.412, 0.387, ...],
  "val_loss": [0.512, 0.425, 0.398, ...],
  "val_recall": {}
}
```

### Visualizing Progress

Plot loss curves:
```python
import json
import matplotlib.pyplot as plt

with open("checkpoints/training_history.json") as f:
    history = json.load(f)

plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

## Common Issues

### Out of Memory
- Reduce batch size
- Reduce model size (embedding_dim, hidden_dims)
- Use mixed precision training

### Training not converging
- Check learning rate (try 0.0005 or 0.005)
- Verify data loading (check data shapes)
- Increase batch size for more negatives

### Slow training
- Check GPU utilization (use `nvidia-smi`)
- Increase num_workers for data loading
- Use mixed precision (torch.cuda.amp)
