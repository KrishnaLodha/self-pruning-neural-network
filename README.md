# Self-Pruning Neural Network

A neural network that learns to remove its own unnecessary connections during training. Instead of pruning weights after training is done, this model figures out which connections matter and which don't — while it's still learning.

Built with PyTorch. Tested on CIFAR-10.

---

## What Does This Do?

Every weight in the network has a "gate" attached to it. This gate is a number between 0 and 1.

- Gate close to **1** → the weight is important, keep it
- Gate close to **0** → the weight is useless, prune it

The network learns these gates automatically during training. We add a penalty (L1 sparsity loss) that pushes gates toward zero. The only gates that survive are the ones the network truly needs.

The result: a smaller, faster model that performs almost as well as the full model.

---

## How It Works

### Custom Layer: `PrunableLinear`

Replaces the standard `nn.Linear`. Each weight gets a learnable `gate_score`. During forward pass:

```
gates = sigmoid(gate_scores)
effective_weight = weight × gates
```

Gradients flow through both `weight` and `gate_scores`, so the network jointly learns what to compute and which connections to keep.

### Loss Function

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

- **CrossEntropyLoss** pushes the model to classify correctly
- **SparsityLoss** (L1 norm of all gates) pushes gates toward zero
- **λ** controls the balance — higher λ means more pruning

### Architecture

```
Input (3072) → Linear (512) → BatchNorm → ReLU → Linear (256) → BatchNorm → ReLU → Linear (10)
```

A simple 3-layer feed-forward network. No convolutions — this is intentional to keep the focus on the pruning mechanism.

---

## Training Strategy

| Technique | Why |
|-----------|-----|
| **Warm-up (4 epochs)** | Let the model learn features before pruning starts |
| **Lambda scheduling** | λ increases gradually — no sudden pruning collapse |
| **AdamW optimizer** | Better weight decay handling than standard Adam |
| **Cosine Annealing LR** | Smooth learning rate decay for stable convergence |
| **Gradient clipping** | Prevents training instability from sparsity gradients |
| **Gate init at 2.0** | sigmoid(2.0) ≈ 0.88, so all connections start active |

---

## Results

Tested across three λ values on CIFAR-10:

| Lambda | Test Accuracy | Sparsity | What Happens |
|--------|--------------|----------|--------------|
| 0.0001 | ~57% | Low | Almost no pruning, full accuracy |
| 0.001 | ~57% | Medium | Moderate pruning, accuracy preserved |
| 0.005 | ~55-56% | High | Aggressive pruning, slight accuracy drop |

**Note on accuracy:** This is a fully connected (dense) network, not a CNN. Dense networks on CIFAR-10 typically max out around 55-60% because they can't extract spatial features from images. The pruning mechanism works correctly — the accuracy-sparsity tradeoff behaves exactly as expected.

---

## Generated Outputs

Running the script produces these files in `results/`:

- `accuracy_lambda_*.png` — accuracy curves per epoch
- `sparsity_lambda_*.png` — sparsity curves per epoch
- `combined_metrics_lambda_*.png` — accuracy vs sparsity on one graph
- `gate_histogram_lambda_*.png` — distribution of final gate values
- `model_lambda_*.pt` — saved model checkpoints
- `results_table.md` — auto-generated results table

---

## How to Run

```bash
pip install torch torchvision matplotlib numpy
python main.py
```

Everything runs from a single file. It trains 3 models (one per λ value), generates all plots, and saves results automatically.

---

## Project Structure

```
├── main.py          # Everything — model, training, plotting, experiments
├── report.md        # Analysis of L1 sparsity and observations
├── README.md        # This file
├── results/         # Generated plots, checkpoints, and tables
└── data/            # CIFAR-10 (auto-downloaded on first run)
```
