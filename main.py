from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple
from typing import List
from typing import List, Dict, Optional
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PrunableLinear(nn.Module):
                                                 
  

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

                                                           
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

                                                        
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

                       
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
                                                                         
                                                                         
                                     
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

                                                                                    
        self.gate_scores.data = torch.ones_like(self.weight) * 2.0

        if self.bias is not None:
                                                                     
            fan_in = self.in_features
            bound = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                           

                                                       
                                          
                                               
           
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self) -> torch.Tensor:
                                                                 
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class SelfPruningNetwork(nn.Module):
                                                                        
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                                                                         
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)                                   
        return x

    def get_prunable_layers(self) -> List['PrunableLinear']:
                                                              
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]




matplotlib.use("Agg")                                            


                                                                             
                 
                                                                             

def set_seed(seed: int = 42) -> None:
                                                        
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


                                                                             
              
                                                                             

def sparsity_loss(model: SelfPruningNetwork) -> torch.Tensor:
                                                            

                                                        

                                                                    
                                                                          
       
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    numel = 0.0
    for layer in model.get_prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.sum()
        numel += gates.numel()
    return total / numel if numel > 0 else total


                                                                             
                   
                                                                             

def compute_sparsity(
    model: SelfPruningNetwork,
    threshold: float = 1e-2,
) -> dict:
                                                  
                         
       
    per_layer = []
    total_pruned = 0
    total_weights = 0

    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gate_values()
            n_pruned = int((gates <= threshold).sum().item())
            n_total = gates.numel()
            pct = 100.0 * n_pruned / n_total
            per_layer.append((name, n_pruned, n_total, pct))
            total_pruned += n_pruned
            total_weights += n_total

    overall_pct = 100.0 * total_pruned / total_weights if total_weights > 0 else 0.0

    return {
        "per_layer": per_layer,
        "total_pruned": total_pruned,
        "total_weights": total_weights,
        "active_weights": total_weights - total_pruned,
        "sparsity_pct": overall_pct,
        "compression_ratio": total_weights / (total_weights - total_pruned) if (total_weights - total_pruned) > 0 else float("inf"),
    }


def print_sparsity_report(model: SelfPruningNetwork, threshold: float = 1e-2):
                                                      
    info = compute_sparsity(model, threshold)
    print("\n" + "=" * 60)
    print("SPARSITY REPORT  (threshold = {:.0e})".format(threshold))
    print("=" * 60)
    for name, pruned, total, pct in info["per_layer"]:
        print(f"  {name:>6s}:  {pruned:>7d} / {total:>7d}  ({pct:5.1f}% pruned)")
    print("-" * 60)
    print(
        f"  TOTAL :  {info['total_pruned']:>7d} / {info['total_weights']:>7d}  "
        f"({info['sparsity_pct']:5.1f}% pruned) | "
        f"Active: {info['active_weights']}/{info['total_weights']}"
    )
    print("=" * 60 + "\n")
    return info


                                                                             
                                       
                                                                             

def collect_gate_values(model: SelfPruningNetwork) -> np.ndarray:
                                                                         
    all_gates = []
    for layer in model.get_prunable_layers():
        all_gates.append(layer.get_gate_values().cpu().numpy().ravel())
    return np.concatenate(all_gates)
              
                                                                             

def get_cifar10_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:                         
       
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, test_loader

matplotlib.use("Agg")


def set_style():
                                                            
    plt.style.use('seaborn-v0_8-darkgrid')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['axes.labelsize'] = 12

def plot_learning_curves(history: List[Dict], output_dir: Path, lambda_val: float):
                                                       
    set_style()
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    sparsity = [h['sparsity_pct'] for h in history]

                   
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, marker='o', label='Train Accuracy', color='#2ca02c')
    ax.plot(epochs, test_acc, marker='o', label='Test Accuracy', color='#1f77b4')
    ax.set_title(f'Accuracy Progression (λ={lambda_val})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f"accuracy_lambda_{lambda_val}.png", dpi=150)
    plt.close(fig)

                   
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, sparsity, marker='s', label='Sparsity %', color='#d62728')
    ax.set_title(f'Sparsity Progression (λ={lambda_val})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sparsity (%)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f"sparsity_lambda_{lambda_val}.png", dpi=150)
    plt.close(fig)

def plot_combined_metrics(history: List[Dict], output_dir: Path, lambda_val: float):
                                                                  
    set_style()
    epochs = [h['epoch'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    sparsity = [h['sparsity_pct'] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = '#1f77b4'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)', color=color)
    ax1.plot(epochs, test_acc, color=color, marker='o', label="Test Acc", linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = '#d62728'
    ax2.set_ylabel('Sparsity (%)', color=color)
    ax2.plot(epochs, sparsity, color=color, marker='s', label="Sparsity", linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'Accuracy vs. Sparsity Dynamics (λ={lambda_val})', fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / f"combined_metrics_lambda_{lambda_val}.png", dpi=150)
    plt.close(fig)

def plot_gate_histogram(
    model: SelfPruningNetwork,
    lambda_val: float,
    save_path: Optional[str] = None
):
                                                             
    gates = collect_gate_values(model)
    set_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=100, color="#9467bd", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Gate Value (sigmoid of gate_score)", fontsize=12)
    ax.set_ylabel("Parameter Count", fontsize=12)
    ax.set_title(f"Final Gate Value Distribution (λ={lambda_val})", fontsize=14)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Pruning Threshold (0.01)")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


                                                                        
                  
                                                                        

   
                                                                    

                                                                      
                                                               

      
                   
   



                                                                             
               
                                                                             

               
                                                                             

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


                                                                             
                    
                                                                             

def train_one_epoch(
    model: SelfPruningNetwork,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lambda_val: float,
    device: torch.device,
) -> dict:
                                                                     
    model.train()
    running_cls_loss = 0.0
    running_sp_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

                 
        logits = model(images)
        cls_loss = criterion(logits, labels)
        sp_loss = sparsity_loss(model)
        total_loss = cls_loss + lambda_val * sp_loss

                  
        optimizer.zero_grad()
        total_loss.backward()
        
                           
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

                 
        running_cls_loss += cls_loss.item() * images.size(0)
        running_sp_loss += sp_loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    n = total
    return {
        "cls_loss": running_cls_loss / n,
        "sp_loss": running_sp_loss / n,
        "total_loss": (running_cls_loss + lambda_val * running_sp_loss) / n,
        "accuracy": 100.0 * correct / n,
    }


                                                                             
            
                                                                             

@torch.no_grad()
def evaluate(
    model: SelfPruningNetwork,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
                                                                       
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": 100.0 * correct / total,
    }


                                                                             
                             
                                                                             

def run_training_routine(
    lambda_val: float,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 4
) -> dict:
                                                                               
    
          
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

           
    model = SelfPruningNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
                                               
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nDevice: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}\n")

                           
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
                                            
            current_lambda = 0.0
        else:
                                                        
            current_lambda = lambda_val * (epoch / epochs)
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, current_lambda, DEVICE,
        )
        test_metrics = evaluate(model, test_loader, criterion, DEVICE)
        
                           
        scheduler.step()

                                 
        sp_info = compute_sparsity(model)
        sparsity_pct = sp_info["sparsity_pct"]
        active_weights = sp_info["active_weights"]
        total_weights = sp_info["total_weights"]

        history.append({
            "epoch": epoch,
            "train_cls_loss": train_metrics["cls_loss"],
            "train_sp_loss": train_metrics["sp_loss"],
            "train_total_loss": train_metrics["total_loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "sparsity_pct": sparsity_pct,
            "active_weights": active_weights,
            "total_weights": total_weights
        })

        print(
            f"Epoch {epoch:>2d}: Loss={train_metrics['cls_loss']:.4f}, "
            f"Accuracy={test_metrics['accuracy']:5.2f}%, "
            f"Sparsity={sparsity_pct:5.1f}%, "
            f"Remaining Params={active_weights}/{total_weights}"
        )

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")

                              
    final_test = evaluate(model, test_loader, criterion, DEVICE)
    sp_report = print_sparsity_report(model)

    print(f"Model compressed from {sp_report['total_weights']} → {sp_report['active_weights']} "
          f"parameters ({sp_report['sparsity_pct']:.1f}% reduction)")

                                 
    plot_learning_curves(history, RESULTS_DIR, lambda_val)
    plot_combined_metrics(history, RESULTS_DIR, lambda_val)
    hist_path = RESULTS_DIR / f"gate_histogram_lambda_{lambda_val}.png"
    plot_gate_histogram(model, lambda_val, save_path=str(hist_path))

                                   
    ckpt_path = RESULTS_DIR / f"model_lambda_{lambda_val}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"→ Model saved to {ckpt_path}\n")

    return {
        "lambda": lambda_val,
        "test_accuracy": final_test["accuracy"],
        "sparsity_pct": sp_report["sparsity_pct"],
        "active_params": sp_report["active_weights"],
        "total_params": sp_report["total_weights"],
        "epochs": epochs,
        "history": history,
    }




                                                                        
                        
                                                                        

   
                                                                    

                                                                              
                                                                                         
   


                                        
set_seed(42)

def generate_markdown_table(results: list, filepath: Path):
                                                              
    with open(filepath, "w") as f:
        f.write("## Experiment Results\n\n")
        f.write("| Lambda | Accuracy | Sparsity | Remaining Params |\n")
        f.write("|--------|----------|----------|------------------|\n")
        
        for res in results:
            lam = res["lambda"]
            acc = res["test_accuracy"]
            sp  = res["sparsity_pct"]
            rem = res["active_params"]
            tot = res["total_params"]
            
            f.write(f"| {lam:.4f} | {acc:.2f}% | {sp:.1f}% | {rem} / {tot} |\n")
            
    print(f"\n✅ Automatically generated Results Table at -> {filepath}")


def run_all_experiments():
    print("=" * 70)
    print("🚀 LAUNCHING MULTI-LAMBDA PRUNING EXPERIMENTS")
    print("=" * 70)

    lambdas_to_test = [0.0001, 0.001, 0.005]
    all_results = []
    results_dir = Path("results")

    for lam in lambdas_to_test:
        print("\n" + "#" * 60)
        print(f"### Evaluating Model Configuration: λ = {lam}")
        print("#" * 60)
        
                                                                
        results = run_training_routine(
            lambda_val=lam, 
            epochs=15, 
            batch_size=128, 
            lr=1e-3, 
            weight_decay=1e-4, 
            warmup_epochs=4
        )
        all_results.append(results)

    print("\n" + "=" * 70)
    print("🏆 ALL EXPERIMENTS COMPLETED")
    print("=" * 70)

                                                   
    generate_markdown_table(all_results, results_dir / "results_table.md")

if __name__ == "__main__":
    run_all_experiments()
