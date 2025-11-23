# Introduction to Artificial Intelligence
# MNIST dataset
# Shallow Neural Network - Problem 2: RMSProp Optimizer
# Based on Mnist_ShallowNN1.py with Problem 1 modifications:
#   - 1000 hidden nodes (instead of 100)
#   - Early stopping disabled (early_stop_patience = 100000)
#   - ReLU activation function
#   - RMSProp optimizer (instead of Adam, its basically Adagrad but with a moving average of the gradients, and a decay rate)
# By Timothy Figueroa and Adrian De Souza
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import sklearn
import torch
import random
import os

# ============================================================
# Configuration for multiple runs
# ============================================================
NUM_RUNS = 3  # Number of runs for consistency
RUN_RESULTS = []  # Store results from each run

# ============================================================
# Load and prepare data
# ============================================================

# Load the training and test data from the Pickle file
with open("mnist_dataset.pickle", "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

# Get some lengths
num_classes = len(np.unique(train_labels))
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants
# ============================================================

n_nodes_l1 = 1000
batch_size = 1024
learning_rate = 1e-3  # Fine-tuned learning rate for RMSProp
num_epochs = 1000
eval_step = 1
early_stop_patience = 100000  # Disable early stopping

# RMSProp specific parameters
alpha = 0.99  # Decay rate (smoothing constant) - fine-tuned
eps = 1e-8   # Small constant for numerical stability
weight_decay = 0.0  # L2 penalty (can be fine-tuned)

# Print the configuration
print("=" * 70)
print("PROBLEM 2: RMSProp Optimizer")
print("=" * 70)
print(f"Num epochs: {num_epochs}  Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"RMSProp alpha (decay rate): {alpha}")
print(f"RMSProp eps: {eps}")
print(f"Number of runs: {NUM_RUNS}")
print("=" * 70)

num_batches = int(np.ceil(nsamples / batch_size))
total_iterations = num_epochs * num_batches
print(f"Number of batches per epoch: {num_batches}")
print(f"Total training iterations: {total_iterations}")

# ============================================================
# Select PyTorch device (GPU if available)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")  # Force CPU usage
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Convert data to PyTorch tensors and move to device
# ============================================================
X = torch.tensor(train_data, dtype=torch.float32, device=device)
Y = torch.tensor(train_labels, dtype=torch.long, device=device)
X_test = torch.tensor(test_data, dtype=torch.float32, device=device)
Y_test = torch.tensor(test_labels, dtype=torch.long, device=device)

# ============================================================
# Training function for a single run
# ============================================================
def train_single_run(run_num, seed=None):
    """
    Train the model for a single run with a given random seed.
    Returns: (training_time, best_test_acc, best_epoch, train_cost_hist, test_acc_hist)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*70}")
    print(f"RUN {run_num} of {NUM_RUNS}")
    print(f"{'='*70}")
    
    # ============================================================
    # Create and initialize neural network
    # ============================================================
    
    # Note that the final layer is skipping the Softmax activation
    # because the cost function contains a Softmax function inside
    model = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, n_nodes_l1),
        torch.nn.ReLU(),  # Using ReLU activation (same as Problem 1)
        torch.nn.Linear(n_nodes_l1, num_classes)
    )
    model.to(device)
    
    # Initialize weights explicitly (same as Problem 1)
    torch.nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
    torch.nn.init.xavier_normal_(model[2].weight)
    
    # ============================================================
    # Optimizer and Loss
    # ============================================================
    optimizer = torch.optim.RMSprop(
        model.parameters(), 
        lr=learning_rate,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"Optimizer: RMSProp with multinomial cross-entropy loss")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Alpha (decay rate): {alpha}")
    print(f"  Eps: {eps}")
    
    # ============================================================
    # Training loop
    # ============================================================
    train_cost_hist = []
    test_acc_hist = []
    
    start_time = time.time()
    
    best_test_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
    
        # Switch model to training mode
        model.train()
    
        cost_sum = 0.0
        for batch_idx in range(num_batches):
    
            # Get the batch data
            start = batch_idx * batch_size
            end = min(start + batch_size, nsamples)
            X_batch = X[start:end]
            Y_batch = Y[start:end]
    
            # Forward pass
            logits = model(X_batch)
            loss = loss_fn(logits, Y_batch)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            cost_sum += loss.item()
    
        # Evaluation
        if (epoch + 1) % eval_step == 0:
    
            avg_train_cost = cost_sum / num_batches
    
            # Switch model to evaluation mode 
            model.eval()
    
            with torch.no_grad():
                test_logits = model(X_test)
                test_pred = torch.argmax(test_logits, dim=1)
                test_acc = (test_pred == Y_test).float().mean().item()
    
            train_cost_hist.append(avg_train_cost)
            test_acc_hist.append(test_acc)
    
            print_line = f"Epoch {epoch+1:3d}, Train cost: {avg_train_cost:.4f}  Test Acc: {test_acc:.4f}"
    
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
                epochs_without_improvement = 0   # reset patience counter
                print_line += "  New best"
            else:
                epochs_without_improvement += 1  # increment patience counter
    
                # Early stopping condition
                if epochs_without_improvement >= early_stop_patience:
                    print_line += "  **Early stop triggered**"
                    print(print_line)
                    break
    
            # Print every 50 epochs to reduce output
            if (epoch + 1) % 50 == 0 or test_acc > best_test_acc:
                print(print_line)
    
    training_time = time.time() - start_time
    
    # ============================================================
    # Restore best model and final evaluation
    # ============================================================
    
    # Safety check: ensure best_state_dict exists (should always be set since best_test_acc starts at 0.0)
    if best_state_dict is None:
        print(f"Warning: No best model found for run {run_num}, using current model state")
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        best_test_acc = 0.0
        best_epoch = 0
    
    model.load_state_dict(best_state_dict)
    model.eval()
    
    with torch.no_grad():
        final_train_pred = torch.argmax(model(X), dim=1)
        final_test_pred = torch.argmax(model(X_test), dim=1)
        final_train_acc = (final_train_pred == Y).float().mean().item()
        final_test_acc = (final_test_pred == Y_test).float().mean().item()
    
    print(f"\nRun {run_num} Results:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Best Test  Accuracy: {final_test_acc:.4f} (Epoch {best_epoch})")
    print(f"  Final Train Accuracy: {final_train_acc:.4f}")
    
    return training_time, final_test_acc, best_epoch, train_cost_hist, test_acc_hist

# ============================================================
# Run multiple training sessions
# ============================================================

print("\nStarting multiple runs for consistency...\n")

for run in range(1, NUM_RUNS + 1):
    # Use different seeds for each run
    seed = 42 + run * 100
    training_time, test_acc, best_epoch, train_cost_hist, test_acc_hist = train_single_run(run, seed=seed)
    RUN_RESULTS.append({
        'run': run,
        'training_time': training_time,
        'test_accuracy': test_acc,
        'best_epoch': best_epoch,
        'train_cost_hist': train_cost_hist,
        'test_acc_hist': test_acc_hist
    })

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS ACROSS ALL RUNS")
print("=" * 70)

training_times = [r['training_time'] for r in RUN_RESULTS]
test_accuracies = [r['test_accuracy'] for r in RUN_RESULTS]
best_epochs = [r['best_epoch'] for r in RUN_RESULTS]

print(f"\nTraining Time (seconds):")
print(f"  Mean: {np.mean(training_times):.2f}")
print(f"  Std:  {np.std(training_times):.2f}")
print(f"  Min:  {np.min(training_times):.2f}")
print(f"  Max:  {np.max(training_times):.2f}")

print(f"\nTest Accuracy:")
print(f"  Mean: {np.mean(test_accuracies):.4f}")
print(f"  Std:  {np.std(test_accuracies):.4f}")
print(f"  Min:  {np.min(test_accuracies):.4f}")
print(f"  Max:  {np.max(test_accuracies):.4f}")

print(f"\nBest Epoch:")
print(f"  Mean: {np.mean(best_epochs):.1f}")
print(f"  Std:  {np.std(best_epochs):.1f}")
print(f"  Min:  {np.min(best_epochs)}")
print(f"  Max:  {np.max(best_epochs)}")

# ============================================================
# Create Problem2_Figures folder if it doesn't exist
# ============================================================
figures_dir = "Problem2_Figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"\nCreated directory: {figures_dir}")
else:
    print(f"\nUsing existing directory: {figures_dir}")

# ============================================================
# Plot results for each run
# ============================================================

for run_result in RUN_RESULTS:
    run_num = run_result['run']
    epochs_hist = np.arange(1, len(run_result['train_cost_hist']) + 1)
    
    # Train Cost plot for this run
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_hist, run_result['train_cost_hist'], "b")
    plt.xlabel("Epoch")
    plt.ylabel("Train Cost")
    plt.title(f"Train Cost Evolution - Run {run_num} (RMSProp)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    train_cost_path = os.path.join(figures_dir, f"Train_cost_{run_num}_RMSProp.png")
    plt.savefig(train_cost_path, dpi=150)
    plt.close()
    print(f"Saved: {train_cost_path}")
    
    # Test Accuracy plot for this run
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_hist, run_result['test_acc_hist'], "r")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy Evolution - Run {run_num} (RMSProp)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    test_acc_path = os.path.join(figures_dir, f"Test_acc_{run_num}_RMSProp.png")
    plt.savefig(test_acc_path, dpi=150)
    plt.close()
    print(f"Saved: {test_acc_path}")
    
    # Combined Output plot for this run
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_hist, run_result['train_cost_hist'], "b")
    plt.xlabel("Epoch")
    plt.ylabel("Train Cost")
    plt.title(f"Train Cost Evolution - Run {run_num}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_hist, run_result['test_acc_hist'], "r")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy Evolution - Run {run_num}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, f"Output_{run_num}_RMSProp.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

print(f"\nAll figures saved to: {figures_dir}/")

