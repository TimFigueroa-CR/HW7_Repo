# Introduction to Artificial Intelligence
# MNIST dataset
# Shallow Neural Network
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import sklearn
import torch

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

n_nodes_l1 = 100
batch_size = 1024
learning_rate = 1e-3
num_epochs = 1000
eval_step = 1
early_stop_patience = 10

# Print the configuration
print(f"Num epochs: {num_epochs}  Batch size: {batch_size}  Learning rate: {learning_rate}")

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
# Create and initialize neural network
# ============================================================

# Note that the final layer is skipping the Softmax activation
# because the cost function contains a Softmax function inside
model = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, n_nodes_l1),
    torch.nn.ELU(),
    torch.nn.Linear(n_nodes_l1, num_classes)
)
model.to(device)
print(model)

# Initialize weights explicitly
torch.nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
torch.nn.init.xavier_normal_(model[2].weight)

# Use defaults for biases

# ============================================================
# Optimizer and Loss
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Optimizer: Adam with multinomial cross-entropy loss.  Learning rate: {learning_rate}")

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

        print(print_line)

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")    

# ============================================================
# Restore best model and final evaluation
# ============================================================

print("\nRestoring best model parameters (Test Accuracy = {:.4f} Epoch = {})".format(best_test_acc, best_epoch))
model.load_state_dict(best_state_dict)
model.eval()

with torch.no_grad():
    final_train_pred = torch.argmax(model(X), dim=1)
    final_test_pred = torch.argmax(model(X_test), dim=1)
    final_train_acc = (final_train_pred == Y).float().mean().item()
    final_test_acc = (final_test_pred == Y_test).float().mean().item()

print("\nFinal Evaluation (best model from epoch {}):".format(best_epoch))
print("Best Test  Accuracy: {:.4f}".format(final_test_acc))
print("     Train Accuracy: {:.4f}".format(final_train_acc))

# ============================================================
# Plot cost and accuracy evolution
# ============================================================
epochs_hist = np.arange(1, len(train_cost_hist) + 1, eval_step)
plt.plot(epochs_hist, train_cost_hist, "b")
plt.xlabel("Epoch")
plt.ylabel("Train Cost")
plt.title("Train Cost Evolution")

plt.figure()
plt.plot(epochs_hist, test_acc_hist, "r")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Evolution")
plt.show()