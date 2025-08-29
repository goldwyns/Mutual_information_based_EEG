# In src/utils/trainer.py

import torch
from tqdm import tqdm
import copy # Import copy module for deepcopy
# Assuming evaluate_model is in src/utils/evaluator, you can import it here
# or pass it as an argument if you prefer
from src.utils.evaluator import evaluate_model
from sklearn.metrics import accuracy_score # Needed for calculating training accuracy

def train_model(model, train_loader, optimizer, criterion, device, num_epochs,
                val_loader=None, patience=None, min_delta=0.001, monitor_metric="val_accuracy",
                disable_tqdm=False):
    """
    Trains the SNN model with optional early stopping based on a validation set.
    Args:
        model: The SNN model to train.
        train_loader: DataLoader for the training set.
        optimizer: Optimizer for the model.
        criterion: Loss function.
        device: Device to train on ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        val_loader: DataLoader for the validation set (optional, for early stopping).
        patience (int): Number of epochs to wait for improvement before stopping (for early stopping).
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        monitor_metric (str): Metric to monitor for early stopping ('val_loss' or 'val_accuracy').
        disable_tqdm (bool): If True, disables tqdm progress bars for the batch loop.
    Returns:
        tuple: (best_model_state_dict, train_loss_history, train_acc_history, val_loss_history, val_acc_history)
               - best_model_state_dict: The state_dict of the best model found during training
                                        (if early stopping is active and improvement found),
                                        otherwise the state_dict of the model at the end of training.
               - train_loss_history (list): List of average training losses per epoch.
               - train_acc_history (list): List of training accuracies per epoch.
               - val_loss_history (list): List of average validation losses per epoch.
               - val_acc_history (list): List of validation accuracies per epoch.
    """
    model.train() # Set the model to training mode

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    # Initialize for early stopping
    # If monitoring accuracy, higher is better, so start with negative infinity
    # If monitoring loss, lower is better, so start with positive infinity
    best_val_metric = -float('inf') if "accuracy" in monitor_metric else float('inf')
    best_model_state_dict = copy.deepcopy(model.state_dict()) # Save initial state as best
    epochs_no_improve = 0
    early_stop_triggered = False

    # Outer tqdm loop for epochs
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", disable=disable_tqdm):
        model.train() # Ensure model is in train mode at start of each epoch
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        
        # Inner tqdm loop for batches (can be disabled)
        batch_loop = tqdm(train_loader, leave=False, disable=disable_tqdm, desc=f"Epoch {epoch+1}/{num_epochs} Batch")
        for data, targets in batch_loop: # Removed enumerate, as batch_idx not used
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # For calculating training accuracy
            predicted_labels = torch.argmax(outputs, dim=1)
            all_train_preds.extend(predicted_labels.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())

            # Update progress bar for batches
            batch_loop.set_postfix(batch_loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_targets, all_train_preds) # Calculate actual train accuracy

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)
        
        # --- Early Stopping Logic ---
        if val_loader is not None and patience is not None:
            model.eval() # Set model to evaluation mode for validation
            
            # Ensure evaluate_model returns loss as well
            # This requires the modification to evaluator.py as discussed previously (Action 2)
            val_accuracy, _, _, val_loss = evaluate_model(model, val_loader, device,
                                                          num_classes=model.num_classes, # Assuming model has num_classes attribute
                                                          disable_tqdm=True, return_loss=True)
            
            val_loss_history.append(val_loss)
            val_acc_history.append(val_accuracy)
            
            current_monitor_score = val_accuracy if "accuracy" in monitor_metric else val_loss
            
            improved = False
            if "accuracy" in monitor_metric: # Monitor accuracy (higher is better)
                if current_monitor_score > best_val_metric + min_delta:
                    best_val_metric = current_monitor_score
                    best_model_state_dict = copy.deepcopy(model.state_dict()) # Save the best model state
                    epochs_no_improve = 0
                    improved = True
            elif "loss" in monitor_metric: # Monitor loss (lower is better)
                if current_monitor_score < best_val_metric - min_delta:
                    best_val_metric = current_monitor_score
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    improved = True
                
            if not improved:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. No improvement in {monitor_metric} for {patience} epochs.")
                    early_stop_triggered = True
                    break # Exit training loop

            model.train() # Set model back to train mode

        # Print epoch summary
        val_info_str = ""
        if val_loader is not None and patience is not None:
            val_info_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}{val_info_str}, No Improve Epochs: {epochs_no_improve}/{patience}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

    # --- FINAL RETURN STATEMENT ---
    # Always return the best model state found, and the histories
    return best_model_state_dict, train_loss_history, train_acc_history, val_loss_history, val_acc_history