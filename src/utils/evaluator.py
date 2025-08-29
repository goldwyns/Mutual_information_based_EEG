# In src/utils/evaluator.py (CORRECTED AGAIN)
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# Corrected function signature: Added 'data_loader' and 'return_loss=False'
def evaluate_model(model, data_loader, device, num_classes, disable_tqdm=False, return_loss=False):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        # Use data_loader here for consistency
        for features, labels in tqdm(data_loader, desc="Evaluating", disable=disable_tqdm):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Generate classification report
    target_names = [f'Class {i}' for i in range(num_classes)]
    report_dict = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0, target_names=target_names)
    
    # Calculate average loss using data_loader length
    avg_loss = total_loss / len(data_loader)

    if return_loss:
        return accuracy, cm, report_dict, avg_loss
    else:
        return accuracy, cm, report_dict