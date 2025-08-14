# src/utils/feature_selection.py

import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import torch

def select_features_mutual_info(X, y, num_features_to_select):
    """
    Selects top features based on Mutual Information (MI) scores.

    Args:
        X (np.ndarray or torch.Tensor): The feature matrix (samples x features).
        y (np.ndarray or torch.Tensor): The target labels (samples).
        num_features_to_select (int): The number of top features to select.

    Returns:
        tuple: (X_selected, selected_feature_indices)
    """
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X

    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = y

    # Mutual Information can be sensitive to scale for continuous features,
    # but mutual_info_classif is generally robust as it estimates density.
    # Scaling can sometimes help stability but is not strictly necessary for MI.

    # Calculate MI scores
    mi_scores = mutual_info_classif(X_np, y_np, random_state=42)

    # Get indices of top features
    # Ensure num_features_to_select does not exceed total features
    n_features_actual = min(num_features_to_select, X_np.shape[1])
    
    # Get indices that would sort mi_scores in descending order
    sorted_indices = np.argsort(mi_scores)[::-1]
    selected_feature_indices = sorted_indices[:n_features_actual]

    X_selected = X_np[:, selected_feature_indices]
    
    print(f"Mutual Information: Selected {X_selected.shape[1]} features out of {X_np.shape[1]}.")
    print(f"Top MI scores (max {n_features_actual}): {mi_scores[selected_feature_indices]}")

    return X_selected, selected_feature_indices, mi_scores


def select_features_rfe(X, y, num_features_to_select, estimator_name='LogisticRegression'):
    """
    Selects features using Recursive Feature Elimination (RFE).

    Args:
        X (np.ndarray or torch.Tensor): The feature matrix (samples x features).
        y (np.ndarray or torch.Tensor): The target labels (samples).
        num_features_to_select (int): The number of features to select.
        estimator_name (str): The base estimator for RFE ('LogisticRegression', 'SVC', 'RandomForestClassifier').

    Returns:
        tuple: (X_selected, selected_feature_indices)
    """
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X

    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = y

    # Choose the base estimator
    if estimator_name == 'LogisticRegression':
        estimator = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000) # liblinear is good for small datasets, L1/L2
    elif estimator_name == 'SVC':
        estimator = SVC(kernel='linear', random_state=42, C=1.0) # Linear kernel for feature importance
    elif estimator_name == 'RandomForestClassifier':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}. Choose from 'LogisticRegression', 'SVC', 'RandomForestClassifier'.")

    # Scale features for estimators that are sensitive to scale (like Logistic Regression, SVC)
    # RandomForest is less sensitive but scaling doesn't hurt.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Ensure num_features_to_select does not exceed total features
    n_features_actual = min(num_features_to_select, X_scaled.shape[1])
    if n_features_actual == 0:
        print("RFE: num_features_to_select is 0 or less, returning all features.")
        return X_np, np.arange(X_np.shape[1]) # Return all features if selection is 0

    rfe = RFE(estimator=estimator, n_features_to_select=n_features_actual, step=1)
    
    # Handle cases where n_features_actual is greater than or equal to total features
    if n_features_actual >= X_scaled.shape[1]:
        print(f"RFE: num_features_to_select ({n_features_actual}) is >= total features ({X_scaled.shape[1]}). Skipping RFE, returning all features.")
        return X_np, np.arange(X_np.shape[1])


    rfe.fit(X_scaled, y_np)

    selected_feature_indices = np.where(rfe.support_)[0]
    X_selected = X_np[:, selected_feature_indices]

    print(f"RFE ({estimator_name}): Selected {X_selected.shape[1]} features out of {X_np.shape[1]}.")
    print(f"Selected feature ranks (lower is better): {rfe.ranking_[selected_feature_indices]}")

    return X_selected, selected_feature_indices, rfe.ranking_