"""End-to-End Real-World Machine Learning Pipeline (NumPy from Scratch).

Demonstrates the complete workflow:
1. Loading real-world benchmark datasets (Wisconsin Breast Cancer & Diabetes Progression).
2. Splitting into Train and Test partitions without data leakage.
3. Feature standardization using training set statistics.
4. Training Logistic Regression and Linear Regression with Gradient Descent.
5. Evaluating on unseen test data with rigorous metrics (Accuracy, Precision, Recall, F1, RMSE, R²).
6. Verifying equivalence against PyTorch on the real dataset.

Run directly:
    python chapters/01-the-neuron/real_world_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nnlib.data import train_test_split, StandardScaler, load_breast_cancer, load_diabetes
from nnlib.linear import LogisticRegression, LinearRegression
from nnlib.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from nnlib.device import get_device_name


def run_classification_pipeline():
    print("\n" + "=" * 70)
    print(" PART 1: BINARY CLASSIFICATION (WISCONSIN BREAST CANCER DATASET)")
    print("=" * 70)

    # 1. Load Real Dataset
    X, y, feature_names = load_breast_cancer()
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    malignant_pct = np.mean(y == 1.0) * 100.0
    print(f"Class distribution: {malignant_pct:.1f}% Malignant (1), {100.0 - malignant_pct:.1f}% Benign (0)")

    # 2. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Standardization (Preventing Data Leakage!)
    # CRITICAL: We compute mean and std ONLY on X_train.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features standardized (mean=0, std=1 on training statistics).")

    # 4. Train Logistic Regression Model
    model = LogisticRegression(in_features=X_train.shape[1])
    print("\nTraining Logistic Regression neuron with Gradient Descent...")
    model.fit(X_train_scaled, y_train, lr=0.1, epochs=500, verbose=False)
    print(f"Initial BCE loss: {model.loss_history[0]:.4f} -> Final BCE loss: {model.loss_history[-1]:.4f}")

    # 5. Evaluate on Unseen Test Set
    test_probs = model.predict_proba(X_test_scaled)
    test_preds = model.predict(X_test_scaled, threshold=0.5)

    print("\n" + classification_report(y_test, test_preds))

    # Pedagogical context on Recall vs Precision in medical diagnosis
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    print("\nMedical Domain Context:")
    print(f"  Recall ({rec * 100:.1f}%): A false negative means a cancerous tumor is missed.")
    print(f"  Precision ({prec * 100:.1f}%): A false positive means unnecessary follow-up biopsy.")
    assert accuracy_score(y_test, test_preds) > 0.90, "Expected >90% test accuracy on Breast Cancer!"
    print("  -> Classification Pipeline PASSED (Accuracy > 90%)")


def run_regression_pipeline():
    print("\n" + "=" * 70)
    print(" PART 2: CONTINUOUS REGRESSION (DIABETES PROGRESSION DATASET)")
    print("=" * 70)

    # 1. Load Real Dataset
    X, y, feature_names = load_diabetes()
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} clinical features.")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}] (mean: {y.mean():.1f}, std: {y.std():.1f})")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # 3. Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Baseline Comparison (Trivial mean predictor)
    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train))
    baseline_rmse = root_mean_squared_error(y_test, baseline_pred)
    print(f"\nBaseline (predicting target mean) RMSE: {baseline_rmse:.2f}")

    # 5. Train Linear Regression Model
    model = LinearRegression(in_features=X_train.shape[1])
    print("Training Linear Regression neuron with Gradient Descent...")
    model.fit(X_train_scaled, y_train, lr=0.05, epochs=1000, verbose=False)

    # 6. Evaluate on Test Set
    test_preds = model.predict(X_test_scaled)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    print(f"Model Test RMSE: {test_rmse:.2f} (Error reduced by {baseline_rmse - test_rmse:.2f} points)")
    print(f"Model Test R²  : {test_r2:.4f} (Explains ~{test_r2 * 100:.1f}% of variance on unseen test patients)")
    assert test_r2 > 0.40, "Linear regression should achieve >0.40 R² on diabetes benchmark!"
    print("  -> Regression Pipeline PASSED (R² > 0.40)")


def verify_against_pytorch_on_real_data():
    print("\n" + "=" * 70)
    print(" PART 3: VERIFYING OUR MODEL VS PYTORCH ON THE BREAST CANCER DATASET")
    print("=" * 70)

    X, y, _ = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, seed=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    N, D = X_train_scaled.shape

    # Set identical initial weights
    init_w = np.zeros((D, 1), dtype=np.float64)
    init_b = 0.0

    # 1. From-scratch model
    scratch_model = LogisticRegression(in_features=D)
    scratch_model.w = init_w.copy()
    scratch_model.b = init_b

    # 2. PyTorch model
    pt_linear = torch.nn.Linear(D, 1, dtype=torch.float64)
    with torch.no_grad():
        pt_linear.weight.zero_()
        pt_linear.bias.zero_()

    X_pt = torch.tensor(X_train_scaled, dtype=torch.float64)
    y_pt = torch.tensor(y_train, dtype=torch.float64)
    pt_optimizer = torch.optim.SGD(pt_linear.parameters(), lr=0.1)
    bce = torch.nn.BCELoss()

    # Train both for 20 epochs
    for epoch in range(20):
        # Scratch
        preds_s = scratch_model.forward(X_train_scaled)
        loss_s = scratch_model.loss_fn.forward(preds_s, y_train)
        gw, gb = scratch_model.backward(X_train_scaled, preds_s, y_train)
        scratch_model.w -= 0.1 * gw
        scratch_model.b -= 0.1 * gb

        # PyTorch
        pt_optimizer.zero_grad()
        preds_pt = torch.sigmoid(pt_linear(X_pt))
        loss_pt = bce(preds_pt, y_pt)
        loss_pt.backward()
        pt_optimizer.step()

    weight_diff = np.max(np.abs(scratch_model.w - pt_linear.weight.detach().numpy().T))
    bias_diff = abs(scratch_model.b - pt_linear.bias.item())
    loss_diff = abs(loss_s - loss_pt.item())

    print(f"Max Weight Difference: {weight_diff:.2e}")
    print(f"Bias Difference:       {bias_diff:.2e}")
    print(f"Final Loss Difference: {loss_diff:.2e}")

    assert weight_diff < 1e-7 and bias_diff < 1e-7 and loss_diff < 1e-7
    print("  -> Complete mathematical equivalence verified on real clinical data!")


if __name__ == "__main__":
    print(f"Running on: {get_device_name()}")
    run_classification_pipeline()
    run_regression_pipeline()
    verify_against_pytorch_on_real_data()
