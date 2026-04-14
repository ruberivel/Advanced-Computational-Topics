import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import pandas as pd

def load_hep_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)

    if "source_file" not in df.columns:
        raise ValueError("Expected a 'source_file' column.")
    if "weight" not in df.columns:
        raise ValueError("Expected a 'weight' column.")

    source = df["source_file"].astype(str)

    # Signal definition:
    # - anything containing 'ggh125'
    # - VBFH125_WW2lep.csv
    is_ggh = source.str.contains("ggh125", case=False, na=False)
    is_vbf = source.str.contains("VBFH125_WW2lep", case=False, na=False, regex=True)

    y = (is_ggh | is_vbf).astype(int)

    weights = pd.to_numeric(df["weight"], errors="coerce").clip(lower=1e-6)

    X = df.drop(columns=["source_file", "weight"]).apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in columns: {bad_cols[:10]}")

    if weights.isna().any():
        raise ValueError("NaN values found in weight column.")

    return X, y, weights


def choose_threshold_youden(y_true: np.ndarray, y_score: np.ndarray, sample_weight=None) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(j[idx])


def make_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost Higgs-vs-background classifier.")
    parser.add_argument("--train", default="C:\\Users\\revel\\Downloads\\New folder (2)\\train_chunk_fixed_filtered.csv", help="Path to training CSV")
    parser.add_argument("--test", default="C:\\Users\\revel\\Downloads\\New folder (2)\\test_chunk_fixed_filtered.csv", help="Path to test CSV")
    parser.add_argument("--output-dir", default="xgb_outputs", help="Directory for outputs")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    outdir = make_output_dir(args.output_dir)

    print("Loading training data...")
    X_train, y_train, w_train = load_hep_csv(args.train)
    print("Loading test data...")
    X_test, y_test, w_test = load_hep_csv(args.test)

    if list(X_train.columns) != list(X_test.columns):
        missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
        missing_in_train = [c for c in X_test.columns if c not in X_train.columns]
        raise ValueError(
            "Train/test feature mismatch. "
            f"Missing in test: {missing_in_test[:10]}; Missing in train: {missing_in_train[:10]}"
        )

    n_signal = int(y_train.sum())
    n_background = int((1 - y_train).sum())
    if n_signal == 0:
        raise ValueError("No ggH125 signal events found in the training labels.")

    scale_pos_weight = n_background / n_signal

    print("\nTraining summary")
    print("----------------")
    print(f"Train shape         : {X_train.shape}")
    print(f"Test shape          : {X_test.shape}")
    print(f"Signal train events : {n_signal}")
    print(f"Bkg train events    : {n_background}")
    print(f"scale_pos_weight    : {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, sample_weight=w_train)

    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score, sample_weight=w_test)
    roc_auc = auc(fpr, tpr)
    threshold, youden_j = choose_threshold_youden(y_test.to_numpy(), y_score, sample_weight=w_test)
    y_pred = (y_score >= threshold).astype(int)

    print("\nTest performance")
    print("----------------")
    print(f"ROC AUC            : {roc_auc:.6f}")
    print(f"Best threshold (J) : {threshold:.6f}")
    print(f"Youden J           : {youden_j:.6f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save predictions
    pred_df = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_score": y_score,
        "y_pred": y_pred,
        "event_weight": w_test.to_numpy(),
    })
    pred_df.to_csv(outdir / "test_predictions.csv", index=False)

    # Save model
    joblib.dump(model, outdir / "xgboost_model.joblib")

    # Save feature importances
    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    importance.sort_values(ascending=False).to_csv(outdir / "feature_importance.csv", header=["importance"])




    roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "thresholds": thresholds
    })

    roc_df.to_csv(outdir / "roc_data_bsc.csv", index=False)




    # ROC curve plot
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost ROC: ggH125 vs background")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close()

    # Score distributions
    sig = y_test.to_numpy() == 1
    bkg = y_test.to_numpy() == 0

    plt.figure(figsize=(7, 5))
    plt.hist(y_score[bkg], bins=50, histtype="step", density=True, label="background")
    plt.hist(y_score[sig], bins=50, histtype="step", density=True, label="ggH125")
    plt.axvline(threshold, linestyle="--", label=f"threshold = {threshold:.3f}")
    plt.xlabel("XGBoost score")
    plt.ylabel("Density")
    plt.title("Classifier score distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "score_distributions.png", dpi=150)
    plt.close()

    # Top-20 feature importance plot
    top_imp = importance.sort_values(ascending=True).tail(20)
    plt.figure(figsize=(8, 6))
    top_imp.plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top 20 XGBoost feature importances")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance_top20.png", dpi=150)
    plt.close()

    # Save short text summary
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("XGBoost Higgs classifier summary\n")
        f.write("===============================\n\n")
        f.write(f"Train file: {args.train}\n")
        f.write(f"Test file : {args.test}\n\n")
        f.write(f"Train shape         : {X_train.shape}\n")
        f.write(f"Test shape          : {X_test.shape}\n")
        f.write(f"Signal train events : {n_signal}\n")
        f.write(f"Bkg train events    : {n_background}\n")
        f.write(f"scale_pos_weight    : {scale_pos_weight:.6f}\n\n")
        f.write(f"ROC AUC             : {roc_auc:.6f}\n")
        f.write(f"Best threshold (J)  : {threshold:.6f}\n")
        f.write(f"Youden J            : {youden_j:.6f}\n")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Files written:")
    for p in sorted(outdir.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()


