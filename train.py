"""可离线运行的分类、线性/非线性回归和聚类入门实验。"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_iris, make_blobs
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASKS = ("classification", "regression", "clustering", "nonlinear_regression")


def train(task, output_dir, seed=42):
    """训练模型并保存完整 Pipeline、指标和可用于预测的示例 CSV。"""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if task == "nonlinear_regression":
        from nonlinear_regression import train_nonlinear

        return train_nonlinear(output_dir, seed)

    target_names = None
    if task == "classification":
        dataset = load_iris()
        X, y = dataset.data, dataset.target
        feature_names = list(dataset.feature_names)
        target_names = dataset.target_names.tolist()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scoring = "accuracy"
        baseline = DummyClassifier(strategy="most_frequent")
    elif task == "regression":
        dataset = load_diabetes(scaled=False)
        X, y = dataset.data, dataset.target
        feature_names = list(dataset.feature_names)
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        scoring = "r2"
        baseline = DummyRegressor(strategy="mean")
    else:
        # 合成数据的标签仅用于最终评估，绝不作为 KMeans 的训练输入。
        X, y = make_blobs(
            n_samples=500, centers=3, n_features=2,
            cluster_std=0.8, random_state=seed,
        )
        feature_names = ["feature_0", "feature_1"]
        model = make_pipeline(
            StandardScaler(), KMeans(n_clusters=3, n_init=10, random_state=seed),
        )

    # 先切分再拟合；Pipeline 保证标准化只从训练数据学习参数。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed,
        stratify=y if task == "classification" else None,
    )
    metrics = {
        "task": task,
        "seed": seed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "joblib": joblib.__version__},
    }
    if task != "clustering":
        # 交叉验证只使用训练集，每折独立拟合标准化步骤。
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        metrics["cv"] = {
            "scoring": scoring,
            "fold_scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
        model.fit(X_train, y_train)
        baseline.fit(X_train, y_train)
        baseline_predictions = baseline.predict(X_test)
    else:
        model.fit(X_train)

    predictions = model.predict(X_test)
    if task == "classification":
        metrics.update(
            accuracy=float(accuracy_score(y_test, predictions)),
            macro_f1=float(f1_score(y_test, predictions, average="macro")),
            confusion_matrix=confusion_matrix(
                y_test, predictions, labels=np.arange(len(target_names)),
            ).tolist(),
            target_names=target_names,
            baseline_accuracy=float(accuracy_score(y_test, baseline_predictions)),
        )
    elif task == "regression":
        metrics.update(
            mae=float(mean_absolute_error(y_test, predictions)),
            rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
            r2=float(r2_score(y_test, predictions)),
            baseline_rmse=float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
        )
    else:
        # 距离指标使用与模型一致的标准化空间。
        transformed = model.named_steps["standardscaler"].transform(X_test)
        labels_count = len(np.unique(predictions))
        metrics.update(
            adjusted_rand_index=float(adjusted_rand_score(y_test, predictions)),
            silhouette=(float(silhouette_score(transformed, predictions))
                        if 1 < labels_count < len(predictions) else None),
            test_cluster_counts={str(int(label)): int(count) for label, count in
                                 zip(*np.unique(predictions, return_counts=True))},
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "task": task, "feature_names": feature_names,
                 "target_names": target_names, "versions": metrics["versions"]},
                output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "sample_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(feature_names)
        writer.writerows(X_test[:5])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        metrics = train(task, args.output_dir / task, args.seed)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Saved to {args.output_dir / task}")


if __name__ == "__main__":
    main()
