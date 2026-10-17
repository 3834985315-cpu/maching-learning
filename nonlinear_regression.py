"""比较三种非线性回归方法和线性基线，仅用训练集交叉验证选择模型。"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


def true_curve(x):
    """已知的非线性关系，便于观察模型是否学到了弯曲趋势。"""
    return 0.6 * x**3 - 1.5 * x**2 + x


def build_models(seed):
    """超参数为教学预设值；可修改后观察训练集交叉验证误差。"""
    return {
        "linear_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # 加入 x² 和 x³ 后，模型可表达关于 x 的非线性关系。
        # 它关于扩展后的特征系数仍然是线性的。
        "polynomial_ridge": make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            StandardScaler(), Ridge(alpha=0.1),
        ),
        # RBF 核学习平滑曲线；特征尺度会影响核距离。
        "rbf_svr": make_pipeline(
            StandardScaler(), SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"),
        ),
        # 树通过分段拟合学习非线性关系，无需标准化。
        "random_forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1,
        ),
    }


def train_nonlinear(output_dir, seed=42):
    """训练、比较并保存所有模型；model.joblib 是按 CV 选出的模型。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(400, 1))
    y = true_curve(X[:, 0]) + rng.normal(0.0, 0.8, size=len(X))
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=seed,
    )
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    models = build_models(seed)
    comparisons = {}
    for name, model in models.items():
        # sklearn 以“越大越好”返回负 RMSE，此处取反还原正误差。
        errors = -cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error",
        )
        comparisons[name] = {
            "cv_rmse_folds": errors.tolist(),
            "cv_rmse_mean": float(errors.mean()),
            "cv_rmse_std": float(errors.std()),
        }
        model.fit(X_train, y_train)

    # 在访问测试集指标前选定模型，避免依据测试误差反复挑选。
    selected = min(comparisons, key=lambda name: comparisons[name]["cv_rmse_mean"])
    for name, model in models.items():
        values = model.predict(X_test)
        comparisons[name].update(
            test_mae=float(mean_absolute_error(y_test, values)),
            test_rmse=float(np.sqrt(mean_squared_error(y_test, values))),
            test_r2=float(r2_score(y_test, values)),
        )
    selected_metrics = comparisons[selected]
    versions = {"numpy": np.__version__, "scikit-learn": sklearn.__version__,
                "joblib": joblib.__version__}
    metrics = {
        "task": "nonlinear_regression", "seed": seed,
        "train_samples": len(X_train), "test_samples": len(X_test),
        "versions": versions, "selected_model": selected,
        "selection_criterion": "lowest mean training 5-fold CV RMSE",
        "models": comparisons,
        "mae": selected_metrics["test_mae"],
        "rmse": selected_metrics["test_rmse"],
        "r2": selected_metrics["test_r2"],
        "baseline_rmse": comparisons["linear_ridge"]["test_rmse"],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        bundle = {
            "model": model, "task": "nonlinear_regression", "model_name": name,
            "feature_names": ["x"], "target_names": None, "versions": versions,
        }
        joblib.dump(bundle, output_dir / f"{name}.joblib")
        if name == selected:
            joblib.dump(bundle, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "sample_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x"])
        writer.writerows(X_test[:5])

    # 保存带噪声观测及其所属集合，方便自行画图和检查训练/测试切分。
    train_set = set(train_indices.tolist())
    with (output_dir / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "split"])
        writer.writerows((float(x), float(target), "train" if i in train_set else "test")
                         for i, (x, target) in enumerate(zip(X[:, 0], y)))
    grid = np.linspace(-3.0, 3.0, 201).reshape(-1, 1)
    curve_values = [model.predict(grid) for model in models.values()]
    with (output_dir / "curve_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "true_y", *models])
        writer.writerows(zip(grid[:, 0], true_curve(grid[:, 0]), *curve_values))
    return metrics


if __name__ == "__main__":
    # 与其他任务使用相同的种子校验和命令行参数。
    import sys
    from train import main

    main(["--task", "nonlinear_regression", *sys.argv[1:]])
