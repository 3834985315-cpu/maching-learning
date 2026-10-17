"""加载自己训练的模型，从带表头的数值 CSV 生成预测。"""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np


def read_features(path, feature_names):
    """按模型的特征顺序重排列；拒绝缺列、重复列和无效数值。"""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if not headers or len(set(headers)) != len(headers):
            raise ValueError("CSV must have unique column headers")
        if set(headers) != set(feature_names):
            raise ValueError(f"CSV columns must be exactly: {feature_names}")
        indices = [headers.index(name) for name in feature_names]
        rows = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise ValueError(f"CSV row {line}: incorrect number of columns")
            try:
                values = [float(row[index]) for index in indices]
            except ValueError as exc:
                raise ValueError(f"CSV row {line}: all features must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"CSV row {line}: features must be finite")
            rows.append(values)
        if not rows:
            raise ValueError("CSV must contain at least one data row")
    return np.asarray(rows, dtype=float)


def predict(model_path, input_path, output_path):
    """输出 prediction；分类任务额外输出对应类别名称。"""
    model_path, input_path, output_path = map(Path, (model_path, input_path, output_path))
    if output_path.resolve() in {input_path.resolve(), model_path.resolve()}:
        raise ValueError("Output must not overwrite the input CSV or model")
    # joblib/pickle 可执行代码；仅加载自己训练或完全信任的模型。
    bundle = joblib.load(model_path)
    X = read_features(input_path, bundle["feature_names"])
    predictions = bundle["model"].predict(X)
    target_names = bundle.get("target_names")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prediction", "class_name"] if target_names else ["prediction"])
        for value in predictions:
            writer.writerow([int(value), target_names[int(value)]] if target_names else [value])
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    args = parser.parse_args()
    try:
        values = predict(args.model, args.input, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {len(values)} predictions to {args.output}")


if __name__ == "__main__":
    main()
