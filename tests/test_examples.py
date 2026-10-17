"""验证模型质量、持久化、特征顺序和错误输入处理。"""

import csv
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from predict import predict, read_features
from train import TASKS, train


class ExamplesTest(unittest.TestCase):
    def test_training_and_saved_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            for task in TASKS:
                with self.subTest(task=task):
                    directory = Path(temp) / task
                    metrics = train(task, directory)
                    if task == "classification":
                        self.assertGreater(metrics["accuracy"], 0.8)
                        self.assertGreater(metrics["accuracy"], metrics["baseline_accuracy"])
                    elif task == "regression":
                        self.assertGreater(metrics["r2"], 0.2)
                        self.assertLess(metrics["rmse"], metrics["baseline_rmse"])
                    else:
                        self.assertGreater(metrics["adjusted_rand_index"], 0.8)
                        self.assertGreater(metrics["silhouette"], 0.4)
                    bundle = joblib.load(directory / "model.joblib")
                    sample = directory / "sample_features.csv"
                    X = read_features(sample, bundle["feature_names"])
                    values = predict(directory / "model.joblib", sample, directory / "predictions.csv")
                    np.testing.assert_allclose(values, bundle["model"].predict(X))
                    with (directory / "predictions.csv").open(newline="") as handle:
                        self.assertEqual(len(list(csv.reader(handle))), 6)
                    # 特征列重排后，预测应当一致。
                    with sample.open(newline="") as handle:
                        rows = list(csv.reader(handle))
                    reordered = directory / "reordered.csv"
                    with reordered.open("w", newline="") as handle:
                        csv.writer(handle).writerows([row[::-1] for row in rows])
                    np.testing.assert_allclose(
                        predict(directory / "model.joblib", reordered, directory / "reordered_predictions.csv"),
                        values,
                    )
                    with self.assertRaisesRegex(ValueError, "overwrite"):
                        predict(directory / "model.joblib", sample, sample)

    def test_invalid_csv(self):
        cases = ["", "a,a\n1,2\n", "a\n1\n", "a,b\n", "a,b\n1\n",
                 "a,b\n1,2,3\n", "a,b\nhello,2\n", "a,b\nnan,2\n", "a,b\n1,inf\n"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_features(path, ["a", "b"])

    def test_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            for task in TASKS:
                with self.subTest(task=task):
                    first = train(task, Path(temp) / "first", seed=7)
                    second = train(task, Path(temp) / "second", seed=7)
                    self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
