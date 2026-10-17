# 机器学习入门示例

四个可以离线运行的 Python + scikit-learn 实验，覆盖数据切分、标准化、训练、评估、模型保存和 CSV 预测。数据使用库内置数据集或程序生成，无需下载。

| 任务 | 数据 | 模型 | 评估 |
| --- | --- | --- | --- |
| 分类 | Iris 鸢尾花，150 条、4 个特征 | 标准化 + 逻辑回归 | Accuracy、Macro F1、混淆矩阵、5 折验证 |
| 回归 | Diabetes，442 条、10 个特征 | 标准化 + Ridge 回归 | MAE、RMSE、R²、5 折验证 |
| 聚类 | 500 条二维合成数据，3 个中心 | 标准化 + K-Means | 测试集轮廓系数、ARI |
| 非线性回归 | 400 条带噪声三次曲线数据 | 多项式 Ridge、RBF-SVR、随机森林，对比线性 Ridge | MAE、RMSE、R²、5 折验证 |

## 安装与运行

使用 Python 3.11 或更新版本；本项目在 Python 3.14.2 上验证。先克隆仓库，进入仓库目录后执行：

```bash
git clone https://github.com/3834985315-cpu/maching-learning.git
cd maching-learning
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux 使用：source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --task all
```

也可以分别运行，或改变随机种子与输出目录：

```bash
python train.py --task classification --seed 42
python train.py --task regression
python train.py --task clustering --output-dir artifacts
python train.py --task nonlinear_regression
```

每次运行会在 `artifacts/<任务名>/` 下生成：

- `metrics.json`：测试指标、交叉验证结果、随机种子、依赖版本。
- `model.joblib`：完整预处理和模型，以及特征名称。
- `sample_features.csv`：5 行测试集特征，用于演示预测。

重复使用同一输出目录会替换上次该任务的产物。模型与生成文件已被 `.gitignore` 排除。

## 用 CSV 预测

```bash
python predict.py --model artifacts/classification/model.joblib --input artifacts/classification/sample_features.csv --output artifacts/classification/predictions.csv
```

回归和聚类同样适用，将路径中的 `classification` 换成对应任务名即可。

## 非线性回归

运行命令：

```bash
python train.py --task nonlinear_regression --seed 42
# 也可直接运行：python nonlinear_regression.py
python predict.py --model artifacts/nonlinear_regression/model.joblib --input artifacts/nonlinear_regression/sample_features.csv --output artifacts/nonlinear_regression/predictions.csv
```

实验生成 `x ∈ [-3, 3]` 上的曲线 `y = 0.6x³ - 1.5x² + x + ε`，其中噪声 `ε` 来自标准差为 0.8 的零均值正态分布。同一份 80% 训练集 / 20% 测试集用于比较以下模型：

| 模型名称 | 思路 | 可尝试调整的参数 |
| --- | --- | --- |
| `linear_ridge` | 用直线拟合，作为欠拟合对照 | `alpha` 正则化强度 |
| `polynomial_ridge` | 构造 x、x²、x³ 特征，再做 Ridge 回归 | `degree` 多项式次数、`alpha` |
| `rbf_svr` | 用 RBF 核学习平滑的非线性关系 | `C`、`epsilon`、`gamma` |
| `random_forest` | 多棵决策树的预测取平均，学习分段关系 | `n_estimators`、`min_samples_leaf` |

多项式回归关于输入 x 是非线性的，但关于扩展后的特征系数仍是线性的。次数过高可能过拟合。SVR 与多项式模型将预处理放在 Pipeline 中；随机森林无需标准化。参数与实现集中在 `nonlinear_regression.py` 的 `build_models()` 中。

四个模型使用相同的训练集五折划分，以平均验证 RMSE 最小者作为保存的主模型。选择完成后才计算各模型的测试指标。测试指标用于最终比较；若据此反复修改模型，应另外保留新的测试集。已知三次关系属于本教学实验的设计，不能据此认为多项式模型总是优于其他模型。

除通用的三个产物外，`artifacts/nonlinear_regression/` 还会生成：

- 四个具名 `*.joblib` 文件，可分别传给 `predict.py --model`；`model.joblib` 是按训练集交叉验证选出的模型。
- `dataset.csv`：x、带噪声的 y 和 train/test 标记，便于自行绘图。
- `curve_predictions.csv`：201 个有序 x、无噪声真值和四个模型的预测，便于画拟合曲线。此文件带有额外列，预测时请使用只有 x 列的 `sample_features.csv`。
- `metrics.json`：各模型的验证 RMSE 均值、标准差、逐折结果和测试 MAE / RMSE / R²。

自己的预测 CSV 只需一个名为 `x` 的数值列。示例仅在训练区间附近评估；超出该区间的外推可能不可靠，尤其是树模型。

2026-10-17 本地验证，种子 42：四组自动测试及命令行训练、CSV 预测均通过。按训练集交叉验证选中 `polynomial_ridge`，以下是保留的 80 条测试样本结果：

| 模型 | 测试 RMSE，越低越好 | 测试 R²，越高越好 |
| --- | --- | --- |
| 线性 Ridge | 4.5940 | 0.7016 |
| 三次多项式 Ridge | 0.8363 | 0.9901 |
| RBF-SVR | 0.8745 | 0.9892 |
| 随机森林 | 0.9794 | 0.9864 |

自己的 CSV 必须使用训练产物 `sample_features.csv` 中的全部表头，每列都为有限数值，不带目标标签列。列顺序可以改变，程序会按特征名重排；缺失、重复、多余的列或非法数值会报错。分类输出类别编号和类别名称；回归输出预测数值；聚类输出簇编号，簇编号本身没有固定语义。

Iris 特征的单位为厘米；回归使用 `load_diabetes(scaled=False)` 的原始特征，由 Pipeline 统一标准化。真实数据必须与训练数据保持相同的特征含义和单位。

仅加载自己生成或完全信任的 `joblib` 文件；其底层反序列化可能执行代码。保存和加载模型请使用相同的依赖版本。

## 学习要点

1. 先按 80% / 20% 切分，再训练；分类采用分层抽样。
2. 标准化放进 Pipeline；只从训练数据学习均值和方差，每个交叉验证折也独立拟合。参见 [scikit-learn 数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。
3. 分类与回归的 5 折验证仅使用训练集，保留测试集用于最终评估；同时输出简单基线，帮助判断模型是否学到了信息。
4. 聚类的生成标签只用于计算 ARI，训练不接触标签；轮廓系数使用标准化后的测试数据。这里的簇数 3 来自合成数据的已知设计，可尝试修改它并比较结果。
5. 固定种子方便复现实验；不同版本、平台可能有浮点误差。指标仅用于这些教学数据，不代表真实业务效果。Diabetes 示例只作数值回归练习。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖四个任务的基本效果、保存模型后的预测、CSV 特征重排、非法输入处理和固定种子复现；另检查三个非线性模型的预测效果和训练集标准化统计量。

2026-10-17 初始三个任务本地验证：Python 3.14.2，依赖版本见 `requirements.txt`，种子 42。训练与 CSV 预测命令均成功。

| 任务 | 本次保留测试集结果 | 对照 |
| --- | --- | --- |
| 分类 | Accuracy 0.9333，Macro F1 0.9333 | 多数类基线 Accuracy 0.3333 |
| 回归 | RMSE 53.6263，R² 0.4572 | 均值基线 RMSE 73.2225 |
| 聚类 | ARI 1.0000，轮廓系数 0.8869 | 合成数据，簇数固定为 3 |

## 代码导航

- `train.py`：四种实验的统一命令行入口；可修改模型、参数或替换数据。
- `nonlinear_regression.py`：非线性数据生成、模型对比、按交叉验证选择模型和曲线预测。
- `predict.py`：CSV 校验、按列名重排和批量预测。
- `tests/test_examples.py`：使用标准库 unittest，不需要额外测试依赖。

进一步可阅读 [KMeans 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)。

非线性回归相关文档：[PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)、[SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)、[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)。
