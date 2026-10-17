# 机器学习入门示例

三个可以离线运行的 Python + scikit-learn 实验，覆盖数据切分、标准化、训练、评估、模型保存和 CSV 预测。数据使用库内置数据集或程序生成，无需下载。

| 任务 | 数据 | 模型 | 评估 |
| --- | --- | --- | --- |
| 分类 | Iris 鸢尾花，150 条、4 个特征 | 标准化 + 逻辑回归 | Accuracy、Macro F1、混淆矩阵、5 折验证 |
| 回归 | Diabetes，442 条、10 个特征 | 标准化 + Ridge 回归 | MAE、RMSE、R²、5 折验证 |
| 聚类 | 500 条二维合成数据，3 个中心 | 标准化 + K-Means | 测试集轮廓系数、ARI |

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

测试覆盖三个任务的基本效果、保存模型后的预测、CSV 特征重排、非法输入处理和固定种子复现。

2026-10-17 本地验证：Python 3.14.2，依赖版本见 `requirements.txt`，种子 42。三组自动测试通过，训练与 CSV 预测命令均成功。

| 任务 | 本次保留测试集结果 | 对照 |
| --- | --- | --- |
| 分类 | Accuracy 0.9333，Macro F1 0.9333 | 多数类基线 Accuracy 0.3333 |
| 回归 | RMSE 53.6263，R² 0.4572 | 均值基线 RMSE 73.2225 |
| 聚类 | ARI 1.0000，轮廓系数 0.8869 | 合成数据，簇数固定为 3 |

## 代码导航

- `train.py`：三种实验和命令行入口；可修改模型、参数或替换数据。
- `predict.py`：CSV 校验、按列名重排和批量预测。
- `tests/test_examples.py`：使用标准库 unittest，不需要额外测试依赖。

进一步可阅读 [KMeans 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)。
