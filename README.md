# 对话问答回避检测与跨场景迁移：基线复现、QNLI 迁移与数据效率分析

深圳大学《自然语言处理》期末大作业配套代码与结果。

姓名：刘名芝　　学号：2021270251

## 1. 环境

- 系统：Windows 11
- Python 3.13.5（Anaconda）
- 无 GPU，PyTorch 为 CPU 版 2.9.1+cpu
- 主要依赖见 `requirements.txt`

安装：

```
pip install -r requirements.txt
```

国内访问 HuggingFace 时设置镜像：

```
set HF_ENDPOINT=https://hf-mirror.com
```

## 2. 随机种子与数据

- 随机种子：42（Python、NumPy、`sklearn` 的 `random_state`）
- 主任务：`ailsntua/QEvasion`，使用官方划分，train 3448 / test 308
- 迁移任务：`nyu-mll/glue` 的 `qnli` 子集；训练集随机抽样 4000，验证集随机抽样 800（QNLI 的 test 无公开标签，故用 validation）
- 字段映射：`configs/qevasion.json`、`configs/qnli.json`

数据文件中的高层标签为 Clear Reply / Ambivalent / Clear Non-Reply。Ambivalent 对应论文中的 Ambivalent Reply。测试集 `evasion_label` 为空，错误分析时用 `annotator1/2/3` 的多数票还原细粒度标签。

## 3. 运行

在本目录执行：

```
python run_experiment.py        # 主实验：QEvasion 基线 + QNLI 迁移
python extend_data_efficiency.py  # 扩展实验一：QNLI 数据效率（10% / 30% / 100%）
python extend_seed_robustness.py  # 扩展实验二：数据效率多种子稳健性（3 种子 mean±std）
```

在数据与 MiniLM 权重已缓存的前提下，主实验约 2–3 分钟，两个扩展实验各约 2 分钟。Sentence-BERT 向量默认写入 `outputs/emb_sbert_*.npy`（本仓库未包含这两个可再生的二进制文件，首次运行会自动生成）。

## 4. 主要输出

| 文件 | 说明 |
|---|---|
| `outputs/metrics.json` | 主实验全部指标与泄漏统计 |
| `outputs/metrics.csv` | 主实验指标汇总表 |
| `outputs/predictions_tfidf.csv` | 主任务 TF-IDF 预测（id, gold, prediction, 各类概率） |
| `outputs/predictions_sbert.csv` | Sentence-BERT 预测 |
| `outputs/predictions_sbert_len.csv` | Sentence-BERT + 长度特征预测 |
| `outputs/predictions_qnli.csv` | QNLI 适配后预测 |
| `outputs/error_cases.csv` | 典型误判样本 |
| `outputs/fig_*.png` | 标签/长度分布与混淆矩阵 |
| `outputs/run_log.txt` | 主实验完整运行日志 |
| `outputs_ext/data_efficiency.json` | 扩展实验一全部指标（10% / 30% / 100%） |
| `outputs_ext/data_efficiency.csv` | 扩展实验一学习曲线数据 |
| `outputs_ext/fig_data_efficiency.png` | QNLI 数据效率学习曲线 |
| `outputs_ext/data_efficiency_seeds.json` | 扩展实验二多种子 mean±std 汇总 |
| `outputs_ext/data_efficiency_seeds.csv` | 扩展实验二逐种子逐档位结果 |
| `outputs_ext/fig_data_efficiency_seeds.png` | 数据效率三种子误差棒图 |

## 5. 代码结构

- `adapter.py`：按 JSON 配置把不同数据集统一成 `text_a / text_b / label / group / id`
- `run_experiment.py`：EDA、多数类基线、TF-IDF+LR、SBERT+LR、长度特征改进、QNLI 直接迁移与适配后训练
- `extend_data_efficiency.py`：数据效率扩展，在 QNLI 上按类别分层截取 10% / 30% / 100% 训练子集，比较 TF-IDF+LR 与 MiniLM+LR
- `extend_seed_robustness.py`：多种子稳健性扩展，用 3 个随机种子（42/43/44）重复数据效率实验，报告 mean±std
- `configs/`：主任务与迁移任务的字段、标签映射和抽样上限

## 6. 主要结果

| 实验 | Accuracy | Macro-F1 |
|---|---|---|
| QEvasion 多数类基线 | 0.669 | 0.267 |
| QEvasion TF-IDF + LR（最佳） | 0.532 | 0.449 |
| QNLI 直接迁移（冻结源域词表） | 0.546 | 0.546 |
| QNLI 适配后训练 | 0.575 | 0.573 |
| QNLI 10% 标注：TF-IDF / MiniLM | 0.552 / 0.500 | 0.551 / 0.498 |
| QNLI 100% 标注：TF-IDF / MiniLM | 0.575 / 0.600 | 0.573 / 0.600 |
| 数据效率 3 种子稳健性（Macro-F1 mean±std） | TF-IDF：0.548±0.004 / 0.555±0.007 / 0.573±0.000 | MiniLM：0.501±0.005 / 0.504±0.016 / 0.600±0.000 |
