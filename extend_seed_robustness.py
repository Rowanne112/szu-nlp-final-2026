# -*- coding: utf-8 -*-
"""期末大作业扩展实验二：数据效率的多种子稳健性。

在 10% / 30% / 100% 三档标注量下，用 3 个随机种子（42/43/44）分别
分层截取训练子集并重训 TF-IDF+LR 与 MiniLM+LR，报告 mean±std，
检验数据效率结论对抽样子集的敏感度。评测集固定为与主实验相同的
800 条 validation 抽样（seed 42）。

姓名：刘名芝  学号：2021270251
运行：python extend_seed_robustness.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "实验报告：对话问答回避检测与跨场景迁移分析"
if not (EXP / "adapter.py").exists():
    EXP = ROOT  # GitHub 仓库布局：adapter.py 与本脚本同目录
sys.path.insert(0, str(EXP))

from adapter import load_config, load_pair_dataset, pair_text  # noqa: E402

OUT = ROOT / "outputs_ext"
LABELS_T = ["entailment", "not_entailment"]
FRACTIONS = [0.1, 0.3, 1.0]
SEEDS = [42, 43, 44]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_plot():
    sns.set_theme(style="whitegrid")
    from matplotlib import font_manager
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=LABELS_T, zero_division=0
    )
    return float(acc), float(mf)


def stratified_subsample(df, frac, seed):
    if frac >= 1.0:
        return df.reset_index(drop=True)
    parts = []
    for lab, g in df.groupby("label"):
        n = max(2, int(round(len(g) * frac)))
        parts.append(g.sample(n=n, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def main():
    setup_plot()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    cfg = load_config(EXP / "configs" / "qnli.json")
    train_df, test_df = load_pair_dataset(cfg, seed=42)
    x_te = pair_text(test_df["text_a"], test_df["text_b"])
    y_te = test_df["label"].tolist()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    x_tr_all = pair_text(train_df["text_a"], train_df["text_b"])
    emb_tr = model.encode(x_tr_all, batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True)
    emb_te = model.encode(x_te, batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True)
    print("emb", emb_tr.shape, emb_te.shape, flush=True)

    rows = []
    for frac in FRACTIONS:
        for seed in SEEDS:
            set_seed(seed)
            sub = stratified_subsample(train_df, frac, seed)
            x_tr = pair_text(sub["text_a"], sub["text_b"])
            y_tr = sub["label"].tolist()

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=50000, sublinear_tf=True)
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000, random_state=seed)
            clf.fit(vec.fit_transform(x_tr), y_tr)
            acc, f1 = metrics(y_te, clf.predict(vec.transform(x_te)))
            rows.append({"model": "TF-IDF + LR", "fraction": frac,
                         "seed": seed, "n_train": len(sub),
                         "accuracy": acc, "macro_f1": f1})

            idx = sub.index.to_numpy()
            clf2 = LogisticRegression(C=1.0, class_weight="balanced",
                                      max_iter=2000, random_state=seed)
            clf2.fit(emb_tr[idx], y_tr)
            acc2, f12 = metrics(y_te, clf2.predict(emb_te))
            rows.append({"model": "MiniLM + LR", "fraction": frac,
                         "seed": seed, "n_train": len(sub),
                         "accuracy": acc2, "macro_f1": f12})
            print("frac=%.2f seed=%d tfidf f1=%.3f sbert f1=%.3f"
                  % (frac, seed, f1, f12), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "data_efficiency_seeds.csv", index=False,
              encoding="utf-8-sig")

    summ = (df.groupby(["model", "fraction"])[["accuracy", "macro_f1"]]
              .agg(["mean", "std"]).reset_index())
    summary = {}
    for (mname, frac), g in df.groupby(["model", "fraction"]):
        summary[f"{mname}|{frac}"] = {
            "acc_mean": float(g["accuracy"].mean()),
            "acc_std": float(g["accuracy"].std(ddof=1)),
            "f1_mean": float(g["macro_f1"].mean()),
            "f1_std": float(g["macro_f1"].std(ddof=1)),
            "runs": int(len(g)),
        }
    with open(OUT / "data_efficiency_seeds.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "fractions": FRACTIONS,
                   "elapsed_sec": round(time.time() - t0, 2),
                   "summary": summary}, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mname, marker in [("TF-IDF + LR", "o"), ("MiniLM + LR", "s")]:
        d = df[df["model"] == mname]
        g = d.groupby("fraction")["macro_f1"]
        xs = [int(x * 100) for x in sorted(d["fraction"].unique())]
        mean = g.mean().values
        std = g.std(ddof=1).values
        ax.errorbar(xs, mean, yerr=std, marker=marker, capsize=4,
                    linewidth=1.8, label=mname)
    ax.set_xlabel("训练数据比例（%）")
    ax.set_ylabel("Macro-F1")
    ax.set_title("QNLI 数据效率：3 种子 mean±std")
    ax.set_xticks([10, 30, 100])
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_data_efficiency_seeds.png", dpi=160)
    print("saved", OUT / "fig_data_efficiency_seeds.png")
    print("elapsed", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
