# -*- coding: utf-8 -*-
"""期末大作业扩展实验：迁移数据（QNLI）上的数据效率对比。

比较 TF-IDF+LR 与冻结 MiniLM 句向量+LR 在 10% / 30% / 100%
标注训练数据下的 Accuracy 与 Macro-F1。训练子样本从与实验阶段
完全相同的 4000 条训练抽样中按类别分层截取，评测集固定为同一
800 条 validation 抽样，保证与主实验可比。

姓名：刘名芝  学号：2021270251
运行：python extend_data_efficiency.py
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

SEED = 42
ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "实验报告：对话问答回避检测与跨场景迁移分析"
if not (EXP / "adapter.py").exists():
    EXP = ROOT  # GitHub 仓库布局：adapter.py 与本脚本同目录
sys.path.insert(0, str(EXP))

from adapter import load_config, load_pair_dataset, pair_text  # noqa: E402

OUT = ROOT / "outputs_ext"
LABELS_T = ["entailment", "not_entailment"]
FRACTIONS = [0.1, 0.3, 1.0]


def set_seed(seed=SEED):
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
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_T, zero_division=0
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=LABELS_T, zero_division=0
    )
    per = {}
    for i, lab in enumerate(LABELS_T):
        per[lab] = {"support": int(sup[i]), "precision": float(p[i]),
                    "recall": float(r[i]), "f1": float(f[i])}
    return {"accuracy": float(acc), "macro_p": float(mp),
            "macro_r": float(mr), "macro_f1": float(mf), "per_class": per}


def stratified_subsample(df, frac, seed):
    if frac >= 1.0:
        return df.reset_index(drop=True)
    parts = []
    for lab, g in df.groupby("label"):
        n = max(2, int(round(len(g) * frac)))
        parts.append(g.sample(n=n, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def train_tfidf(x_tr, y_tr, x_te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                          sublinear_tf=True)
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, random_state=SEED)
    Xtr = vec.fit_transform(x_tr)
    Xte = vec.transform(x_te)
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


def main():
    set_seed()
    setup_plot()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    cfg = load_config(EXP / "configs" / "qnli.json")
    train_df, test_df = load_pair_dataset(cfg, seed=SEED)
    print("train", len(train_df), "test", len(test_df))
    print(train_df["label"].value_counts().to_dict())

    x_te = pair_text(test_df["text_a"], test_df["text_b"])
    y_te = test_df["label"].tolist()

    # SBERT 句向量只编码一次（全量 4000 训练 + 800 评测）
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    x_tr_all = pair_text(train_df["text_a"], train_df["text_b"])
    emb_tr = model.encode(x_tr_all, batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True)
    emb_te = model.encode(x_te, batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True)
    print("emb", emb_tr.shape, emb_te.shape)

    results = {}
    rows = []
    for frac in FRACTIONS:
        sub = stratified_subsample(train_df, frac, SEED)
        x_tr = pair_text(sub["text_a"], sub["text_b"])
        y_tr = sub["label"].tolist()
        n = len(sub)
        print("frac", frac, "n", n, sub["label"].value_counts().to_dict())

        pred = train_tfidf(x_tr, y_tr, x_te)
        m = metrics(y_te, pred)
        results[f"tfidf_{int(frac*100)}"] = {"n_train": n, **m}
        rows.append({"model": "TF-IDF + LR", "fraction": frac,
                     "n_train": n, "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"]})

        idx = sub.index.to_numpy()
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, random_state=SEED)
        clf.fit(emb_tr[idx], y_tr)
        pred2 = clf.predict(emb_te)
        m2 = metrics(y_te, pred2)
        results[f"sbert_{int(frac*100)}"] = {"n_train": n, **m2}
        rows.append({"model": "MiniLM + LR", "fraction": frac,
                     "n_train": n, "accuracy": m2["accuracy"],
                     "macro_f1": m2["macro_f1"]})
        print("  tfidf acc=%.3f f1=%.3f | sbert acc=%.3f f1=%.3f"
              % (m["accuracy"], m["macro_f1"], m2["accuracy"], m2["macro_f1"]))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "data_efficiency.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "data_efficiency.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "elapsed_sec": round(time.time() - t0, 2),
                   "fractions": FRACTIONS, "results": results}, f,
                  ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, marker in [("TF-IDF + LR", "o"), ("MiniLM + LR", "s")]:
        d = df[df["model"] == name].sort_values("fraction")
        xs = [int(x * 100) for x in d["fraction"]]
        axes[0].plot(xs, d["accuracy"], marker=marker, label=name)
        axes[1].plot(xs, d["macro_f1"], marker=marker, label=name)
    for ax, ttl in zip(axes, ["Accuracy", "Macro-F1"]):
        ax.set_xlabel("训练数据比例（%）")
        ax.set_ylabel(ttl)
        ax.set_title(f"QNLI 数据效率：{ttl}")
        ax.set_xticks([10, 30, 100])
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_data_efficiency.png", dpi=160)
    print("saved", OUT / "fig_data_efficiency.png")
    print("elapsed", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
