# -*- coding: utf-8 -*-
"""对话问答回避检测与跨场景迁移：从读取数据到指标/预测文件。
姓名：刘名芝  学号：2021270251
运行：python run_experiment.py
"""
import json
import os
import random
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline

from adapter import load_config, load_pair_dataset, pair_text

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CFG_Q = ROOT / "configs" / "qevasion.json"
CFG_T = ROOT / "configs" / "qnli.json"

# 与数据文件一致：Ambivalent 而不是论文里的 Ambivalent Reply
LABELS_Q = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_plot():
    sns.set_theme(style="whitegrid")
    from matplotlib import font_manager
    for fp in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"


def word_len(s):
    return s.fillna("").astype(str).str.split().map(len)


def eval_split(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    per_class = {}
    for i, lab in enumerate(labels):
        per_class[lab] = {
            "support": int(sup[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
        }
    return {
        "accuracy": float(acc),
        "macro_p": float(macro_p),
        "macro_r": float(macro_r),
        "macro_f1": float(macro_f),
        "per_class": per_class,
    }


def save_preds(path, ids, gold, pred, proba, classes):
    rows = {"id": ids, "gold": gold, "prediction": pred}
    for j, c in enumerate(classes):
        rows[f"prob_{c}"] = proba[:, j]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrBr",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def majority_predict(y_train, n):
    maj = pd.Series(y_train).mode().iloc[0]
    return np.array([maj] * n), maj


def train_tfidf(texts_tr, y_tr, ngram=(1, 2), class_weight="balanced"):
    vec = TfidfVectorizer(
        ngram_range=ngram,
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        lowercase=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )
    pipe = make_pipeline(vec, clf)
    pipe.fit(texts_tr, y_tr)
    return pipe


def sbert_encode(texts, cache_path=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            emb = np.load(cache_path)
            return emb, int(emb.shape[1])
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.max_seq_length = 256
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = int(getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)())
    if cache_path is not None:
        np.save(cache_path, emb)
    return emb, dim


def train_lr(X, y, class_weight="balanced"):
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def run_eda(train_df, test_df):
    setup_plot()
    OUT.mkdir(exist_ok=True)

    def miss_rate(df, col):
        s = df[col] if col in df.columns else pd.Series([""] * len(df))
        return float(((s.isna()) | (s.astype(str).str.strip() == "") | (s.astype(str) == "None")).mean())

    stats = []
    for name, df in [("train", train_df), ("test", test_df)]:
        stats.append({
            "split": name,
            "n": int(len(df)),
            "missing_text_a": miss_rate(df, "text_a"),
            "missing_text_b": miss_rate(df, "text_b"),
            "missing_label": miss_rate(df, "label"),
            "avg_q_len": float(word_len(df["text_a"]).mean()),
            "avg_a_len": float(word_len(df["text_b"]).mean()),
            "median_a_len": float(word_len(df["text_b"]).median()),
        })
    pd.DataFrame(stats).to_csv(OUT / "eda_split_stats.csv", index=False, encoding="utf-8-sig")

    # 标签分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, df, title in [(axes[0], train_df, "训练集"), (axes[1], test_df, "测试集")]:
        cnt = df["label"].value_counts().reindex(LABELS_Q).fillna(0)
        sns.barplot(x=cnt.index, y=cnt.values, ax=ax, color="#C4A35A")
        ax.set_title(f"{title} clarity_label")
        ax.set_ylabel("样本数")
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(cnt.values):
            ax.text(i, v + 0.5, int(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_label_dist.png", dpi=160)
    plt.close(fig)

    # 细粒度：训练集 evasion；测试集用三人标注多数票
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    if "evasion_label" in train_df.columns:
        ev = train_df["evasion_label"].fillna("").astype(str)
        ev = ev[ev.str.len() > 0]
        cnt = ev.value_counts()
        sns.barplot(y=cnt.index, x=cnt.values, ax=axes[0], color="#3D5A80")
        axes[0].set_title("训练集 evasion_label")
        axes[0].set_xlabel("样本数")
    if {"annotator1", "annotator2", "annotator3"} <= set(test_df.columns):
        maj = []
        for _, r in test_df.iterrows():
            votes = [r["annotator1"], r["annotator2"], r["annotator3"]]
            votes = [v for v in votes if isinstance(v, str) and v]
            maj.append(pd.Series(votes).mode().iloc[0] if votes else "")
        cnt = pd.Series(maj).value_counts()
        sns.barplot(y=cnt.index, x=cnt.values, ax=axes[1], color="#B85C38")
        axes[1].set_title("测试集 evasion（三人多数票）")
        axes[1].set_xlabel("样本数")
        test_df = test_df.copy()
        test_df["evasion_maj"] = maj
    fig.tight_layout()
    fig.savefig(OUT / "fig_evasion_dist.png", dpi=160)
    plt.close(fig)

    # 长度分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    tmp = train_df.copy()
    tmp["q_len"] = word_len(tmp["text_a"])
    tmp["a_len"] = word_len(tmp["text_b"])
    sns.kdeplot(data=tmp, x="q_len", hue="label", ax=axes[0], common_norm=False)
    axes[0].set_title("问题词数分布（训练）")
    axes[0].set_xlabel("question 词数")
    sns.boxplot(data=tmp, x="label", y="a_len", ax=axes[1], order=LABELS_Q)
    axes[1].set_title("回答词数按清晰度（训练）")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("interview_answer 词数")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_length_dist.png", dpi=160)
    plt.close(fig)

    # 泄漏：测试 url 是否出现在训练
    leak_n = 0
    if "group" in train_df.columns:
        tr_g = set(train_df["group"].astype(str)) - {"", "None", "nan"}
        leak_mask = test_df["group"].astype(str).isin(tr_g)
        leak_n = int(leak_mask.sum())
        pd.Series(leak_mask.astype(int), name="url_in_train").to_csv(
            OUT / "leak_mask.csv", index=False
        )
    leak_info = {
        "test_n": int(len(test_df)),
        "url_overlap_n": leak_n,
        "url_overlap_rate": float(leak_n / max(len(test_df), 1)),
        "train_presidents": train_df["president"].value_counts().to_dict()
        if "president" in train_df.columns else {},
        "test_president_missing": float(
            (test_df["president"].isna() | (test_df["president"].astype(str).isin(["None", "nan", ""]))).mean()
        ) if "president" in test_df.columns else 1.0,
    }
    (OUT / "eda_leakage.json").write_text(
        json.dumps(leak_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "evasion_maj" in test_df.columns:
        return stats, leak_info, test_df
    return stats, leak_info, test_df


def dump_errors(test_df, pred, proba, classes, path, n=12):
    df = test_df.copy()
    df["prediction"] = pred
    df["gold"] = df["label"]
    df["correct"] = df["gold"] == df["prediction"]
    idx = [classes.index(c) if c in classes else 0 for c in pred]
    df["conf"] = [float(proba[i, idx[i]]) for i in range(len(df))]
    df["q_len"] = word_len(df["text_a"])
    df["a_len"] = word_len(df["text_b"])
    wrong = df[~df["correct"]].copy()

    def tag(r):
        g, p, alen = r["gold"], r["prediction"], r["a_len"]
        ev = str(r.get("evasion_maj", r.get("evasion_label", "")))
        mq = bool(r.get("multiple_questions", False))
        if g == "Ambivalent" and p == "Clear Reply" and alen >= 300:
            return "长而空泛"
        if g == "Clear Reply" and p == "Ambivalent" and ev == "Implicit":
            return "隐式回答"
        if mq and g != p:
            return "部分回答"
        if g == "Clear Non-Reply" and p != g:
            return "明确拒答"
        if g == "Ambivalent" and p == "Clear Non-Reply":
            return "主题转移/边界"
        if ev in {"Dodging", "General", "Deflection"}:
            return "主题转移"
        if ev == "Partial/half-answer":
            return "部分回答"
        return "标签边界模糊"

    wrong["error_type"] = wrong.apply(tag, axis=1)
    # 每类尽量各取，再按置信度高的误判补齐
    picked = []
    for t in ["长而空泛", "部分回答", "隐式回答", "主题转移", "明确拒答", "标签边界模糊"]:
        sub = wrong[wrong["error_type"] == t].sort_values("conf", ascending=False)
        picked.extend(sub.head(2).index.tolist())
    rest = wrong.sort_values("conf", ascending=False).index.tolist()
    for i in rest:
        if i not in picked:
            picked.append(i)
        if len(picked) >= n:
            break
    cols = ["id", "text_a", "text_b", "gold", "prediction", "error_type", "conf", "q_len", "a_len"]
    if "evasion_maj" in wrong.columns:
        cols.append("evasion_maj")
    out = wrong.loc[picked[:n], cols].copy()
    out["question_short"] = out["text_a"].str.replace("\n", " ").str.slice(0, 120)
    out["answer_short"] = out["text_b"].str.replace("\n", " ").str.slice(0, 180)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def print_block(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main():
    t0 = time.time()
    set_seed(SEED)
    OUT.mkdir(exist_ok=True)
    setup_plot()

    print_block("1. 读取 QEvasion")
    cfg_q = load_config(CFG_Q)
    train_df, test_df = load_pair_dataset(cfg_q, seed=SEED)
    print("train", len(train_df), "test", len(test_df))
    print("train labels", train_df["label"].value_counts().to_dict())
    print("test labels", test_df["label"].value_counts().to_dict())
    print("字段", list(train_df.columns))

    print_block("2. EDA")
    stats, leak_info, test_df = run_eda(train_df, test_df)
    print("split stats", stats)
    print("url overlap", leak_info)

    y_tr = train_df["label"].tolist()
    y_te = test_df["label"].tolist()
    texts_tr = pair_text(train_df["text_a"], train_df["text_b"])
    texts_te = pair_text(test_df["text_a"], test_df["text_b"])

    metrics = {}

    print_block("3. 多数类基线")
    maj_pred, maj_name = majority_predict(y_tr, len(y_te))
    metrics["majority"] = eval_split(y_te, maj_pred, LABELS_Q)
    metrics["majority"]["note"] = f"always {maj_name}"
    print(metrics["majority"])

    print_block("4. TF-IDF + LR")
    t_tfidf = time.time()
    pipe = train_tfidf(texts_tr, y_tr, ngram=(1, 2), class_weight="balanced")
    pred = pipe.predict(texts_te)
    proba = pipe.predict_proba(texts_te)
    classes = list(pipe.classes_)
    metrics["tfidf_lr"] = eval_split(y_te, pred, LABELS_Q)
    metrics["tfidf_lr"]["seconds"] = round(time.time() - t_tfidf, 2)
    metrics["tfidf_lr"]["ngram"] = [1, 2]
    metrics["tfidf_lr"]["class_weight"] = "balanced"
    metrics["tfidf_lr"]["max_features"] = 50000
    print("TF-IDF", metrics["tfidf_lr"])
    save_preds(OUT / "predictions_tfidf.csv", test_df["id"], y_te, pred, proba, classes)
    plot_confusion(y_te, pred, LABELS_Q, "TF-IDF + LR 混淆矩阵", OUT / "fig_cm_tfidf.png")

    # 无类别权重消融
    pipe_no = train_tfidf(texts_tr, y_tr, ngram=(1, 2), class_weight=None)
    pred_no = pipe_no.predict(texts_te)
    metrics["tfidf_lr_no_weight"] = eval_split(y_te, pred_no, LABELS_Q)

    # 泄漏子集
    leak_mask = test_df["group"].astype(str).isin(
        set(train_df["group"].astype(str)) - {"", "None", "nan"}
    )
    if leak_mask.any() and (~leak_mask).any():
        metrics["tfidf_lr_overlap_url"] = eval_split(
            np.array(y_te)[leak_mask], pred[leak_mask], LABELS_Q
        )
        metrics["tfidf_lr_heldout_url"] = eval_split(
            np.array(y_te)[~leak_mask], pred[~leak_mask], LABELS_Q
        )

    print_block("5. Sentence-BERT + LR")
    t_sbert = time.time()
    Xtr, dim = sbert_encode(texts_tr, OUT / "emb_sbert_train.npy")
    Xte, _ = sbert_encode(texts_te, OUT / "emb_sbert_test.npy")
    clf = train_lr(Xtr, y_tr, class_weight="balanced")
    pred_s = clf.predict(Xte)
    proba_s = clf.predict_proba(Xte)
    classes_s = list(clf.classes_)
    metrics["sbert_lr"] = eval_split(y_te, pred_s, LABELS_Q)
    metrics["sbert_lr"]["seconds"] = round(time.time() - t_sbert, 2)
    metrics["sbert_lr"]["dim"] = int(dim)
    metrics["sbert_lr"]["max_seq_length"] = 256
    metrics["sbert_lr"]["model"] = "sentence-transformers/all-MiniLM-L6-v2"
    print("SBERT", metrics["sbert_lr"])
    save_preds(OUT / "predictions_sbert.csv", test_df["id"], y_te, pred_s, proba_s, classes_s)
    plot_confusion(y_te, pred_s, LABELS_Q, "SBERT + LR 混淆矩阵", OUT / "fig_cm_sbert.png")

    print_block("6. SBERT + 长度特征（改进）")
    extra_tr = np.vstack([
        np.log1p(word_len(train_df["text_a"])),
        np.log1p(word_len(train_df["text_b"])),
        word_len(train_df["text_b"]) / np.maximum(word_len(train_df["text_a"]), 1),
    ]).T
    extra_te = np.vstack([
        np.log1p(word_len(test_df["text_a"])),
        np.log1p(word_len(test_df["text_b"])),
        word_len(test_df["text_b"]) / np.maximum(word_len(test_df["text_a"]), 1),
    ]).T
    clf2 = train_lr(np.hstack([Xtr, extra_tr]), y_tr, class_weight="balanced")
    pred2 = clf2.predict(np.hstack([Xte, extra_te]))
    proba2 = clf2.predict_proba(np.hstack([Xte, extra_te]))
    metrics["sbert_lr_len"] = eval_split(y_te, pred2, LABELS_Q)
    metrics["sbert_lr_len"]["note"] = "SBERT concat log(q_len), log(a_len), a_len/q_len"
    print("SBERT+len", metrics["sbert_lr_len"])
    save_preds(OUT / "predictions_sbert_len.csv", test_df["id"], y_te, pred2, proba2, list(clf2.classes_))
    plot_confusion(y_te, pred2, LABELS_Q, "SBERT+长度 混淆矩阵", OUT / "fig_cm_sbert_len.png")

    # 选 Macro-F1 最好的做错误分析
    best_name = max(["tfidf_lr", "sbert_lr", "sbert_lr_len"], key=lambda k: metrics[k]["macro_f1"])
    best_pred = {"tfidf_lr": pred, "sbert_lr": pred_s, "sbert_lr_len": pred2}[best_name]
    best_proba = {"tfidf_lr": proba, "sbert_lr": proba_s, "sbert_lr_len": proba2}[best_name]
    best_cls = {"tfidf_lr": classes, "sbert_lr": classes_s, "sbert_lr_len": list(clf2.classes_)}[best_name]
    print("best model", best_name, metrics[best_name]["macro_f1"])
    err = dump_errors(test_df, best_pred, best_proba, list(best_cls), OUT / "error_cases.csv", n=12)
    print(err[["id", "gold", "prediction", "error_type"]].to_string(index=False))

    def write_metrics(elapsed=None):
        payload = {
            "seed": SEED,
            "elapsed_sec": round(elapsed if elapsed is not None else time.time() - t0, 2),
            "qevasion_train": int(len(train_df)),
            "qevasion_test": int(len(test_df)),
            "best_model": best_name,
            "metrics": metrics,
            "leakage": leak_info,
            "label_note": "数据文件中高层标签为 Clear Reply / Ambivalent / Clear Non-Reply；Ambivalent 对应论文 Ambivalent Reply。",
        }
        (OUT / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        flat = []
        for k, v in metrics.items():
            if not isinstance(v, dict) or "macro_f1" not in v:
                continue
            flat.append({
                "model": k,
                "accuracy": v.get("accuracy"),
                "macro_p": v.get("macro_p"),
                "macro_r": v.get("macro_r"),
                "macro_f1": v.get("macro_f1"),
            })
        pd.DataFrame(flat).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
        return payload

    write_metrics()

    print_block("7. 迁移 QNLI")
    cfg_t = load_config(CFG_T)
    qnli_tr, qnli_te = load_pair_dataset(cfg_t, seed=SEED)
    print("qnli train", len(qnli_tr), qnli_tr["label"].value_counts().to_dict())
    print("qnli test", len(qnli_te), qnli_te["label"].value_counts().to_dict())
    labels_t = sorted(qnli_tr["label"].unique().tolist())
    q_tr = pair_text(qnli_tr["text_a"], qnli_tr["text_b"])
    q_te = pair_text(qnli_te["text_a"], qnli_te["text_b"])

    maj_t, maj_t_name = majority_predict(qnli_tr["label"], len(qnli_te))
    metrics["qnli_majority"] = eval_split(qnli_te["label"], maj_t, labels_t)
    metrics["qnli_majority"]["note"] = f"always {maj_t_name}"

    # 适配后训练：在 QNLI 上重新拟合 TF-IDF
    pipe_q = train_tfidf(q_tr, qnli_tr["label"].tolist(), ngram=(1, 2), class_weight="balanced")
    pred_q = pipe_q.predict(q_te)
    proba_q = pipe_q.predict_proba(q_te)
    metrics["qnli_tfidf_adapt"] = eval_split(qnli_te["label"], pred_q, labels_t)
    save_preds(OUT / "predictions_qnli.csv", qnli_te["id"], qnli_te["label"], pred_q, proba_q, list(pipe_q.classes_))
    plot_confusion(qnli_te["label"], pred_q, labels_t, "QNLI 适配后 TF-IDF+LR", OUT / "fig_cm_qnli.png")

    # 直接迁移：沿用 QEvasion 上拟合的 TF-IDF 词表，只在 QNLI 上重训分类头
    src_vec = pipe.named_steps["tfidfvectorizer"]
    Xtr_t = src_vec.transform(q_tr)
    Xte_t = src_vec.transform(q_te)
    head = train_lr(Xtr_t, qnli_tr["label"].tolist(), class_weight="balanced")
    pred_d = head.predict(Xte_t)
    metrics["qnli_tfidf_direct"] = eval_split(qnli_te["label"], pred_d, labels_t)
    metrics["qnli_tfidf_direct"]["note"] = "frozen QEvasion TF-IDF vocab + new LR head"

    metrics["transfer_sizes"] = {
        "qnli_train": int(len(qnli_tr)),
        "qnli_test": int(len(qnli_te)),
        "qnli_train_limit": cfg_t.get("train_limit"),
        "qnli_test_limit": cfg_t.get("test_limit"),
    }

    payload = write_metrics()
    print_block("完成")
    print("elapsed", payload["elapsed_sec"], "s")
    print("outputs ->", OUT)


if __name__ == "__main__":
    main()
