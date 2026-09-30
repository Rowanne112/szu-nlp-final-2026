# -*- coding: utf-8 -*-
"""字段适配器：把不同数据集统一成 text_a / text_b / label / group / id。"""
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _map_label(raw, mapping):
    if mapping is None:
        return str(raw)
    key = str(raw)
    if key in mapping:
        return mapping[key]
    if raw in mapping:
        return mapping[raw]
    return str(raw)


def _load_hf(name, subset):
    """兼容 glue 这类无命名空间仓库在新版 huggingface_hub 下的加载。"""
    aliases = [name]
    if name == "glue":
        aliases = ["nyu-mll/glue", "glue"]
    elif name == "nyu-mll/glue":
        aliases = ["nyu-mll/glue", "glue"]
    last = None
    for n in aliases:
        try:
            return load_dataset(n, subset) if subset else load_dataset(n)
        except Exception as e:
            last = e
    raise last


def load_pair_dataset(cfg, seed=42):
    """按配置读取 HuggingFace 数据集，返回 train/test 两个 DataFrame。"""
    name = cfg["dataset_name"]
    subset = cfg.get("subset")
    ds = _load_hf(name, subset)

    test_split = cfg.get("test_split", "test")
    if test_split not in ds:
        test_split = "validation" if "validation" in ds else list(ds.keys())[-1]

    train_df = _to_frame(ds["train"], cfg, seed, cfg.get("train_limit"), prefix="train")
    test_df = _to_frame(ds[test_split], cfg, seed + 1, cfg.get("test_limit"), prefix="test")
    return train_df, test_df


def _to_frame(split, cfg, seed, limit, prefix):
    df = split.to_pandas()
    if limit is not None and len(df) > int(limit):
        df = df.sample(n=int(limit), random_state=seed).reset_index(drop=True)

    a_col, b_col, y_col = cfg["text_a_column"], cfg["text_b_column"], cfg["label_column"]
    id_col = cfg.get("id_column")
    g_col = cfg.get("group_column")
    mapping = cfg.get("label_mapping") or {}

    out = pd.DataFrame({
        "text_a": df[a_col].fillna("").astype(str),
        "text_b": df[b_col].fillna("").astype(str),
        "label": df[y_col].map(lambda x: _map_label(x, mapping)),
    })
    if id_col and id_col in df.columns:
        out["id"] = prefix + "_" + df[id_col].astype(str)
    else:
        out["id"] = [f"{prefix}_{i}" for i in range(len(df))]
    if g_col and g_col in df.columns:
        out["group"] = df[g_col].astype(str)
    else:
        out["group"] = ""

    extras = cfg.get("extra_columns") or []
    for col in extras:
        if col in df.columns:
            out[col] = df[col].values
    out["split"] = prefix
    return out.reset_index(drop=True)


def pair_text(a, b, sep=" [SEP] "):
    return (a.fillna("").astype(str) + sep + b.fillna("").astype(str)).tolist()
