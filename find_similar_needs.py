# find_similar_needs.py
# ------------------------------------------------------------
# SBERT（sonoisa/sentence-bert-base-ja-mean-tokens-v2）で意味類似検索。
# コーパス（=比較対象。今回ロームのみ）の埋め込みはディスクにキャッシュし、
# 初回だけ計算・以降は差分だけ再計算します。
#
# 入口: run(df_selected, params)
#   params:
#     df_all: pd.DataFrame                … CSV全体（必須）
#     json_obj: list | None               … excel_analysis.json（任意）
#     top_n: int                          … 既定 3
#     target_exclude_companies: [str]     … 既定 ["ローム"]  → クエリから除外
#     cand_include_companies:   [str]     … 既定 ["ローム"]  → コーパスに含める
#     exclude_same_company: bool          … 既定 True       → 同一企業候補を除外
#
# 返り値: UI想定のカラムで並んだ DataFrame
# ------------------------------------------------------------

from __future__ import annotations
import re, os, json, hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ====== モデル名（日本語SBERT） ======
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"

# ====== キャッシュ設定（このファイルと同じフォルダに作成） ======
CACHE_DIR = Path(__file__).with_name(".cache_sbert")
CACHE_DIR.mkdir(exist_ok=True)

def _fingerprint(text: str) -> str:
    """テキスト内容のハッシュ（キャッシュキー用）"""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def _cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return CACHE_DIR / f"embeddings_{safe}.parquet"

def load_corpus_cache(model_name: str) -> pd.DataFrame:
    """コーパス埋め込みキャッシュを読み込み（無ければ空）"""
    p = _cache_path(model_name)
    if p.exists():
        df = pd.read_parquet(p)
        # list -> np.array に戻す
        df["emb"] = df["emb"].apply(lambda x: np.array(x, dtype="float32"))
        return df
    return pd.DataFrame(columns=["key", "text", "emb"])

def save_corpus_cache(model_name: str, df_cache: pd.DataFrame) -> None:
    """コーパス埋め込みキャッシュを保存"""
    p = _cache_path(model_name)
    to_save = df_cache.copy()
    to_save["emb"] = to_save["emb"].apply(lambda v: np.asarray(v, dtype="float32").tolist())
    to_save.to_parquet(p, index=False)

def get_corpus_embeddings_cached(model: SentenceTransformer,
                                 model_name: str,
                                 rows: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
    """
    rows: [{"id": "...", "text": "埋め込み対象テキスト"}, ...]
    戻り: (コーパス行列 [N, D], keys[長さN])  ※keysは text の fingerprint
    """
    cache = load_corpus_cache(model_name)
    cache_map = {k: i for i, k in enumerate(cache["key"].tolist())}

    embs_list: List[np.ndarray] = []
    keys: List[str] = []
    to_compute: List[Tuple[str, str]] = []  # (key, text)

    for r in rows:
        text = r.get("text", "") or ""
        key = _fingerprint(text)
        if key in cache_map:
            emb = cache.iloc[cache_map[key]]["emb"]
            embs_list.append(emb)
            keys.append(key)
        else:
            to_compute.append((key, text))

    if to_compute:
        texts = [t for _, t in to_compute]
        new_embs = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 内積=コサインに
        )
        new_records = []
        for (key, text), emb in zip(to_compute, new_embs):
            new_records.append({"key": key, "text": text, "emb": emb})
            embs_list.append(emb)
            keys.append(key)
        if new_records:
            cache = pd.concat([cache, pd.DataFrame(new_records)], ignore_index=True)
            save_corpus_cache(model_name, cache)

    D = model.get_sentence_embedding_dimension()
    corpus_mat = np.vstack(embs_list) if embs_list else np.zeros((0, D), dtype="float32")
    return corpus_mat, keys

# ====== 軽い前処理 ======
def _norm_text(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ====== JSON取り回し ======
def _index_json(json_obj: Any) -> Dict[str, Dict[str, List[str]]]:
    """
    excel_analysis.json（想定構造）を index 化：
    { file_name: {"needs":[...], "features":[...]} }
    """
    idx: Dict[str, Dict[str, List[str]]] = {}
    if not isinstance(json_obj, list):
        return idx
    for item in json_obj:
        if not isinstance(item, dict):
            continue
        fn = item.get("file_name") or item.get("filename") or item.get("name")
        if not fn:
            continue
        needs = []
        feats = []
        tn = item.get("技術ニーズ")
        if isinstance(tn, dict) and isinstance(tn.get("needs"), list):
            needs = [str(x) for x in tn["needs"]]
        ps = item.get("製品シーズ")
        if isinstance(ps, dict) and isinstance(ps.get("features"), list):
            feats = [str(x) for x in ps["features"]]
        idx[str(fn)] = {"needs": needs, "features": feats}
    return idx

def _guess_key_by_news(news: str, keys: List[str]) -> str | None:
    """ニュース名から file_name をゆるく推測（完全一致→部分一致）"""
    news = (news or "").strip().lower()
    if not news or not keys:
        return None
    for k in keys:
        if news == str(k).strip().lower():
            return k
    for k in keys:
        kk = str(k).strip().lower()
        if kk in news or news in kk:
            return k
    return None

def _needs_text(lst: Any, sep: str = " / ") -> str:
    if isinstance(lst, list) and lst:
        flat = []
        for x in lst:
            if isinstance(x, list): flat.extend(x)
            else: flat.append(x)
        return sep.join([str(s) for s in flat if str(s).strip()])
    return "該当なし"

# ====== コーパス用テキストの組み立て ======
def _build_corpus_text(row: pd.Series,
                       json_idx: Dict[str, Dict[str, List[str]]],
                       json_keys: List[str]) -> Tuple[str, str, str]:
    """
    返り: (埋め込みテキスト, needs_text_for_display, features_text_for_display)
    優先: JSON needs（あれば）→ 連結文字列。無ければ 要約。
    """
    news = str(row.get("技術ニーズのニュース名", "") or "")
    summary = str(row.get("要約", "") or "")
    key = _guess_key_by_news(news, json_keys) if json_idx else None
    needs_list = json_idx.get(key, {}).get("needs") if key else None
    feats_list = json_idx.get(key, {}).get("features") if key else None

    needs_text = _needs_text(needs_list)
    feats_text = _needs_text(feats_list)

    if isinstance(needs_list, list) and len(needs_list) > 0:
        emb_text = "。".join(needs_list)
    else:
        emb_text = summary

    return _norm_text(emb_text), needs_text, feats_text

# ====== 入口：アプリから呼ばれる ======
def run(df_selected: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    クエリ=ローム以外（画面選択行でさらに絞り込み）
    コーパス=ロームのみ
    SBERTでTopN（既定3）を返す（キャッシュで高速化）
    """
    if params is None:
        params = {}

    df_all = params.get("df_all", None)
    if df_all is None or df_all.empty:
        raise ValueError("params['df_all'] が空です（CSV全体の DataFrame を渡してください）。")

    json_obj = params.get("json_obj", None)
    top_n = int(params.get("top_n", 3))
    exclude_companies = params.get("target_exclude_companies", ["ローム"])
    include_companies = params.get("cand_include_companies", ["ローム"])
    exclude_same = bool(params.get("exclude_same_company", True))

    # 正規列名（app.py と合わせる）
    COL_NO   = "番号"
    COL_COMP = "企業名"
    COL_NEWS = "技術ニーズのニュース名"
    COL_CATL = "大分類(連番付き項目名)"
    COL_CATM = "中分類(連番付き項目名)"
    COL_SUM  = "要約"
    COL_SEED = "関連する製品シーズ"

    # 入力の整形
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        for c in [COL_NO, COL_COMP, COL_NEWS, COL_CATL, COL_CATM, COL_SUM]:
            if c not in df.columns:
                df[c] = ""
            df[c] = df[c].astype(str)
        if COL_SEED not in df.columns:
            df[COL_SEED] = ""
        return df

    df_all = _coerce(df_all.copy())
    df_selected = _coerce(df_selected.copy())

    # JSON index
    json_idx = _index_json(json_obj) if json_obj else {}
    json_keys = list(json_idx.keys())

    # クエリ=ローム以外（かつ画面選択の番号に限定）
    selected_ids = set(df_selected[COL_NO].astype(str).tolist())
    df_query = df_all[df_all[COL_COMP] != "ローム"].copy()
    if selected_ids:
        df_query = df_query[df_query[COL_NO].astype(str).isin(selected_ids)].copy()

    # コーパス=ロームのみ
    df_corpus = df_all[df_all[COL_COMP] == "ローム"].copy()

    if df_query.empty or df_corpus.empty:
        return pd.DataFrame(columns=[
            "番号","元技術ニーズのニュース名","元技術ニーズの要約","元技術ニーズ一覧",
            "類似先ニュースの要約","類似先技術ニーズ一覧","similarity",
            "技術ニーズのニュース名","関連する製品シーズ",
            "大分類(連番付き項目名)","中分類(連番付き項目名)",
            "番号(類似先)","企業名(類似先)"
        ])

    # ====== モデル読込 ======
    model = SentenceTransformer(MODEL_NAME)

    # ====== コーパス側：埋め込みテキスト作成 ＆ キャッシュ利用 ======
    corpus_rows: List[Dict[str, str]] = []
    corpus_disp_needs: Dict[int, str] = {}
    corpus_disp_feats: Dict[int, str] = {}
    for idx, r in df_corpus.iterrows():
        emb_text, needs_text, feats_text = _build_corpus_text(r, json_idx, json_keys)
        corpus_rows.append({"id": str(r.get(COL_NO, "")), "text": emb_text})
        corpus_disp_needs[idx] = needs_text
        corpus_disp_feats[idx] = feats_text

    corpus_mat, _keys = get_corpus_embeddings_cached(model, MODEL_NAME, corpus_rows)
    # 正規化済みなので、内積 = コサイン

    # ====== クエリごとにTopN ======
    rows_out: List[Dict[str, Any]] = []
    for _, tgt in df_query.iterrows():
        origin_news    = str(tgt.get(COL_NEWS, "") or "")
        origin_company = str(tgt.get(COL_COMP, "") or "")

        # クエリの埋め込みテキスト（needs優先）
        q_text, origin_needs_text, _ = _build_corpus_text(tgt, json_idx, json_keys)
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]

        if corpus_mat.shape[0] == 0:
            sims = np.array([])
        else:
            sims = corpus_mat @ q_emb  # cos

        # 候補DFにスコア付与
        work = df_corpus.copy()
        work = work.assign(similarity=sims)

        # 自分自身/同一企業を除外（安全側）
        work = work[work[COL_NEWS].astype(str).str.strip() != origin_news.strip()]
        if exclude_same:
            work = work[work[COL_COMP] != origin_company]

        # TopN
        work_top = (work.sort_values("similarity", ascending=False)
                        .drop_duplicates(subset=[COL_NEWS], keep="first")
                        .head(max(1, top_n)))

        for idx2, cand in work_top.iterrows():
            rows_out.append({
                "番号": tgt.get(COL_NO, ""),
                "元技術ニーズのニュース名": origin_news,
                "元技術ニーズの要約": tgt.get(COL_SUM, ""),
                "元技術ニーズ一覧": origin_needs_text,  # クエリ側 needs の見出し
                "類似先ニュースの要約": cand.get(COL_SUM, ""),
                "類似先技術ニーズ一覧": corpus_disp_needs.get(idx2, "該当なし"),
                "similarity": float(cand.get("similarity", 0.0)),
                "技術ニーズのニュース名": cand.get(COL_NEWS, ""),
                "関連する製品シーズ": corpus_disp_feats.get(idx2, "該当なし"),
                "大分類(連番付き項目名)": cand.get(COL_CATL, ""),
                "中分類(連番付き項目名)": cand.get(COL_CATM, ""),
                "番号(類似先)": cand.get(COL_NO, ""),
                "企業名(類似先)": cand.get(COL_COMP, ""),
            })

    out = pd.DataFrame(rows_out)
    wanted = [
        "番号","元技術ニーズのニュース名","元技術ニーズの要約","元技術ニーズ一覧",
        "類似先ニュースの要約","類似先技術ニーズ一覧","similarity",
        "技術ニーズのニュース名","関連する製品シーズ",
        "大分類(連番付き項目名)","中分類(連番付き項目名)",
        "番号(類似先)","企業名(類似先)"
    ]
    for c in wanted:
        if c not in out.columns:
            out[c] = ""
    return out[wanted]
