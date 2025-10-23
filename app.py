# app.py （CSV+JSON 読込 → 検索 → マッチングTop3 → アイデア生成 → PR作成）
from __future__ import annotations
import io, os, sys, re, json as _json, datetime
from pathlib import Path
from typing import List, Dict
from zipfile import ZipFile

import pandas as pd
import streamlit as st
import chardet


import os
import pandas as pd
from PyPDF2 import PdfReader

# ===============================
# 🔹 1. 環境変数の読み込み
# ===============================

# --- ユーザー認証情報（USERNAME_1 / PASSWORD_1 形式）---
def load_users_from_env(max_users=50):
    users = []
    for i in range(1, max_users + 1):
        u = os.getenv(f"USERNAME_{i}")
        p = os.getenv(f"PASSWORD_{i}")
        if u and p:
            users.append({"username": u, "password": p})
    return users

USERS = load_users_from_env()

# ------------------------------------------------------

# 3) ログインUI（ペア一致必須）

# ------------------------------------------------------

def login_ui():

    st.title("🔐 ログイン")

    st.caption("※ Render の Environment に設定した USERNAME_i / PASSWORD_i のペアで認証します。")

    in_user = st.text_input("ユーザー名")

    in_pass = st.text_input("パスワード", type="password")

    if st.button("ログイン"):

        if any(u["username"] == in_user and u["password"] == in_pass for u in USERS):

            st.session_state["logged_in"] = True

            st.session_state["user_name"] = in_user

            st.success(f"ようこそ、{in_user} さん！")

            st.rerun()

        else:

            st.error("ユーザー名またはパスワードが間違っています。")

  

if "logged_in" not in st.session_state:

    st.session_state["logged_in"] = False

  

if not USERS:

    st.error("Environment に USERNAME_1/PASSWORD_1 形式でユーザーが設定されていません。")

    st.stop()

  

if not st.session_state["logged_in"]:

    login_ui()

    st.stop()

  

# ログイン後サイドバー

with st.sidebar:

    st.success(f"👤 ログイン中：{st.session_state['user_name']}")

    if st.button("ログアウト"):

        st.session_state.clear()

        st.rerun()


# ================== ページ設定 ==================
st.set_page_config(page_title="技術ニーズ マッチング UI", layout="wide")
st.title("🧩 技術ニーズ マッチング UI")
st.sidebar.header("操作")

# ---- 正規列名（アプリ内部名）
COL_NO   = "番号"
COL_COMP = "企業名"
COL_NEWS = "技術ニーズのニュース名"
COL_CATL = "大分類(連番付き項目名)"
COL_CATM = "中分類(連番付き項目名)"
COL_SUM  = "要約"

# ---- セッション
ss = st.session_state
ss.setdefault("page", "select")            # select -> result -> idea
ss.setdefault("df_raw", None)
ss.setdefault("json_obj", None)
ss.setdefault("selection", set())
ss.setdefault("result_df", pd.DataFrame())
ss.setdefault("idea_df", pd.DataFrame())
ss.setdefault("idea_selection", set())     # ← PR出力用の行選択を保持

# =========================================================
# 共通ユーティリティ
# =========================================================
def normalize_col(name: str) -> str:
    s = str(name).strip()
    s = s.replace("　", " ").replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", " ", s)
    return s

@st.cache_data(show_spinner=False)
def load_dataframe(file) -> pd.DataFrame:
    raw = file.read()
    name = file.name.lower()
    if name.endswith(".csv"):
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        for e in [enc, "utf-8", "cp932", "shift_jis", "utf-16", "utf-16le", "utf-16be"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=e)
            except Exception:
                continue
        raise RuntimeError("CSVの文字コード判定に失敗（UTF-8/Shift-JIS/UTF-16を試行）")
    return pd.read_excel(io.BytesIO(raw))

def auto_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    fix = {}
    if "大分類" in df.columns: fix["大分類"] = COL_CATL
    if "中分類" in df.columns: fix["中分類"] = COL_CATM
    if "大分類 (連番付き項目名)" in df.columns: fix["大分類 (連番付き項目名)"] = COL_CATL
    if "中分類 (連番付き項目名)" in df.columns: fix["中分類 (連番付き項目名)"] = COL_CATM
    if fix: df = df.rename(columns=fix)
    aliases = {
        COL_NO:   ["No","no","ID","Id","id","番号"],
        COL_COMP: ["会社名","企業","社名","企業名"],
        COL_NEWS: ["ニュース名","技術ニーズニュース名","技術ニーズのニュース名"],
        COL_CATL: ["大分類(連番付き項目名)","大分類（連番付き項目名）"],
        COL_CATM: ["中分類(連番付き項目名)","中分類（連番付き項目名）"],
        COL_SUM:  ["サマリー","摘要","要旨","要約"],
    }
    cmap = {}
    for canon, cands in aliases.items():
        for c in cands:
            if c in df.columns:
                cmap[c] = canon
                break
    if cmap: df = df.rename(columns=cmap)
    return df

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in [COL_NO, COL_COMP, COL_NEWS, COL_CATL, COL_CATM, COL_SUM]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].astype(str)
    return df

# =========================================================
# 外部スクリプト呼び出し（find_similar / idea_generator / PR生成）
# =========================================================
def run_find_similar_needs(df_selected: pd.DataFrame, params: dict) -> pd.DataFrame:
    """同フォルダの find_similar_needs.py の run(df_selected, params) を実行"""
    from importlib import import_module
    script = Path(__file__).with_name("find_similar_needs.py")
    if not script.exists():
        raise FileNotFoundError(f"{script} が見つかりません。")
    sys.path.insert(0, str(script.parent))
    try:
        mod = import_module(script.stem)
        if not hasattr(mod, "run"):
            raise RuntimeError("find_similar_needs.py に run(df_selected, params) がありません。")
        return mod.run(df_selected, params)
    finally:
        try: sys.path.remove(str(script.parent))
        except ValueError: pass

def run_idea_generator(result_df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """同フォルダの idea_generator_from_excel_openai.py の run(result_df, params) を実行"""
    from importlib import import_module
    script = Path(__file__).with_name("idea_generator_from_excel_openai.py")
    if not script.exists():
        raise FileNotFoundError(f"{script} が見つかりません。")
    sys.path.insert(0, str(script.parent))
    try:
        mod = import_module(script.stem)
        fn = getattr(mod, "run", None)
        if not callable(fn):
            raise RuntimeError("idea_generator_from_excel_openai.py に run(result_df, params) がありません。")
        out = fn(result_df, params or {})
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)
    finally:
        try: sys.path.remove(str(script.parent))
        except ValueError: pass

# ---- PR生成（添付PGを内部実行）
def _sanitize_filename(name: str, maxlen: int = 80) -> str:
    s = (name or "").strip()
    s = re.sub(r'[\\/:*?"<>|]', '_', s)
    s = re.sub(r'\s+', ' ', s)
    return (s[:maxlen]).rstrip(' .')

def run_press_release_from_ideas(idea_df: pd.DataFrame,
                                 rows: list[int] | None = None,
                                 refs_dir: Path | None = None,
                                 outdir: Path | None = None,
                                 model: str = "o3-2025-04-16") -> Dict[str, bytes]:
    """
    idea_df の選択行から PR 文書(.docx)を作成し、{ファイル名: バイト列} を返す。
    - rows: 1始まりの行番号（未指定なら全行）
    - refs_dir: 参照PDFフォルダ（省略時は app.py と同階層の「プレスリリース作成」）
    - outdir: 生成先（指定すれば実ファイルも保存）
    """
    if idea_df is None or idea_df.empty:
        return {}

    from importlib import import_module
    mod_path = Path(__file__).with_name("batch_pr_from_excel_docx.py")
    if not mod_path.exists():
        raise FileNotFoundError(f"{mod_path} が見つかりません。")
    sys.path.insert(0, str(mod_path.parent))
    try:
        pr = import_module(mod_path.stem)
    finally:
        try: sys.path.remove(str(mod_path.parent))
        except ValueError: pass

    # 参照PDF（存在すればスタイル適用）
    refs_dir = refs_dir or Path(__file__).with_name("プレスリリース作成")
    style_text = pr.prepare_style_corpus(refs_dir) if refs_dir.exists() else None

    if not rows:
        rows = list(range(1, len(idea_df) + 1))
    valid_idx = [i for i in rows if 1 <= i <= len(idea_df)]
    if not valid_idx:
        return {}

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today()
    date_jp = f"{today.year}年{today.month:02d}月{today.day:02d}日（作成日）"

    # 列名マップ（内部→PR側）
    df = idea_df.rename(columns={
        "入力技術ニーズ": "入力_技術ニーズ",
        "入力製品シーズ": "入力_製品シーズ",
    }).copy()

    outputs: Dict[str, bytes] = {}
    for i in valid_idx:
        row = df.iloc[i - 1]
        idea_name = str(row.get("アイデア名", f"idea_{i}")).strip() or f"idea_{i}"
        fname = _sanitize_filename(idea_name) + ".docx"

        fields = pr.build_fields_from_row(row)
        md = pr.call_openai(model, fields, date_jp, style_text)

        # 保存 & バイト化
        if outdir:
            out_path = outdir / fname
            pr.mdish_to_docx(md, out_path)
            outputs[fname] = out_path.read_bytes()
        else:
            tmp = Path.cwd() / fname
            pr.mdish_to_docx(md, tmp)
            outputs[fname] = tmp.read_bytes()
            tmp.unlink(missing_ok=True)

    return outputs

# =========================================================
# 画面: 選択・検索
# =========================================================
def page_select():
    st.subheader("1) 入力ファイルの読込（自動/手動）")

    # 自動読込（同フォルダ）
    #default_csv = Path(__file__).with_name("/etc/secrets/tech_needs_classification.csv")
    default_csv = Path("display/tech_needs_classification.csv")
    default_json = Path("display/excel_analysis.json")

    if ss.df_raw is None and default_csv.exists():
        try:
            df = pd.read_csv(default_csv, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(default_csv, encoding="cp932", errors="ignore")
        df = enforce_types(auto_alias_columns(df))
        ss.df_raw = df
        st.info(f"CSV 自動読込: {default_csv.name}")

    if ss.json_obj is None and default_json.exists():
        try:
            ss.json_obj = _json.loads(default_json.read_text(encoding="utf-8"))
            st.info(f"JSON 自動読込: {default_json.name}")
        except Exception as e:
            st.warning(f"JSON 自動読込失敗: {e}")

    # 手動アップロード
    up_csv = st.file_uploader("CSV/Excel を選択（任意：自動読込済みなら不要）", type=["csv","xlsx","xls"])
    if up_csv:
        try:
            df = enforce_types(auto_alias_columns(load_dataframe(up_csv)))
            ss.df_raw = df
            st.success(f"読込完了: {df.shape[0]}行 × {df.shape[1]}列")
        except Exception as e:
            st.error(f"読込エラー: {e}")

    up_json = st.file_uploader("JSON（excel_analysis.json・任意）", type=["json"])
    if up_json:
        try:
            ss.json_obj = _json.load(io.TextIOWrapper(up_json, encoding="utf-8"))
            st.success("JSON 読込完了")
        except Exception as e:
            st.error(f"JSON解析エラー: {e}")

    if ss.df_raw is None:
        st.stop()

    # 検索UI
    st.subheader("2) 検索条件")
    df = ss.df_raw.copy()
    c1, c2, c3 = st.columns([2,2,2])
    comp = c1.selectbox("企業名：", ["(すべて)"] + sorted(df[COL_COMP].dropna().unique().tolist()), index=0)
    catL = c2.selectbox("大分類：", ["(すべて)"] + sorted(df[COL_CATL].dropna().unique().tolist()), index=0)
    mids = sorted(df[df[COL_CATL] == catL][COL_CATM].dropna().unique().tolist()) if catL != "(すべて)" \
           else sorted(df[COL_CATM].dropna().unique().tolist())
    catM = c3.selectbox("中分類：", ["(すべて)"] + mids, index=0)

    c4, c5, c6 = st.columns([3,1,1])
    q = c4.text_input("ニュース名：", value="", placeholder="（曖昧検索）")
    do_search = c5.button("検索")
    do_clear  = c6.button("クリア")

    if do_clear:
        comp = catL = catM = "(すべて)"; q = ""

    flt = df.copy()
    if comp != "(すべて)": flt = flt[flt[COL_COMP] == comp]
    if catL != "(すべて)": flt = flt[flt[COL_CATL] == catL]
    if catM != "(すべて)": flt = flt[flt[COL_CATM] == catM]
    if do_search and q.strip():
        flt = flt[flt[COL_NEWS].str.contains(q.strip(), case=False, na=False)]

    # 行選択
    st.subheader("3) 対象の行を選択")
    view_cols = [COL_NO, COL_COMP, COL_NEWS, COL_CATL, COL_CATM, COL_SUM]
    v = flt[view_cols].copy()
    v.insert(0, "選択", v[COL_NO].astype(str).isin(ss.selection))
    edited = st.data_editor(
        v, hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={"選択": st.column_config.CheckboxColumn("", help="この行を選択")}
    )
    ss.selection = set(edited[edited["選択"]][COL_NO].astype(str).tolist())
    st.caption(f"選択件数：{len(ss.selection)}")

    # 実行
    if st.button("🔎 マッチング実行（Top3）", type="primary"):
        if not ss.selection:
            st.warning("先に行を選択してください。"); st.stop()
        df_selected = df[df[COL_NO].astype(str).isin(list(ss.selection))].copy()
        with st.spinner("CSV+JSON を用いてマッチング中…"):
            res = run_find_similar_needs(
                df_selected,
                params={
                    "df_all": ss.df_raw,
                    "json_obj": ss.json_obj,
                    "top_n": 3,  # Top3固定
                    "target_exclude_companies": ["ローム"],  # クエリ=ローム以外
                    "cand_include_companies": ["ローム"],   # コーパス=ロームのみ
                    "exclude_same_company": True,
                }
            )
        # 念のため各元ニュースにつきTop3を強制
        res = (res.sort_values("similarity", ascending=False)
                 .groupby("元技術ニーズのニュース名", as_index=False, sort=False)
                 .head(3).reset_index(drop=True))
        # 番号=行番号に振り直し
        res[COL_NO] = (pd.Series(range(1, len(res)+1))).astype(str)
        ss.result_df = res
        ss.page = "result"; st.rerun()

# =========================================================
# 画面: 結果（Top3）
# =========================================================
def page_result():
    st.subheader("マッチング結果（Top3）")
    df = ss.result_df.copy()
    st.dataframe(df, use_container_width=True, height=560)

    c1, c2, c3 = st.columns([2,1,1])
    c1.download_button("CSVでダウンロード",
                       data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="matching_result.csv", mime="text/csv")
    back = c2.button("← 戻る")
    go_idea = c3.button("アイデア生成 ▶", type="primary")

    if back:
        ss.page = "select"; st.rerun()

    if go_idea:
        try:
            with st.spinner("アイデア生成中…"):
                ss.idea_df = run_idea_generator(df, params={"max_ideas_per_need": 3})
            ss.page = "idea"; st.rerun()
        except Exception as e:
            import traceback
            st.error("アイデア生成でエラーが発生しました。")
            st.exception(e); st.code(traceback.format_exc())

# =========================================================
# 画面: アイデア一覧 ＋ プレスリリース作成（チェックボックス選択）
# =========================================================
def page_idea():
    st.subheader("アイデア一覧（自動生成）")
    df = ss.idea_df.copy()
    if df.empty:
        st.info("アイデアデータが空です。")
        if st.button("← 結果に戻る"):
            ss.page = "result"; st.rerun()
        return

    # 表示：1始まりの行番号 & チェック列
    df_view = df.copy()
    df_view.insert(0, "行番号", range(1, len(df_view) + 1))
    pre_checked = df_view["行番号"].astype(int).isin(ss.idea_selection)
    df_view.insert(0, "選択", pre_checked)

    show_cols = [
        "選択", "行番号", "番号", "元シート名",
        "入力技術ニーズ", "入力製品シーズ",
        "アイデア名", "アイデアの切り口",
        "具体的なソリューション", "アイデア創出のプロセス",
    ]
    show_cols = [c for c in show_cols if c in df_view.columns]

    st.markdown(
        "<div style='text-align:right; margin: 6px 0;'>"
        "<span style='font-size:20px; font-weight:700; "
        "padding:6px 18px; border:2px solid #333; border-radius:6px;'>ファイル作成</span>"
        "</div>", unsafe_allow_html=True
    )

    edited = st.data_editor(
        df_view[show_cols],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={"選択": st.column_config.CheckboxColumn("")},
        height=520,
    )

    selected_rows = edited.loc[edited["選択"], "行番号"].astype(int).tolist()
    ss.idea_selection = set(selected_rows)

    # 参照PDFフォルダ（app.py と同階層の「プレスリリース作成/」）
    ref_dir = Path(__file__).with_name("プレスリリース作成")
    if ref_dir.exists():
        st.caption(f"参照PDFフォルダ：{ref_dir}（検出 → スタイル適用）")
    else:
        st.caption("参照PDFフォルダ：未検出（スタイル参照なしで作成）")

    c1, c2, c3 = st.columns([2,1,2])
    do_make = c1.button("📰 選択した行だけプレスリリース作成", type="primary")
    back    = c2.button("← 結果に戻る")
    c3.download_button(
        "アイデアCSVをダウンロード",
        data=edited.drop(columns=["選択"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="ideas.csv", mime="text/csv"
    )

    if do_make:
        if not selected_rows:
            st.warning("先にチェックボックスで出力対象の行を選択してください。")
        else:
            with st.spinner("プレスリリースを生成中..."):
                files = run_press_release_from_ideas(
                    idea_df=df,                        # 元のアイデアDF
                    rows=selected_rows,                # ← 選択した行だけ
                    refs_dir=(ref_dir if ref_dir.exists() else None),
                    outdir=None,
                    model="o3-2025-04-16"
                )
            if not files:
                st.warning("生成対象がありません or 生成に失敗しました。")
            else:
                st.success(f"{len(files)} 件の .docx を生成しました。")
                # 個別ダウンロード（ファイル名＝アイデア名）
                st.markdown("**個別ダウンロード**")
                for name, data in files.items():
                    st.download_button(
                        f"⬇ {name}", data=data, file_name=name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                # ZIP 一括
                mem = io.BytesIO()
                with ZipFile(mem, mode="w") as zf:
                    for name, data in files.items():
                        zf.writestr(name, data)
                mem.seek(0)
                st.download_button("📦 ZIPで一括ダウンロード", data=mem.read(),
                                   file_name="press_releases.zip", mime="application/zip")

    if back:
        ss.page = "result"; st.rerun()

# =========================================================
# ルーティング
# =========================================================
if ss.page == "select":
    page_select()
elif ss.page == "result":
    page_result()
else:
    page_idea()
