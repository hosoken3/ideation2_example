# -*- coding: utf-8 -*-
"""
idea_generator_from_excel_openai.py
- Streamlit 側から import され、 run(result_df, params) を呼ばれる想定
- 各マッチ行につき **3案（直接/隣接/異分野）** を LLM で生成（既定）
"""

import os, re, json
from typing import List, Dict
import pandas as pd

# 出力列
OUTPUT_COLUMNS = [
    "番号","元シート名","入力技術ニーズ","入力製品シーズ",
    "アイデア名","アイデアの切り口","具体的なソリューション","アイデア創出のプロセス"
]

# 必須入力列（無い場合は空で補完）
REQ_INPUT_COLUMNS = [
    "番号","元技術ニーズのニュース名","元技術ニーズの要約","類似先ニュースの要約",
    "技術ニーズのニュース名","関連する製品シーズ"
]

APPROACHES = ["直接的な課題解決アイデア","隣接領域への応用アイデア","異分野への転用アイデア"]
COLUMN_TECH_NEED = "技術ニーズのニュース名"
COLUMN_TECH_SEED = "関連する製品シーズ"

# ---------- ユーティリティ ----------
def _shorten(text: str, limit: int = 48) -> str:
    t = (str(text) if text is not None else "").strip().replace("\n", "／")
    return t if len(t) <= limit else t[:limit] + "…"

def _extract_keywords_ja(s: str, k: int = 2) -> str:
    if not s: return ""
    parts = re.split(r"[、。.\s　/｜|;:・,\n]+", str(s))
    parts = [p for p in parts if p and len(p) >= 2]
    return "／".join(parts[:k])

def _build_process(origin_sum: str, sim_sum: str, seeds: str, approach: str) -> str:
    origin_kw = _extract_keywords_ja(origin_sum, k=2) or "主要論点抽出"
    sim_kw    = _extract_keywords_ja(sim_sum,    k=2) or "参考事例抽出"
    seeds_kw  = _shorten(seeds, 24) if seeds else ""
    if approach == "直接的な課題解決アイデア":
        steps = [f"元ニュースの論点化（{origin_kw}）", f"類似先の解決要素抽出（{sim_kw}）",
                 "ギャップ特定→実装案設計" + (f"→シーズ適用（{seeds_kw}）" if seeds_kw else "")]
    elif approach == "隣接領域への応用アイデア":
        steps = [f"元課題の要素分解（{origin_kw}）", f"隣接領域の事例参照（{sim_kw}）",
                 "制約差を調整→応用試作" + (f"→シーズ適用（{seeds_kw}）" if seeds_kw else "")]
    else:
        steps = [f"本質機能の抽出（{origin_kw}）", f"異分野の類比探索（{sim_kw}）",
                 "類比転用→PoC検証" + (f"→シーズ適用（{seeds_kw}）" if seeds_kw else "")]
    return " → ".join(steps)

def _fallback_solution(need: str, seed: str, origin_sum: str, sim_sum: str) -> str:
    parts = []
    if need:
        parts.append(f"【ターゲット/ユースケース】ニーズ『{need}』に対して導入前→運用→保守の3局面で適用を具体化。")
    if seed:
        parts.append(f"【機能群】『{seed}』の特性を活かし、①取得/前処理 ②分析/推定 ③通知/UI ④運用/保守 ⑤ログ/監査。")
    src = (sim_sum or "") + "\n" + (origin_sum or "")
    if src.strip():
        parts.append(f"【実装方針】データ/センサー/API/アルゴ/非機能を段階導入。参考: {_shorten(src,200)}")
    parts.append("【検証】PoC→限定公開→本番。KPI: 精度/スループット/誤警報/ROI。")
    return "\n".join(parts)

def _fallback_process(need: str, seed: str, approach: str) -> str:
    return f"1) ニーズ『{_shorten(need,32)}』を分解 2) シーズ『{_shorten(seed,32)}』の強みを対応付け 3) 要件/制約整理 4) 機能設計 5) PoCでKPI検証 6) 改善して本番展開"

# ---------- OpenAI 呼び出し（Responses API: input_text 必須） ----------
def _openai_call(prompt: str, system: str = "", model: str = "o3",
                 temperature: float = 0.7, api_key: str = None, extra: dict = None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def _extract_text(resp):
        if hasattr(resp, "output_text") and resp.output_text:
            return str(resp.output_text).strip()
        texts=[]
        out = getattr(resp, "output", None)
        if out:
            for seg in out:
                if getattr(seg, "type", None) == "output_text":
                    texts.append(getattr(seg, "text", "") or "")
        return "\n".join([t for t in texts if t]).strip()

    if model.startswith("o"):  # o3/o4系 → Responses API
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort":"medium"},
                temperature=temperature,
                input=[
                    {"role":"system","content":[{"type":"input_text","text": system or "You are a helpful Japanese product ideation assistant."}]},
                    {"role":"user","content":[{"type":"input_text","text": prompt}]},
                ],
                max_output_tokens=1200,
                **(extra or {})
            )
            text = _extract_text(resp)
            if text: return text
        except Exception:
            pass  # 後段へ

    # フォールバック（chat.completions）
    try:
        fallback_model = ("gpt-4o-mini" if model.startswith("o") else model)
        comp = client.chat.completions.create(
            model=fallback_model,
            temperature=temperature,
            messages=[
                {"role":"system","content": system or "You are a helpful Japanese product ideation assistant."},
                {"role":"user","content": prompt},
            ],
            **(extra or {})
        )
        if comp and comp.choices:
            return comp.choices[0].message.content.strip()
    except Exception:
        pass
    raise RuntimeError("OpenAI 応答が取得できませんでした。")

# ---------- LLM プロンプト ----------
PROMPT_TEMPLATE_TRIPLE = r"""
# 役割（Role）
あなたは、B2B市場を主戦場とする電子部品メーカーの、経験豊富かつ革新的な製品開発エンジニアです。同時に、技術に詳しくない営業担当者にもアイデアの価値を分かりやすく説明できる、優れたコミュニケーターでもあります。

# 目的（Objective）
インプットとして提示された「新しい技術ニーズ」と、それを解決しうる「既存の技術シーズ」のペアに基づき、具体的なソリューションを体系的に創出します。さらに、そのアイデアが生まれた思考プロセスも明確に言語化します。

# 実行プロセス（Execution Process）
AIは以下の思考プロセスを厳密に実行してください。
1.  インプットの理解: まず、与えられた「新しい技術ニーズ」と「既存の技術シーズ」を深く理解します。
2.  多様なアイデアへの展開: 技術シーズを元に、後述の「アウトプット」で定義された3つの異なる切り口で、具体的な製品・サービスのソリューションを考案します。
3.  プロセスの言語化: 各ソリューションについて、なぜその発想に至ったのか、シーズのどの特性がどう活かされているのかを、非技術者にも理解できるように平易な言葉で説明します。

# インプット（Input）
- 新しい技術ニーズ:
\"\"\"
{new_technology_need}
\"\"\"
- 既存の技術シーズ:
\"\"\"
{existing_solution_seed}
\"\"\"


# アウトプット（Output）
- 形式: 必ず "generated_ideas" というキーを持つ単一のJSONオブジェクトとして出力してください。そのキーの値は、後述のガイドラインに基づいた3つのアイデアを含むJSON配列とします。注釈や前置きは一切不要です。
- 品質: 各項目は、実務の技術検討に耐えうるレベルまで、具体的かつ詳細に記述してください。
- アイデアの切り口（ガイドライン）: 以下の3つの視点を必ず含めてください。
    1. 直接的な課題解決アイデア
    2. 隣接領域への応用アイデア
    3. 異分野への転用アイデア

```json
{{
  "generated_ideas": [
    {{
      "アイデア名": "（技術転用の内容がわかるユニークな名称）",
      "アイデアの切り口": "1. 直接的な課題解決",
      "具体的なソリューション": "（ガイドライン1に沿ったソリューションの詳細）",
      "アイデア創出のプロセス": "（このソリューションが生まれた思考の過程や着眼点を、技術に詳しくない人にも分かるように説明。例：「シーズの『〇〇』という特性が、ニーズの『△△』という課題を解決する鍵になると考えました。そこで…」）"
    }},
    {{
      "アイデア名": "（技術転用の内容がわかるユニークな名称）",
      "アイデアの切り口": "2. 隣接領域への応用",
      "具体的なソリューション": "（ガイドライン2に沿ったソリューションの詳細）",
      "アイデア創出のプロセス": "（同上）"
    }},
    {{
      "アイデア名": "（技術転用の内容がわかるユニークな名称）",
      "アイデアの切り口": "3. 異分野への転用",
      "具体的なソリューション": "（ガイドライン3に沿ったソリューションの詳細）",
      "アイデア創出のプロセス": "（同上）"
    }}
  ]
}}
"""

def _gen_three_ideas_with_llm_triple(tech_need: str, tech_seed: str, model: str, temperature: float, api_key: str) -> List[Dict[str,str]]:
    system = "日本語で、実務で使える具体性で、冗長になりすぎないように。"
    tries = 0
    seen = set()
    ideas_all: List[Dict[str,str]] = []
    while tries < 2:
        prompt = PROMPT_TEMPLATE_TRIPLE.format(
            new_technology_need=tech_need or "（不明）",
            existing_solution_seed=tech_seed or "（不明）",
        )
        raw = _openai_call(prompt.strip(), system=system, model=model, temperature=temperature, api_key=api_key)
        m = re.search(r'\{.*\}', raw, re.S)
        ideas = []
        if m:
            try:
                data = json.loads(m.group(0))
                ideas = data.get("generated_ideas", [])
            except Exception:
                ideas = []
        norm=[]
        for it in ideas or []:
            name = str(it.get("アイデア名","")).strip()
            if not name or name in seen: continue
            seen.add(name)
            for k in ["アイデア名","アイデアの切り口","具体的なソリューション","アイデア創出のプロセス"]:
                if k in it and isinstance(it[k], str): it[k]=it[k].strip()
            norm.append(it)
        ideas_all.extend(norm)
        if len(ideas_all)>=3: break
        tries += 1
    return ideas_all[:3]

def _gen_with_llm(origin_title, origin_sum, sim_sum, seeds, approach, model, temperature, api_key, tech_need="", tech_seed=""):
    system = "日本語で簡潔に。ただし実務で使える具体性は確保する。"
    prompt = f"""
あなたは事業化アイデアの発想支援アシスタントです。以下のインプットから「{approach}」の観点で 1 件の提案を作成してください。
出力は必ず JSON 1行で: {{"idea_name":"…","solution":"…","process":"…"}}

[入力技術ニーズ]
{tech_need or "（不明）"}

[入力製品シーズ]
{tech_seed or "（不明）"}

[元ニュース要約]（参考）
{origin_sum or "（無し）"}

[マッチング先の要約]（参考）
{sim_sum or "（無し）"}

[要件]
- idea_name: 20字以内。差別化点が分かる短い名前
- solution : ターゲット/ユースケース/機能群/実装方針（データ・センサー・API・アルゴ・UI・非機能）/リスクと回避策/検証プランまで。構造的に詳述
- process  : 観察→仮説→要素分解→設計（シーズ適用）→検証 の番号付き短文（案件ごとに言い回しを変える）
- JSON以外は出力しない
"""
    try:
        text = _openai_call(prompt.strip(), system=system, model=model, temperature=temperature, api_key=api_key)
        m = re.search(r'\{.*\}', text, re.S)
        if not m: raise ValueError("JSON抽出不可")
        data = json.loads(m.group(0))
        idea = (data.get("idea_name") or f"{(tech_need or origin_title)[:16]}の案").strip()
        solution = (data.get("solution") or "").strip() or _fallback_solution(tech_need, tech_seed, origin_sum, sim_sum)
        process = (data.get("process") or "").strip() or _fallback_process(tech_need, tech_seed, approach)
        return {"idea_name": idea, "solution": solution, "process": process}
    except Exception:
        return {
            "idea_name": f"{(tech_need or origin_title)[:16]}の案",
            "solution": _fallback_solution(tech_need, tech_seed, origin_sum, sim_sum),
            "process": _fallback_process(tech_need, tech_seed, approach),
        }

# ---------- エントリポイント ----------
def run(result_df: pd.DataFrame, params: dict):
    """
    既定：各マッチ行ごとに 3案（直接/隣接/異分野）を生成
    params:
      - openai_model (default "o3")
      - openai_api_key
      - temperature (default 0.7)
      - triple_mode (default True)  # False で固定切り口モード
      - fixed_approach / ideas_per_match  # triple_mode=False の場合のみ使用
    """
    out_cols = OUTPUT_COLUMNS[:]
    if result_df is None or result_df.empty:
        return pd.DataFrame(columns=out_cols)

    work = result_df.copy()
    for col in REQ_INPUT_COLUMNS:
        if col not in work.columns: work[col] = ""
    if "similarity" in work.columns:
        try: work = work.sort_values("similarity", ascending=False)
        except Exception: pass

    p = params or {}
    model = p.get("openai_model", "o3")
    temperature = float(p.get("temperature", 0.7))
    api_key = p.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    triple_mode = bool(p.get("triple_mode", True))
    fixed_approach = p.get("fixed_approach", "直接的な課題解決アイデア")
    ideas_per_match = int(p.get("ideas_per_match", 1))

    groups = work.groupby(["番号","元技術ニーズのニュース名"], dropna=False)
    out_rows: List[Dict[str,str]] = []

    for (no, origin_title), g in groups:
        for _, row in g.iterrows():
            origin_sum = str(row.get("元技術ニーズの要約","") or "")
            sim_sum    = str(row.get("類似先ニュースの要約","") or "")
            tech_need  = str(row.get(COLUMN_TECH_NEED,"") or "")
            tech_seed  = str(row.get(COLUMN_TECH_SEED,"") or "")

            if triple_mode and api_key:
                ideas = _gen_three_ideas_with_llm_triple(
                    tech_need=tech_need, tech_seed=tech_seed,
                    model=model, temperature=temperature, api_key=api_key
                )
                if ideas:
                    for idea in ideas:
                        out_rows.append({
                            "番号": no,
                            "元シート名": str(origin_title)[:60],
                            "入力技術ニーズ": tech_need,
                            "入力製品シーズ": tech_seed,
                            "アイデア名": idea.get("アイデア名",""),
                            "アイデアの切り口": idea.get("アイデアの切り口",""),
                            "具体的なソリューション": idea.get("具体的なソリューション",""),
                            "アイデア創出のプロセス": idea.get("アイデア創出のプロセス",""),
                        })
                    continue
                # LLM失敗時フォールバック（3案）
                for i, approach in enumerate(APPROACHES,1):
                    out_rows.append({
                        "番号": no,
                        "元シート名": str(origin_title)[:60],
                        "入力技術ニーズ": tech_need,
                        "入力製品シーズ": tech_seed,
                        "アイデア名": f"{_shorten(tech_need or origin_title,16)}の案{i}",
                        "アイデアの切り口": f"{i}. " + ["直接的な課題解決","隣接領域への応用","異分野への転用"][i-1],
                        "具体的なソリューション": _fallback_solution(tech_need, tech_seed, origin_sum, sim_sum),
                        "アイデア創出のプロセス": _fallback_process(tech_need, tech_seed, approach),
                    })
                continue

            # 固定切り口モード
            for _ in range(max(1, ideas_per_match)):
                out = _gen_with_llm(
                    origin_title=origin_title, origin_sum=origin_sum, sim_sum=sim_sum, seeds=tech_seed,
                    approach=fixed_approach, model=model, temperature=temperature, api_key=api_key,
                    tech_need=tech_need, tech_seed=tech_seed
                )
                out_rows.append({
                    "番号": no,
                    "元シート名": str(origin_title)[:60],
                    "入力技術ニーズ": tech_need,
                    "入力製品シーズ": tech_seed,
                    "アイデア名": out["idea_name"],
                    "アイデアの切り口": fixed_approach,
                    "具体的なソリューション": out["solution"],
                    "アイデア創出のプロセス": out["process"],
                })

    return pd.DataFrame(out_rows, columns=out_cols)
