"""第 5 步 · 选股器（多因子加权打分 → 今日买入池）。"""

import altair as alt
import pandas as pd
import streamlit as st

from quant.web import viewmodel
from quant.web._shared import mode_selector, step_header

st.set_page_config(page_title="⑤ 选股器", layout="wide")

step_header(
    5, "选股器",
    "用多因子加权对最新截面打分，产出今日买入池。跑完可去 📒台账 回溯。",
    prev=("pages/4_④_holdout闸门.py", "④ holdout 闸门"),
    next=("pages/6_📒_台账.py", "📒 台账"),
)
mode = mode_selector()

names = st.multiselect("因子", viewmodel.available_factors(), default=["momentum"])
weights = {}
if names:
    cols = st.columns(len(names))
    for i, nm in enumerate(names):
        weights[nm] = cols[i].slider(f"{nm} 权重", 0, 100, 100 // len(names))
    st.caption(f"总权重 {sum(weights.values())}（提交时自动归一为 100%）")
top_n = st.number_input("买入池数量 top-N", 1, 100, 3)
neutralize = st.checkbox("行业中性化")

if st.button("选股") and names and sum(weights.values()) > 0:
    out = viewmodel.selector(names, weights, top_n=int(top_n),
                             neutralize=neutralize, mode=mode)
    t = out["table"]
    n_buy = int((t["zone"] == "buy").sum())
    st.success(f"今日买入池 {n_buy} 只 · 截面日 {out['as_of']}")

    st.subheader("归一权重")
    st.dataframe(pd.DataFrame(
        {"因子": list(out["weights"]), "权重": list(out["weights"].values())}
    ))

    chart = alt.Chart(t).mark_bar().encode(
        x=alt.X("score:Q", title="综合得分 (0-100)"),
        y=alt.Y("instrument_id:N", sort="-x", title=None),
        color=alt.Color(
            "zone:N",
            scale=alt.Scale(domain=["buy", "candidate"], range=["#2e8b2e", "#3b6fe0"]),
            legend=alt.Legend(title="池"),
        ),
        tooltip=["instrument_id", "score", "rank", "sector", "zone"],
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(t)
