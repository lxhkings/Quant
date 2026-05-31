"""第 3 步 · 多因子合成（加权合成 + 共线性预警 + 回测）。"""

import pandas as pd
import streamlit as st

from quant.web import viewmodel
from quant.web._shared import GLOSSARY, mode_selector, step_header

st.set_page_config(page_title="③ 多因子合成", layout="wide")

step_header(
    3, "多因子合成",
    "把几个因子加权合成一个综合分，查因子间共线性，再回测组合表现。",
    prev=("pages/2_②_因子工坊.py", "② 因子工坊"),
    next=("pages/4_④_holdout闸门.py", "④ holdout 闸门"),
)
mode = mode_selector()

names = st.multiselect("因子（多选）", viewmodel.available_factors(),
                       default=["momentum", "ma_bias"])
weighting = st.radio("加权法", ["equal", "ic"], horizontal=True)
quantiles = st.number_input("分位档数", 2, 10, 5, help=GLOSSARY["分位档数"])

if st.button("合成回测") and names:
    out = viewmodel.combine(names, weighting=weighting, quantiles=quantiles, mode=mode)

    # 通俗结论
    warns = out["warnings"]
    sharpe = out["metrics"]["sharpe"]
    col_msg = "因子间无明显共线性" if not warns else f"⚠ 共线性预警 {len(warns)} 处（相关≥0.7，可能重复下注）"
    sharpe_msg = "Sharpe 达标（≥1）" if sharpe >= 1 else "Sharpe 偏低（<1）"
    st.info(f"**通俗结论**：{col_msg}；{sharpe_msg}。")

    st.subheader("权重")
    st.dataframe(pd.DataFrame(
        {"因子": list(out["weights"]), "权重": list(out["weights"].values())}
    ))

    st.subheader("共线性预警（|相关|≥0.7）")
    st.write(warns or "无")

    st.subheader("绩效")
    m = out["metrics"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("年化收益", f"{m['annual_return']:.1%}")
    g2.metric("Sharpe", f"{m['sharpe']:.2f}")
    g3.metric("最大回撤", f"{m['max_drawdown']:.1%}")
    g4.metric("月胜率", f"{m['monthly_win_rate']:.1%}")

    st.line_chart(out["nav"])
