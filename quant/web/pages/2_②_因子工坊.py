"""第 2 步 · 因子工坊（单因子端到端体检）。"""

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import GLOSSARY, mode_selector, step_header

st.set_page_config(page_title="② 因子工坊", layout="wide")

step_header(
    2, "因子工坊",
    "对单个因子做 IC + 分位 + 多空体检，看它有没有预测力。"
    "跑完看报告卡顶部红绿灯三关是否通过。",
    prev=("1_①_因子扫描.py", "① 因子扫描"),
    next=("3_③_多因子合成.py", "③ 多因子合成"),
)
mode = mode_selector()

name = st.selectbox("因子", viewmodel.available_factors())

# 参数随因子显隐：momentum 用 lookback/skip，其余用 window
lookback = skip = window = None
c1, c2, c3 = st.columns(3)
if name == "momentum":
    lookback = c1.number_input("lookback", 1, 1000, 252, help=GLOSSARY["lookback"])
    skip = c2.number_input("skip", 0, 250, 21, help=GLOSSARY["skip"])
else:
    window = c1.number_input("window", 1, 1000, 200, help=GLOSSARY["window"])

horizon = c2.number_input("horizon", 1, 120, 21, help=GLOSSARY["horizon"])
quantiles = c3.number_input("分位档数", 2, 10, 5, help=GLOSSARY["分位档数"])
neutralize = c3.checkbox("行业中性化", help=GLOSSARY["行业中性化"])

if st.button("跑检验"):
    md = viewmodel.workshop(
        name,
        lookback=lookback if lookback is not None else 252,
        skip=skip if skip is not None else 21,
        window=window if window is not None else 200,
        horizon=horizon, quantiles=quantiles, mode=mode, neutralize=neutralize,
    )
    st.markdown(md)
