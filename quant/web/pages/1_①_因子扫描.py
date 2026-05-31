"""第 1 步 · 因子扫描（批量排行榜）。"""

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import step_header

st.set_page_config(page_title="① 因子扫描", layout="wide")

step_header(
    1, "因子扫描",
    "一键体检全库因子，按 ICIR 排序，先看哪些有预测力、值得深挖。",
    prev=None, next=("2_②_因子工坊.py", "② 因子工坊"),
)
from quant.web._shared import mode_selector
mode = mode_selector()

st.caption("批量扫描写独立 scan 台账，不影响 DSR 主台账。")
if st.button("批量体检全部因子"):
    df = viewmodel.leaderboard(mode=mode)
    st.dataframe(df)
    st.success("ICIR 高的几个，去 ②因子工坊 深挖。")
