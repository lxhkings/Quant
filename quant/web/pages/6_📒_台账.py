"""台账 · 全程试验记录（不属漏斗某一步）。"""

from pathlib import Path

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import mode_selector

st.set_page_config(page_title="📒 台账", layout="wide")

st.title("📒 台账")
st.caption("全程试验记录：工坊体检、回测、holdout 都会写在这里。")
mode_selector()

rows = viewmodel.history(Path("quant_out/ledger.jsonl"))
if rows:
    st.dataframe(rows)
else:
    st.info("暂无记录。先在 ②因子工坊 或 ④holdout闸门 里跑一次。")
