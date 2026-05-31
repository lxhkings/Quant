"""第 4 步 · holdout 闸门（定稿因子最终验证，仅一次，跑后锁定）。"""

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import mode_selector, step_header

st.set_page_config(page_title="④ holdout 闸门", layout="wide")

step_header(
    4, "holdout 闸门",
    "对定稿因子在留出的 holdout 数据上做最终验证。这是不可逆操作。",
    prev=("3_③_多因子合成.py", "③ 多因子合成"),
    next=("5_⑤_选股器.py", "⑤ 选股器"),
)
mode_selector()  # 渲染以保持各页一致；holdout 固定用 holdout 数据，mode 不参与

st.error("⚠ 破坏性操作：每个因子仅可在 holdout 跑一次，跑后永久锁定，不可重来。")
st.warning("务必确认这是最终定稿因子，再继续。")

name = st.selectbox("定稿因子", viewmodel.available_factors())
confirm = st.checkbox("我确认这是最终定稿，仅跑一次")
if st.button("在 holdout 跑最终验证") and confirm:
    try:
        md = viewmodel.holdout_run(name)
        st.markdown(md)
        st.success("holdout 已消耗，因子已锁定。")
    except RuntimeError as e:
        st.error(str(e))
