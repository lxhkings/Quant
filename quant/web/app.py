"""Quant 因子研究 Web —— 四页：因子工坊 / 多因子合成 / holdout 闸门 / 历史。

启动：uv run streamlit run quant/web/app.py
"""

from pathlib import Path

import streamlit as st

from quant.web import viewmodel

st.set_page_config(page_title="Quant 因子研究", layout="wide")
PAGES = ["因子工坊", "多因子合成", "holdout 闸门", "历史"]
page = st.sidebar.radio("页面", PAGES)
mode = st.sidebar.selectbox("数据模式", ["research", "full"], index=0)

if page == "因子工坊":
    st.header("因子工坊")
    name = st.selectbox("因子", viewmodel.available_factors())
    c1, c2, c3 = st.columns(3)
    lookback = c1.number_input("lookback", 1, 1000, 252)
    skip = c2.number_input("skip", 0, 250, 21)
    window = c3.number_input("window", 1, 1000, 200)
    horizon = c1.number_input("horizon", 1, 120, 21)
    quantiles = c2.number_input("分位档数", 2, 10, 5)
    neutralize = c3.checkbox("行业中性化")
    if st.button("跑检验"):
        md = viewmodel.workshop(
            name, lookback=lookback, skip=skip, window=window,
            horizon=horizon, quantiles=quantiles, mode=mode, neutralize=neutralize,
        )
        st.markdown(md)

elif page == "多因子合成":
    st.header("多因子合成")
    names = st.multiselect("因子（多选）", viewmodel.available_factors(),
                           default=["momentum", "ma_bias"])
    weighting = st.radio("加权法", ["equal", "ic"], horizontal=True)
    quantiles = st.number_input("分位档数", 2, 10, 5)
    if st.button("合成回测") and names:
        out = viewmodel.combine(names, weighting=weighting, quantiles=quantiles, mode=mode)
        st.subheader("权重")
        st.json(out["weights"])
        st.subheader("共线性预警（|相关|>=0.7）")
        st.write(out["warnings"] or "无")
        st.subheader("绩效")
        st.json(out["metrics"])
        st.line_chart(out["nav"])

elif page == "holdout 闸门":
    st.header("holdout 闸门")
    st.warning("定稿因子仅可在 holdout 跑一次，跑后锁定。")
    name = st.selectbox("定稿因子", viewmodel.available_factors())
    confirm = st.checkbox("我确认这是最终定稿，仅跑一次")
    if st.button("在 holdout 跑最终验证") and confirm:
        try:
            md = viewmodel.holdout_run(name)
            st.markdown(md)
            st.success("holdout 已消耗，因子已锁定。")
        except RuntimeError as e:
            st.error(str(e))

else:  # 历史
    st.header("历史试验台账")
    rows = viewmodel.history(Path("quant_out/ledger.jsonl"))
    if rows:
        st.dataframe(rows)
    else:
        st.info("暂无记录。先在因子工坊或回测里跑一次。")
