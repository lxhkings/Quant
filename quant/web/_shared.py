"""Web 多页共用组件：术语表、数据模式选择器、步骤导航头部。

只渲染 UI，不含计算逻辑。被 app.py 与 pages/ 下各页引用。
"""

import streamlit as st

# 术语 → 白话，给各页 number_input/checkbox 的 help= 用
GLOSSARY = {
    "lookback": "回看多少天算动量，常用 252（约 1 年）",
    "skip": "跳过最近几天，避开短期反转，常用 21",
    "window": "计算窗口天数，常用 200",
    "horizon": "预测未来多少天的收益，常用 21",
    "quantiles": "按因子值把股票分几组看单调性，常用 5",
    "IC": "因子值与未来收益的相关性，越高预测力越强",
    "ICIR": "IC 均值 / IC 波动，>0.5 算不错",
    "分位档数": "按因子值把股票分几组看单调性，常用 5",
    "共线性": "因子间太像（相关>0.7）会重复下注",
    "DSR": "抗过拟合指标，扣掉多次试验的运气成分",
    "行业中性化": "去掉行业影响，只看因子本身",
}

_MODE_HELP = (
    "research：留出近 2 年数据不看，防偷看（研究阶段用）。\n\n"
    "full：用全部数据，仅最终选股时用。"
)


def mode_selector() -> str:
    """在 sidebar 渲染数据模式选择器，返回选中的 mode。"""
    return st.sidebar.selectbox(
        "数据模式", ["research", "full"], index=0, help=_MODE_HELP
    )


def step_header(step_no: int, title: str, what: str,
                prev: tuple[str, str] | None,
                next: tuple[str, str] | None) -> None:
    """渲染步骤页统一头部：第 N 步 / 5 + 上下步链接 + 这步干啥。

    prev/next 为 (page_path, label) 或 None。page_path 相对当前页面所在目录。
    """
    st.caption(f"第 {step_no} 步 / 5 · {title}")
    c1, c2 = st.columns(2)
    if prev:
        c1.page_link(prev[0], label=f"◀ {prev[1]}")
    if next:
        c2.page_link(next[0], label=f"{next[1]} ▶")
    st.info(f"**这步干啥**：{what}")
