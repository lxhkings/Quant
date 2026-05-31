"""Quant 因子研究 Web —— 首页总览。

启动：uv run streamlit run quant/web/app.py
streamlit 自动发现同目录 pages/ 下的步骤页并生成 sidebar。
"""

import streamlit as st

st.set_page_config(page_title="Quant 因子研究", layout="wide")

st.title("🏠 Quant 因子研究 · 总览")
st.write(
    "多因子量化选股研究平台。下面是一条完整研究流水线，像漏斗一样："
    "想法多 → 层层筛 → 活下来的少数 → 组合产出。"
)

st.markdown(
    """
```
想法多 ┌─────────────────────────────────────────────┐ 信号少
       │ ①扫描  →  ②工坊  →  ③合成  →  ④闸门  → ⑤选股│
       │  粗筛     精测      增强      终验锁定   产出 │
       └─────────────────────────────────────────────┘
                 📒 台账：全程自动记录
```
"""
)

st.divider()

_CARDS = [
    ("pages/1_①_因子扫描.py", "① 因子扫描",
     "一键体检全库因子，按 ICIR 排序，找出哪些值得深挖。", "开局粗筛", "进入 ①"),
    ("pages/2_②_因子工坊.py", "② 因子工坊",
     "单因子深度体检：预测力 / 单调性 / 多空三关红绿灯。", "盯一个因子", "进入 ②"),
    ("pages/3_③_多因子合成.py", "③ 多因子合成",
     "几个因子加权合成，查共线性，回测。", "组合增强", "进入 ③"),
    ("pages/4_④_holdout闸门.py", "④ holdout 闸门",
     "⚠ 定稿因子最终验证，仅一次，跑后锁定。", "即将上线", "进入 ④"),
    ("pages/5_⑤_选股器.py", "⑤ 选股器",
     "多因子加权打分，产出今日买入池。", "要选股清单", "进入 ⑤"),
    ("pages/6_📒_台账.py", "📒 台账",
     "看历史每次试验记录。", "回溯", "查看 📒"),
]

cols = st.columns(2)
for i, (path, title, what, when, btn) in enumerate(_CARDS):
    with cols[i % 2].container(border=True):
        st.subheader(title)
        st.write(what)
        st.caption(f"何时用：{when}")
        st.page_link(path, label=btn)
