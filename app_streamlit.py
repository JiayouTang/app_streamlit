import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib, json
from pathlib import Path
import matplotlib.pyplot as plt

# ===============================
# 页面配置
# ===============================
st.set_page_config(page_title="胆结石预测与解释", layout="wide")

# ===============================
# 加载资源
# ===============================
ART_DIR = Path(r"C:\Users\唐嘉佑\Desktop\科研论文\model\20251226-数据集2结果图-01\artifacts")

model = joblib.load(ART_DIR / "final_pipe.joblib")

with open(ART_DIR / "meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

bg = pd.read_csv(ART_DIR / "bg_sample.csv")

# ===============================
# 页面标题
# ===============================
st.title("胆结石预测（单样本）与 SHAP 瀑布图")
st.markdown("请在左侧输入病人指标，点击按钮进行预测。")

# ===============================
# 输入区
# ===============================
with st.sidebar:
    st.header("输入参数")
    inputs = {}

    for c in meta["num_cols"]:
        rng = meta["num_ranges"][c]
        inputs[c] = st.number_input(
            c,
            min_value=float(rng["min"]),
            max_value=float(rng["max"]),
            value=float(rng["mean"])
        )

    for c in meta["cat_cols"]:
        options = meta["cat_values"].get(c, [])
        inputs[c] = st.selectbox(c, options) if options else st.text_input(c)

    submit = st.button("预测并解释")

# ===============================
# 预测 + SHAP
# ===============================
if submit:
    x = pd.DataFrame([inputs])[meta["selected_features"]]
    prob = model.predict_proba(x)[0, 1]
    pred = int(prob >= 0.5)

    st.subheader("预测结果")
    st.metric("患病概率", f"{prob:.3f}")
    st.write("预测类别：", "**阳性(1)**" if pred else "**阴性(0)**")

    # ===============================
    # SHAP
    # ===============================
    st.subheader("SHAP 瀑布图（单样本）")

    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    Z_x = pre.transform(x)
    Z_bg = pre.transform(bg)

    is_tree = any(k in clf.__class__.__name__.lower()
                  for k in ["tree", "forest", "xgb", "boost"])

    if is_tree:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(Z_x)

        # 二分类：取正类
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0]  # (n_features, n_outputs) or (n_features,)

        if shap_values.ndim == 2:
            shap_values = shap_values[:, 1]  # 只取正类

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

    else:
        explainer = shap.KernelExplainer(
            lambda X_: clf.predict_proba(X_)[:, 1],
            Z_bg
        )
        shap_values = explainer.shap_values(Z_x, nsamples=200)[0]
        base_value = explainer.expected_value

    feature_names = pre.get_feature_names_out().tolist()

    # ===============================
    # Explanation（单样本 + 单输出）
    # ===============================
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=Z_x[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(2, 1),dpi=200)
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("请在左侧输入参数后点击 **预测并解释**")

st.caption("⚠ 本系统仅用于科研演示，临床应用请结合专业医师判断。")
