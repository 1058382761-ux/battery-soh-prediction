import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# 设置页面
st.set_page_config(page_title="蓄电池SOH预测", layout="wide")
st.title("蓄电池健康状态(SOH)预测模型")

# ====================== 加载模型数据 ======================
try:
    # 加载训练好的模型和参数
    with open('model_data.pkl', 'rb') as f:
        model_data = pickle.load(f)
    best_model = model_data['best_model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    test_metrics = model_data['test_metrics']
except FileNotFoundError:
    st.warning("⚠️ 未找到模型文件，请先运行 soh_prediction.py 训练模型！")
    st.stop()

# ====================== 侧边栏参数输入 ======================
st.sidebar.header("输入电池参数")
cycle = st.sidebar.number_input("循环次数", min_value=1, max_value=5000, value=100)
voltage = st.sidebar.number_input("电压(V)", min_value=2.0, max_value=4.5, value=3.2, step=0.01)
current = st.sidebar.number_input("电流(A)", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
temperature = st.sidebar.number_input("温度(℃)", min_value=0, max_value=60, value=25)
capacity = st.sidebar.number_input("容量(Ah)", min_value=5.0, max_value=15.0, value=9.5, step=0.01)

# 计算衍生特征（与训练时的特征工程逻辑一致）
window_size = 5
capacity_decay_rate = (10 - capacity) / cycle if cycle > 0 else 0
voltage_trend = 0.001 * cycle
temp_change_rate = 0.01 * cycle
current_std = 0.05
capacity_per_cycle = capacity / cycle

# 构建特征向量
input_features = np.array([[
    cycle, voltage, current, temperature, capacity,
    capacity_decay_rate, voltage_trend, temp_change_rate, current_std, capacity_per_cycle
]])

# ====================== 预测逻辑 ======================
if st.sidebar.button("预测SOH"):
    # 标准化输入特征
    input_scaled = scaler.transform(input_features)
    # 预测SOH
    soh_pred = best_model.predict(input_scaled)[0]
    soh_pred = np.clip(soh_pred, 0, 100)
    
    # 显示预测结果
    st.success(f"🎯 预测的蓄电池健康状态(SOH): {soh_pred:.2f}%")
    
    # 显示健康状态等级
    st.subheader("健康状态评估")
    if soh_pred >= 90:
        st.success("✅ 健康状态：优秀（电池性能良好，无需更换）")
    elif soh_pred >= 80:
        st.info("🟡 健康状态：良好（电池性能正常，建议定期监测）")
    elif soh_pred >= 70:
        st.warning("🟠 健康状态：一般（电池性能衰减，建议准备更换）")
    else:
        st.error("🔴 健康状态：较差（电池性能严重衰减，需立即更换）")

# ====================== 模型性能展示 ======================
st.header("📊 模型性能评估")
col1, col2, col3 = st.columns(3)
col1.metric("均方误差(MSE)", f"{test_metrics['mse']:.4f}", help="越小越好，衡量预测值与真实值的平方误差")
col2.metric("平均绝对误差(MAE)", f"{test_metrics['mae']:.4f}", help="越小越好，衡量预测值与真实值的平均绝对误差")
col3.metric("决定系数(R²)", f"{test_metrics['r2']:.4f}", help="越接近1越好，衡量模型拟合效果")

# ====================== 预测效果可视化 ======================
st.header("📈 预测效果可视化")
try:
    st.image("soh_prediction.png", caption="蓄电池SOH真实值与预测值对比", use_column_width=True)
except:
    st.warning("⚠️ 未找到预测效果图，请先运行 soh_prediction.py 生成！")

# ====================== 使用说明 ======================
st.sidebar.markdown("---")
st.sidebar.info("""
### 使用说明
1. 输入电池的各项参数
2. 点击「预测SOH」按钮
3. 查看预测结果和健康状态评估

### 指标说明
- SOH：电池健康状态（0-100%）
- MSE：均方误差，越小预测越准确
- R²：决定系数，越接近1模型效果越好
""")