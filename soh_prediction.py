import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 数据预处理 ======================
def load_and_preprocess_data(file_path=None, generate_sample=True):
    """
    加载并预处理数据
    :param file_path: 数据文件路径
    :param generate_sample: 是否生成示例数据
    :return: 预处理后的DataFrame
    """
    # 生成示例数据（实际使用时替换为真实数据路径）
    if generate_sample:
        np.random.seed(42)
        n_samples = 2000
        cycle = np.arange(1, n_samples+1)
        # 基础容量随循环次数衰减
        base_capacity = 10 - 0.005 * cycle + np.random.normal(0, 0.1, n_samples)
        voltage = 3.2 + 0.001 * cycle + np.random.normal(0, 0.02, n_samples)
        current = 1.0 + np.random.normal(0, 0.05, n_samples)
        temperature = 25 + 0.01 * cycle + np.random.normal(0, 0.5, n_samples)
        # SOH计算：基于容量的百分比
        soh = (base_capacity / 10) * 100
        soh = np.clip(soh, 60, 100)  # 限制范围
        
        df = pd.DataFrame({
            'Cycle': cycle,
            'Voltage': voltage,
            'Current': current,
            'Temperature': temperature,
            'Capacity': base_capacity,
            'SOH': soh
        })
    else:
        df = pd.read_csv(file_path)
    
    # 处理缺失值
    df = df.dropna()
    
    # 处理异常值（3σ原则）
    numeric_cols = ['Voltage', 'Current', 'Temperature', 'Capacity', 'SOH']
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    return df

# ====================== 2. 特征工程 ======================
def feature_engineering(df):
    """
    特征工程：提取时间序列特征和衍生特征
    """
    # 复制数据
    df_features = df.copy()
    
    # 1. 容量衰减率（滑动窗口计算）
    window_size = 5
    df_features['Capacity_decay_rate'] = df_features['Capacity'].rolling(window=window_size).apply(
        lambda x: (x.iloc[0] - x.iloc[-1]) / window_size
    )
    
    # 2. 电压变化趋势
    df_features['Voltage_trend'] = df_features['Voltage'].rolling(window=window_size).mean().diff()
    
    # 3. 温度变化率
    df_features['Temp_change_rate'] = df_features['Temperature'].diff()
    
    # 4. 电流稳定性（标准差）
    df_features['Current_std'] = df_features['Current'].rolling(window=window_size).std()
    
    # 5. 容量与循环次数的比值
    df_features['Capacity_per_cycle'] = df_features['Capacity'] / df_features['Cycle']
    
    # 填充滑动窗口产生的缺失值
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    
    return df_features

# ====================== 3. 数据划分与标准化 ======================
def prepare_data(df):
    """
    数据划分和标准化
    """
    # 选择特征和目标变量
    feature_cols = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity',
                   'Capacity_decay_rate', 'Voltage_trend', 'Temp_change_rate', 'Current_std', 'Capacity_per_cycle']
    X = df[feature_cols]
    y = df['SOH']
    
    # 划分数据集：先分训练集(70%)和临时集(30%)，再将临时集分为验证集和测试集(各15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler, feature_cols)

# ====================== 4. 模型训练与调参 ======================
def train_models(X_train, y_train, X_val, y_val):
    """
    训练多个模型并选择最优模型
    """
    # 定义模型和参数网格
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'SVM': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    }
    
    # 训练和调参
    best_models = {}
    results = {}
    
    for name, config in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_models[name] = grid_search.best_estimator_
        
        # 在验证集上评估
        y_val_pred = grid_search.best_estimator_.predict(X_val)
        results[name] = {
            'mse': mean_squared_error(y_val, y_val_pred),
            'mae': mean_absolute_error(y_val, y_val_pred),
            'r2': r2_score(y_val, y_val_pred),
            'best_params': grid_search.best_params_
        }
        
        print(f"{name} - Validation R2: {results[name]['r2']:.4f}")
    
    # 选择最优模型（基于R2）
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = best_models[best_model_name]
    
    return best_model, best_model_name, results

# ====================== 5. 模型评估 ======================
def evaluate_model(model, X_test, y_test):
    """
    在测试集上评估模型性能
    """
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    
    # 排序后绘图（按循环次数）
    test_df = pd.DataFrame({
        'Cycle': X_test[:, 0],  # Cycle是第一个特征
        'True_SOH': y_test.values,
        'Pred_SOH': y_pred
    }).sort_values('Cycle')
    
    plt.plot(test_df['Cycle'], test_df['True_SOH'], label='真实SOH', color='blue', alpha=0.7)
    plt.plot(test_df['Cycle'], test_df['Pred_SOH'], label='预测SOH', color='red', alpha=0.7)
    plt.xlabel('循环次数')
    plt.ylabel('SOH (%)')
    plt.title('蓄电池SOH真实值与预测值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('soh_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

# ====================== 6. 主函数 ======================
def main():
    # 1. 数据预处理
    print("加载并预处理数据...")
    df = load_and_preprocess_data(generate_sample=True)
    
    # 2. 特征工程
    print("进行特征工程...")
    df_features = feature_engineering(df)
    
    # 3. 数据准备
    print("划分和标准化数据...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = prepare_data(df_features)
    
    # 4. 模型训练
    print("训练模型...")
    best_model, best_model_name, val_results = train_models(X_train, y_train, X_val, y_val)
    
    # 5. 模型评估
    print("评估模型...")
    test_metrics = evaluate_model(best_model, X_test, y_test)
    
    # 输出结果
    print("\n=== 模型性能评估结果 ===")
    print(f"最优模型: {best_model_name}")
    print("\n验证集性能:")
    for metric, value in val_results[best_model_name].items():
        if metric != 'best_params':
            print(f"{metric.upper()}: {value:.4f}")
    
    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # 返回关键结果（供app.py使用）
    return best_model, scaler, feature_cols, test_metrics

# 运行主程序（生成模型和评估指标）
if __name__ == "__main__":
    best_model, scaler, feature_cols, test_metrics = main()
    
    # 保存关键变量（方便app.py调用）
    import pickle
    with open('model_data.pkl', 'wb') as f:
        pickle.dump({
            'best_model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'test_metrics': test_metrics
        }, f)
    print("\n✅ 模型数据已保存到 model_data.pkl")