import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. 資料載入與前處理 (鎖定標題四大維度)
# ==========================================
def run_analysis():
    print("正在讀取資料...")
    # 請確認資料檔案與此 .py 檔放在同一個資料夾
    try:
        df = pd.read_csv('youth_smoking_drug_data_10000_rows_expanded.csv')
    except FileNotFoundError:
        print("錯誤：找不到資料檔案！請確認資料庫名稱是否為 'youth_smoking_drug_data_10000_rows_expanded.csv'")
        return

    # 篩選高中職階段 (15-19 歲)
    df_youth = df[df['Age_Group'] == '15-19'].copy()

    # 編碼類別變數
    le = LabelEncoder()
    df_youth['SES_Code'] = le.fit_transform(df_youth['Socioeconomic_Status']) 
    df_youth['School_Prog_Code'] = df_youth['School_Programs'].map({'Yes': 1, 'No': 0}) 
    df_youth['Sub_Edu_Code'] = df_youth['Substance_Education'].map({'Yes': 1, 'No': 0}) 

    # 特徵選取 (標題四大維度：環境、教育、心理、社經)
    features = ['Peer_Influence', 'Parental_Supervision', 'Mental_Health', 
                'SES_Code', 'School_Prog_Code', 'Sub_Edu_Code']

    # ==========================================
    # 2. 監督式學習：因素重要性分析 (成因探討)
    # ==========================================
    print("正在執行監督式學習 (隨機森林)...")
    X_s = df_youth[features]
    y_s = df_youth['Smoking_Prevalence']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_s, y_s)

    # 繪製影響因子排行圖
    importance = pd.DataFrame({'Factor': features, 'Importance': rf.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='Importance', y='Factor', palette='viridis')
    plt.title('Influencing Factors: Supervised Learning (Smoking Prediction)')
    plt.tight_layout()
    plt.savefig('supervised_importance.png')
    print("- 圖表已存為 'supervised_importance.png'")

    # ==========================================
    # 3. 非監督式學習：精準分群標定 (精準預防)
    # ==========================================
    print("正在執行非監督式學習 (K-means 分群)...")
    cluster_cols = features + ['Smoking_Prevalence', 'Drug_Experimentation']
    X_u = df_youth[cluster_cols]

    # 標準化數據
    scaler = StandardScaler()
    X_u_scaled = scaler.fit_transform(X_u)

    # 執行分群
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_youth['Cluster'] = kmeans.fit_predict(X_u_scaled)

    # 繪製分群結果散佈圖
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_youth, x='Peer_Influence', y='Smoking_Prevalence', 
                    hue='Cluster', palette='Set1', alpha=0.6)
    plt.title('Student Risk Profiling: Unsupervised Learning (K-means)')
    plt.tight_layout()
    plt.savefig('unsupervised_clusters.png')
    print("- 圖表已存為 'unsupervised_clusters.png'")

    # 儲存分群摘要結果 CSV
    df_youth.groupby('Cluster')[cluster_cols].mean().to_csv('cluster_profiles.csv')
    print("- 數據摘要已存為 'cluster_profiles.csv'")
    print("="*30)
    print("分析成功完成！現在你可以將檔案上傳到 GitHub 了。")

if _name_ == "_main_":
    run_analysis()