import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Predictive Maintenance Interface", layout="wide")
st.title("Machine Warning System and Model Evaluation")


# --- 1. VERİ YÜKLEME ---
@st.cache_data
def load_sensor_data():
    return pd.read_csv("Test_Data_Filtered.csv")


@st.cache_data
def load_metrics_data():
    try:
        return pd.read_csv("Model_Sonuclari_Tum.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_score_data():
    try:
        return pd.read_csv("Model_Skorlari_GridSearch.csv")
    except FileNotFoundError:
        return pd.DataFrame()


test_df_filtered = load_sensor_data()
df_metrics = load_metrics_data()
df_scores = load_score_data()

PROBA_THRESHOLD = 0.5

# --- 2. SOL MENÜ (SIDEBAR) FİLTRELERİ (Sekme 1, 2 ve 4 için) ---
st.sidebar.header("🔍 Model Filtering")
if not df_metrics.empty:
    val_secim = st.sidebar.selectbox("Validation Technique:", ["All"] + list(df_metrics["Validation"].unique()))
    norm_secim = st.sidebar.selectbox("Normalization:", ["All"] + list(df_metrics["Normalization"].unique()))
    model_secim = st.sidebar.selectbox("Model Algorithm:", ["All"] + list(df_metrics["Model"].unique()))

    filtreli_df = df_metrics.copy()
    if val_secim != "All":
        filtreli_df = filtreli_df[filtreli_df["Validation"] == val_secim]
    if norm_secim != "All":
        filtreli_df = filtreli_df[filtreli_df["Normalization"] == norm_secim]
    if model_secim != "All":
        filtreli_df = filtreli_df[filtreli_df["Model"] == model_secim]
else:
    filtreli_df = pd.DataFrame()

# --- 3. SEKMELER (TABS) - YENİ SIRA ---
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Training Metrics",
    "🏆 Test Results",
    "📈 Timeline Graphs",
    "🎯 Score"
])

# ==========================================
# SEKME 1: EĞİTİM (TRAIN) METRİKLERİ
# ==========================================
with tab1:
    st.header("Training Set Performances")
    if len(filtreli_df) > 0:
        train_kolonlar = ["Validation", "Normalization", "Model", "Train_Accuracy", "Train_F1"]
        mevcut_kolonlar = [col for col in train_kolonlar if col in filtreli_df.columns]
        st.dataframe(filtreli_df[mevcut_kolonlar].sort_values(by="Train_F1", ascending=False), use_container_width=True)
    else:
        st.warning("No Data.")

# ==========================================
# SEKME 2: TEST METRİKLERİ VE CONFUSION MATRIX
# ==========================================
with tab2:
    st.header("Test Set Performance and Confusion Matrix")
    if len(filtreli_df) > 0:
        test_kolonlar = ["Validation", "Normalization", "Model", "Test_Accuracy", "Test_F1", "Test_Precision",
                         "Test_Recall"]
        mevcut_test_kolonlari = [col for col in test_kolonlar if col in filtreli_df.columns]
        st.dataframe(filtreli_df[mevcut_test_kolonlari].sort_values(by="Test_F1", ascending=False),
                     use_container_width=True)

        st.markdown("---")
        st.subheader("🧩 Confusion Matrix Analysis")
        kombinasyon_listesi = filtreli_df.apply(
            lambda
                x: f"{x['Model']} | Norm: {x['Normalization']} | Val: {x['Validation']} (Test F1: {x.get('Test_F1', 0):.4f})",
            axis=1).tolist()
        secilen_isim = st.selectbox("Matrix Combination:", kombinasyon_listesi)
        secili_satir = filtreli_df.iloc[kombinasyon_listesi.index(secilen_isim)]

        cm_col1, cm_col2 = st.columns([1, 2])
        with cm_col1:
            st.metric("TP", int(secili_satir['TP']))
            st.metric("TN", int(secili_satir['TN']))
            st.metric("FP", int(secili_satir['FP']))
            st.metric("FN", int(secili_satir['FN']))
        with cm_col2:
            cm_matrix = [[int(secili_satir['TN']), int(secili_satir['FP'])],
                         [int(secili_satir['FN']), int(secili_satir['TP'])]]
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Prediction: Sound (0)", "Prediction: Malfunction (1)"],
                        yticklabels=["Fact: Sound (0)", "Fact: Malfunction (1)"])
            ax_cm.set_title(f"Confusion Matrix: {secili_satir['Model']} ({secili_satir['Normalization']})")
            st.pyplot(fig_cm)
    else:
        st.warning("No data.")

# ==========================================
# SEKME 3: ZAMAN SERİSİ
# ==========================================
with tab3:
    st.header("Prediction Graphs")
    remaining_data_nos = test_df_filtered["Data_No"].unique()
    secili_data_no = st.selectbox("Select the machine you wish to examine:", remaining_data_nos)
    demo = test_df_filtered[test_df_filtered["Data_No"] == secili_data_no].copy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(demo["Time"], demo["pred_proba"], label="pred_proba (warning)", color="blue")
    ax.plot(demo["Time"], demo["warning_flag"], label="true warning_flag", color="orange")
    ax.axhline(PROBA_THRESHOLD, linestyle="--", linewidth=1.5, color="red", label=f"Threshold ({PROBA_THRESHOLD})")
    ax.set_title(f"Timeline (Data_No={secili_data_no})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability / Flag")
    ax.legend()
    st.pyplot(fig)

# ==========================================
# SEKME 4: SKOR VE ALARM AYARLARI
# ==========================================
with tab4:
    st.header("Asymmetric Scoring and Alarm Optimization")

    if not df_scores.empty:
        st.write("The effect of different Probability Thresholds and Consecutive Hit settings on penalty scores:")

        # 1. Ana Sidebar'daki Filtreleri 4. Sekmeye de Uygula (Validation, Norm, Model)
        filtreli_skor_df = df_scores.copy()

        # Artık Validation kolonu olduğu için sol menüden K-Fold veya Holdout seçimi burayı da süzecek!
        if val_secim != "All" and "Validation" in filtreli_skor_df.columns:
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Validation"] == val_secim]
        if norm_secim != "All":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Normalization"] == norm_secim]
        if model_secim != "All":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Model"] == model_secim]

        # 2. Sadece bu sekmeye özel (Threshold ve Hits) filtreleri
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            thresh_secim = st.selectbox("Threshold:", ["All"] + sorted(list(df_scores["Threshold"].unique())))
        with sc_col2:
            hits_secim = st.selectbox("Hits:", ["All"] + sorted(list(df_scores["Consecutive_Hits"].unique())))

        if thresh_secim != "All":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Threshold"] == thresh_secim]
        if hits_secim != "All":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Consecutive_Hits"] == hits_secim]

        st.markdown(f"**Results found ({len(filtreli_skor_df)} count - Sorted by lowest score):**")
        st.dataframe(filtreli_skor_df.sort_values(by="Score", ascending=True), use_container_width=True)

        # En iyi sonucu (Skoru en düşük olan) öne çıkarma
        if len(filtreli_skor_df) > 0:
            best_score_row = filtreli_skor_df.sort_values(by="Score", ascending=True).iloc[0]

            # Seçilen validasyon yöntemini de başlıkta belirtelim
            st.success(f"🏆 The best score according to this filtering: **{best_score_row['Score']:.2f}** ile **{best_score_row['Model']}** (Norm: {best_score_row['Normalization']} | Val: {best_score_row['Validation']})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score (Lower is Better)", f"{best_score_row['Score']:.2f}")
            c2.metric("Missed failure", int(best_score_row['Missed_Alerts']))
            c3.metric("Tolerable early", best_score_row['Tol_Early_Rate'])
            c4.metric("Tolerable late", best_score_row['Tol_Late_Rate'])
    else:
        st.warning("The file Model_Skorlari_GridSearch.csv could not be found. Please run the grid search code in Jupyter Notebook.")