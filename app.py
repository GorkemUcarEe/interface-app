import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa yapılandırması
st.set_page_config(page_title="Predictive Maintenance Interface", layout="wide")
st.title("Machine Warning System and Model Evaluation")


# --- 1. VERİ YÜKLEME FONKSİYONLARI ---

@st.cache_data
def load_sensor_data():
    try:
        return pd.read_csv("Test_Data_Filtered.csv")
    except FileNotFoundError:
        return pd.DataFrame()


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


@st.cache_data
def load_all_predictions():
    try:
        return pd.read_csv("All_Model_Predictions.csv")
    except FileNotFoundError:
        return pd.DataFrame()


# Verileri belleğe al
df_metrics = load_metrics_data()
df_scores = load_score_data()
df_all_preds = load_all_predictions()

# --- 2. SOL MENÜ (SIDEBAR) FİLTRELERİ ---

st.sidebar.header("🔍 Model Filtering")

if not df_metrics.empty:
    val_list = ["All"] + sorted(list(df_metrics["Validation"].unique()))
    val_secim = st.sidebar.selectbox("Validation Technique:", val_list)

    norm_list = ["All"] + sorted(list(df_metrics["Normalization"].unique()))
    norm_secim = st.sidebar.selectbox("Normalization:", norm_list)

    model_list = ["All"] + sorted(list(df_metrics["Model"].unique()))
    model_secim = st.sidebar.selectbox("Model Algorithm:", model_list)

    # Genel filtreleme işlemi (Tab 1, 2 ve 4 için)
    filtreli_df = df_metrics.copy()
    if val_secim != "All":
        filtreli_df = filtreli_df[filtreli_df["Validation"] == val_secim]
    if norm_secim != "All":
        filtreli_df = filtreli_df[filtreli_df["Normalization"] == norm_secim]
    if model_secim != "All":
        filtreli_df = filtreli_df[filtreli_df["Model"] == model_secim]
else:
    filtreli_df = pd.DataFrame()

# --- 3. SEKMELER (TABS) ---

tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Training Metrics",
    "🏆 Test Results",
    "📈 Timeline Graphs",
    "🎯 Score"
])

# ==========================================
# SEKME 1: TRAINING METRICS
# ==========================================
with tab1:
    st.header("Training Set Performances")
    if not filtreli_df.empty:
        train_cols = ["Validation", "Normalization", "Model", "Train_Accuracy", "Train_F1"]
        cols_to_show = [c for c in train_cols if c in filtreli_df.columns]
        st.dataframe(filtreli_df[cols_to_show].sort_values(by="Train_F1", ascending=False), use_container_width=True)
    else:
        st.warning("No data found for the selected filters.")

# ==========================================
# SEKME 2: TEST RESULTS & CONFUSION MATRIX
# ==========================================
with tab2:
    st.header("Test Set Performance and Confusion Matrix")
    if not filtreli_df.empty:
        test_cols = ["Validation", "Normalization", "Model", "Test_Accuracy", "Test_F1", "Test_Precision",
                     "Test_Recall"]
        cols_to_show = [c for c in test_cols if c in filtreli_df.columns]
        st.dataframe(filtreli_df[cols_to_show].sort_values(by="Test_F1", ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("🧩 Confusion Matrix Analysis")

        kombinasyon_listesi = filtreli_df.apply(
            lambda
                x: f"{x['Model']} | Norm: {x['Normalization']} | Val: {x['Validation']} (Test F1: {x.get('Test_F1', 0):.4f})",
            axis=1).tolist()

        if kombinasyon_listesi:
            secilen_isim = st.selectbox("Select combination for Matrix:", kombinasyon_listesi)
            secili_satir = filtreli_df.iloc[kombinasyon_listesi.index(secilen_isim)]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("True Positive (TP)", int(secili_satir['TP']))
                st.metric("True Negative (TN)", int(secili_satir['TN']))
                st.metric("False Positive (FP)", int(secili_satir['FP']))
                st.metric("False Negative (FN)", int(secili_satir['FN']))

            with c2:
                cm_matrix = [[int(secili_satir['TN']), int(secili_satir['FP'])],
                             [int(secili_satir['FN']), int(secili_satir['TP'])]]
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Pred: 0", "Pred: 1"],
                            yticklabels=["True: 0", "True: 1"])
                ax_cm.set_title(f"CM: {secili_satir['Model']} ({secili_satir['Normalization']})")
                st.pyplot(fig_cm)
    else:
        st.warning("No data found.")

# ==========================================
# SEKME 3: TIMELINE GRAPHS (EŞİK KONTROLÜ BURADA)
# ==========================================
with tab3:
    st.header("Prediction Graphs & Threshold Analysis")

    if not df_all_preds.empty:
        # 1. EŞİK KONTROLÜ (Bu sekmenin üstünde yer alır)
        local_proba_threshold = st.slider(
            "Select Probability Threshold (T) for this Graph:",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05
        )

        st.markdown("---")

        # Sidebar filtrelerini grafik verisine uygula
        graph_filter = df_all_preds.copy()
        if val_secim != "All":
            graph_filter = graph_filter[graph_filter["Validation"] == val_secim]
        if norm_secim != "All":
            graph_filter = graph_filter[graph_filter["Normalization"] == norm_secim]
        if model_secim != "All":
            graph_filter = graph_filter[graph_filter["Model"] == model_secim]

        available_combos = graph_filter[["Model", "Normalization", "Validation"]].drop_duplicates()

        if len(available_combos) > 1:
            st.info("💡 Multiple models match your sidebar filters. Pick one to visualize:")
            combo_names = available_combos.apply(lambda x: f"{x['Model']} | {x['Normalization']} | {x['Validation']}",
                                                 axis=1).tolist()
            picked_combo = st.selectbox("Visualization Model:", combo_names)
            m, n, v = picked_combo.split(" | ")
            plot_df = graph_filter[(graph_filter["Model"] == m) &
                                   (graph_filter["Normalization"] == n) &
                                   (graph_filter["Validation"] == v)]
        elif len(available_combos) == 1:
            plot_df = graph_filter
        else:
            plot_df = pd.DataFrame()

        if not plot_df.empty:
            data_nos = sorted(plot_df["Data_No"].unique())
            secili_data_no = st.selectbox("Select Machine (Data_No):", data_nos)

            demo_data = plot_df[plot_df["Data_No"] == secili_data_no].copy()

            # Dinamik uyarı bayrağını seçilen yerel Threshold'a göre hesapla
            demo_data["dynamic_warning"] = (demo_data["pred_proba"] >= local_proba_threshold).astype(int)

            fig_tl, ax_tl = plt.subplots(figsize=(12, 4))
            ax_tl.plot(demo_data["Time"], demo_data["pred_proba"], label="Prediction Probability", color="blue",
                       alpha=0.5)
            ax_tl.plot(demo_data["Time"], demo_data["warning_flag"], label="True Warning Flag", color="orange",
                       linewidth=2.5)

            # Seçilen eşik değerini kırmızı kesik çizgi olarak çiz
            ax_tl.axhline(local_proba_threshold, linestyle="--", color="red",
                          label=f"Threshold (T={local_proba_threshold})")

            # Eşiği geçen noktaları işaretle
            alarms = demo_data[demo_data["dynamic_warning"] == 1]
            if not alarms.empty:
                ax_tl.scatter(alarms["Time"], alarms["pred_proba"], color="red", marker="x", s=20,
                              label="Alarm Triggered")

            ax_tl.set_title(f"Timeline for Machine {secili_data_no} (Model: {plot_df['Model'].iloc[0]})")
            ax_tl.set_xlabel("Time")
            ax_tl.set_ylabel("Probability / Status")
            ax_tl.legend(loc='upper left')
            st.pyplot(fig_tl)

            # Erken uyarı özeti
            true_t = demo_data.loc[demo_data["warning_flag"] == 1, "Time"]
            pred_t = demo_data.loc[demo_data["dynamic_warning"] == 1, "Time"]

            if not true_t.empty and not pred_t.empty:
                delta = pred_t.iloc[0] - true_t.iloc[0]
                if delta < 0:
                    st.success(f"✅ Early Warning: Model detected failure {abs(delta):.1f} units BEFORE actual event.")
                else:
                    st.error(f"⚠️ Late Warning: Model detected failure {delta:.1f} units AFTER actual event.")
        else:
            st.warning("No prediction data matches the current filters.")
    else:
        st.error("All_Model_Predictions.csv not found.")

# ==========================================
# SEKME 4: SCORE (ASİMETRİK SKOR)
# ==========================================
with tab4:
    st.header("Asymmetric Scoring and Alarm Optimization")
    if not df_scores.empty:
        sc_filter = df_scores.copy()
        if val_secim != "All":
            sc_filter = sc_filter[sc_filter["Validation"] == val_secim]
        if norm_secim != "All":
            sc_filter = sc_filter[sc_filter["Normalization"] == norm_secim]
        if model_secim != "All":
            sc_filter = sc_filter[sc_filter["Model"] == model_secim]

        c1, c2 = st.columns(2)
        with c1:
            t_list = ["All"] + sorted(list(df_scores["Threshold"].unique()))
            t_secim = st.selectbox("Select Optimized Threshold (T):", t_list)
        with c2:
            h_list = ["All"] + sorted(list(df_scores["Consecutive_Hits"].unique()))
            h_secim = st.selectbox("Select Hits (K):", h_list)

        if t_secim != "All":
            sc_filter = sc_filter[sc_filter["Threshold"] == t_secim]
        if h_secim != "All":
            sc_filter = sc_filter[sc_filter["Consecutive_Hits"] == h_secim]

        st.dataframe(sc_filter.sort_values(by="Score", ascending=True), use_container_width=True)

        if not sc_filter.empty:
            best = sc_filter.sort_values(by="Score", ascending=True).iloc[0]
            st.success(
                f"🏆 Optimal Configuration: {best['Model']} | Norm: {best['Normalization']} | Score: {best['Score']:.2f}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Score", f"{best['Score']:.2f}")
            m2.metric("Missed", int(best['Missed_Alerts']))
            m3.metric("Early Rate", best['Tol_Early_Rate'])
            m4.metric("Late Rate", best['Tol_Late_Rate'])
    else:
        st.warning("Model_Skorlari_GridSearch.csv not found.")