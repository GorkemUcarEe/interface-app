import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tahminci Bakım Arayüzü", layout="wide")
st.title("Makine Uyarı Sistemi ve Model Değerlendirmesi")


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

# --- 2. SOL MENÜ (SIDEBAR) FİLTRELERİ (Sekme 2 ve 3 için) ---
st.sidebar.header("🔍 Model Filtreleme")
if not df_metrics.empty:
    val_secim = st.sidebar.selectbox("Validation Tekniği:", ["Tümü"] + list(df_metrics["Validation"].unique()))
    norm_secim = st.sidebar.selectbox("Normalizasyon:", ["Tümü"] + list(df_metrics["Normalization"].unique()))
    model_secim = st.sidebar.selectbox("Model Algoritması:", ["Tümü"] + list(df_metrics["Model"].unique()))

    filtreli_df = df_metrics.copy()
    if val_secim != "Tümü":
        filtreli_df = filtreli_df[filtreli_df["Validation"] == val_secim]
    if norm_secim != "Tümü":
        filtreli_df = filtreli_df[filtreli_df["Normalization"] == norm_secim]
    if model_secim != "Tümü":
        filtreli_df = filtreli_df[filtreli_df["Model"] == model_secim]
else:
    filtreli_df = pd.DataFrame()

# --- 3. SEKMELER (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Zaman Serisi Grafikleri",
    "⚙️ Eğitim (Train) Metrikleri",
    "🏆 Test Sonuçları",
    "🎯 Skor ve Alarm Ayarları"  # YENİ SEKME
])

# ==========================================
# SEKME 1: ZAMAN SERİSİ
# ==========================================
with tab1:
    st.header("Makine Bazlı Tahmin Grafikleri")
    remaining_data_nos = test_df_filtered["Data_No"].unique()
    secili_data_no = st.selectbox("İncelemek istediğiniz makineyi seçiniz:", remaining_data_nos)
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
# SEKME 2: EĞİTİM (TRAIN) METRİKLERİ
# ==========================================
with tab2:
    st.header("Eğitim (Train) Seti Performansları")
    if len(filtreli_df) > 0:
        train_kolonlar = ["Validation", "Normalization", "Model", "Train_Accuracy", "Train_F1"]
        mevcut_kolonlar = [col for col in train_kolonlar if col in filtreli_df.columns]
        st.dataframe(filtreli_df[mevcut_kolonlar].sort_values(by="Train_F1", ascending=False), use_container_width=True)
    else:
        st.warning("Veri bulunamadı.")

# ==========================================
# SEKME 3: TEST METRİKLERİ VE CONFUSION MATRIX
# ==========================================
with tab3:
    st.header("Test Seti Performansı ve Karmaşıklık Matrisi")
    if len(filtreli_df) > 0:
        test_kolonlar = ["Validation", "Normalization", "Model", "Test_Accuracy", "Test_F1", "Test_Precision",
                         "Test_Recall"]
        mevcut_test_kolonlari = [col for col in test_kolonlar if col in filtreli_df.columns]
        st.dataframe(filtreli_df[mevcut_test_kolonlari].sort_values(by="Test_F1", ascending=False),
                     use_container_width=True)

        st.markdown("---")
        st.subheader("🧩 Confusion Matrix İncelemesi")
        kombinasyon_listesi = filtreli_df.apply(
            lambda
                x: f"{x['Model']} | Norm: {x['Normalization']} | Val: {x['Validation']} (Test F1: {x.get('Test_F1', 0):.4f})",
            axis=1).tolist()
        secilen_isim = st.selectbox("Matris Kombinasyonu:", kombinasyon_listesi)
        secili_satir = filtreli_df.iloc[kombinasyon_listesi.index(secilen_isim)]

        cm_col1, cm_col2 = st.columns([1, 2])
        with cm_col1:
            st.metric("Doğru Pozitif (TP)", int(secili_satir['TP']))
            st.metric("Doğru Negatif (TN)", int(secili_satir['TN']))
            st.metric("Yanlış Pozitif (FP)", int(secili_satir['FP']))
            st.metric("Yanlış Negatif (FN)", int(secili_satir['FN']))
        with cm_col2:
            cm_matrix = [[int(secili_satir['TN']), int(secili_satir['FP'])],
                         [int(secili_satir['FN']), int(secili_satir['TP'])]]
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Tahmin: Sağlam (0)", "Tahmin: Arıza (1)"],
                        yticklabels=["Gerçek: Sağlam (0)", "Gerçek: Arıza (1)"])
            ax_cm.set_title(f"Confusion Matrix: {secili_satir['Model']} ({secili_satir['Normalization']})")
            st.pyplot(fig_cm)
    else:
        st.warning("Veri bulunamadı.")

# ==========================================
# SEKME 4: SKOR VE ALARM AYARLARI
# ==========================================
with tab4:
    st.header("Asimetrik Skor ve Alarm Optimizasyonu")

    if not df_scores.empty:
        st.write("Farklı Olasılık Eşikleri ve Peş Peşe Vuruş ayarlarının ceza skorlarına etkisi:")

        # 1. Ana Sidebar'daki Filtreleri 4. Sekmeye de Uygula (Validation, Norm, Model)
        filtreli_skor_df = df_scores.copy()

        # Artık Validation kolonu olduğu için sol menüden K-Fold veya Holdout seçimi burayı da süzecek!
        if val_secim != "Tümü" and "Validation" in filtreli_skor_df.columns:
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Validation"] == val_secim]
        if norm_secim != "Tümü":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Normalization"] == norm_secim]
        if model_secim != "Tümü":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Model"] == model_secim]

        # 2. Sadece bu sekmeye özel (Threshold ve Hits) filtreleri
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            thresh_secim = st.selectbox("Olasılık Eşiği (Threshold):", ["Tümü"] + sorted(list(df_scores["Threshold"].unique())))
        with sc_col2:
            hits_secim = st.selectbox("Peş Peşe Vuruş (Hits):", ["Tümü"] + sorted(list(df_scores["Consecutive_Hits"].unique())))

        if thresh_secim != "Tümü":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Threshold"] == thresh_secim]
        if hits_secim != "Tümü":
            filtreli_skor_df = filtreli_skor_df[filtreli_skor_df["Consecutive_Hits"] == hits_secim]

        st.markdown(f"**Bulunan Sonuçlar ({len(filtreli_skor_df)} adet - En düşük skora göre sıralı):**")
        st.dataframe(filtreli_skor_df.sort_values(by="Score", ascending=True), use_container_width=True)

        # En iyi sonucu (Skoru en düşük olan) öne çıkarma
        if len(filtreli_skor_df) > 0:
            best_score_row = filtreli_skor_df.sort_values(by="Score", ascending=True).iloc[0]

            # Seçilen validasyon yöntemini de başlıkta belirtelim
            st.success(f"🏆 Bu filtrelere göre en iyi skor **{best_score_row['Score']:.2f}** ile **{best_score_row['Model']}** (Norm: {best_score_row['Normalization']} | Val: {best_score_row['Validation']}) modeline ait.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Skor (Düşük Daha İyi)", f"{best_score_row['Score']:.2f}")
            c2.metric("Kaçırılan Arıza", int(best_score_row['Missed_Alerts']))
            c3.metric("Kabul Edilebilir Erken", best_score_row['Tol_Early_Rate'])
            c4.metric("Kabul Edilebilir Geç", best_score_row['Tol_Late_Rate'])
    else:
        st.warning("Model_Skorlari_GridSearch.csv dosyası bulunamadı. Lütfen Jupyter Notebook'ta grid search kodunu çalıştırın.")