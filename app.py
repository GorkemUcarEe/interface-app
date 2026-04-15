import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Page configuration
st.set_page_config(page_title="Predictive Maintenance Interface", layout="wide")
st.title("Machine Warning System and Model Evaluation")


# --- 1. DATA LOADING FUNCTIONS ---

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


@st.cache_data
def load_mb_metrics():
    try:
        return pd.read_csv("Moving_Bracket_Metrics.csv")
    except FileNotFoundError:
        return pd.DataFrame()


df_metrics = load_metrics_data()
df_scores = load_score_data()
df_all_preds = load_all_predictions()
df_mb_metrics = load_mb_metrics()


# --- HELPER: MOVING BRACKET FUNCTION ---
def select_moving_bracket(df_group, start_last_m, stop_last_n, strict=False):
    g = df_group.sort_values("Time").copy()
    L = len(g)
    if strict and L < start_last_m: return g.iloc[0:0].copy()
    start_idx = max(0, L - start_last_m)
    stop_idx = max(0, L - stop_last_n)
    if stop_idx < start_idx: stop_idx = start_idx
    return g.iloc[start_idx:stop_idx].copy()


WINDOW_M = 495
WINDOW_N = 0

# --- 2. SIDEBAR FILTERS ---

st.sidebar.header("🔍 Global Model Filtering")

if not df_metrics.empty:
    val_list = ["All"] + sorted(list(df_metrics["Validation"].unique()))
    val_secim = st.sidebar.selectbox("Validation Technique:", val_list)

    norm_list = ["All"] + sorted(list(df_metrics["Normalization"].unique()))
    norm_secim = st.sidebar.selectbox("Normalization:", norm_list)

    model_list = ["All"] + sorted(list(df_metrics["Model"].unique()))
    model_secim = st.sidebar.selectbox("Model Algorithm:", model_list)

    filtered_df = df_metrics.copy()
    if val_secim != "All": filtered_df = filtered_df[filtered_df["Validation"] == val_secim]
    if norm_secim != "All": filtered_df = filtered_df[filtered_df["Normalization"] == norm_secim]
    if model_secim != "All": filtered_df = filtered_df[filtered_df["Model"] == model_secim]

    filtered_mb_df = df_mb_metrics.copy() if not df_mb_metrics.empty else pd.DataFrame()
    if not filtered_mb_df.empty:
        if val_secim != "All": filtered_mb_df = filtered_mb_df[filtered_mb_df["Validation"] == val_secim]
        if norm_secim != "All": filtered_mb_df = filtered_mb_df[filtered_mb_df["Normalization"] == norm_secim]
        if model_secim != "All": filtered_mb_df = filtered_mb_df[filtered_mb_df["Model"] == model_secim]
else:
    filtered_df = pd.DataFrame()
    filtered_mb_df = pd.DataFrame()

# --- 3. TABS ---

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ Training Metrics",
    "🏆 Test Results",
    "📈 Timeline Graphs",
    "🎯 Score",
    "🔎 Moving Bracket"
])

# ==========================================
# TAB 1: TRAINING METRICS
# ==========================================
with tab1:
    st.header("Training Set Performances")
    if not filtered_df.empty:
        train_cols = ["Validation", "Normalization", "Model", "Train_Accuracy", "Train_F1"]
        cols_to_show = [c for c in train_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[cols_to_show].sort_values(by="Train_F1", ascending=False), use_container_width=True)
    else:
        st.warning("No data found for the selected filters.")

# ==========================================
# TAB 2: TEST RESULTS & CONFUSION MATRIX
# ==========================================
with tab2:
    st.header("Test Set Performance and Confusion Matrix")
    if not filtered_df.empty:
        test_cols = ["Validation", "Normalization", "Model", "Test_Accuracy", "Test_F1", "Test_Precision",
                     "Test_Recall"]
        cols_to_show = [c for c in test_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[cols_to_show].sort_values(by="Test_F1", ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("🧩 Confusion Matrix Analysis")

        combo_list = filtered_df.apply(lambda x: f"{x['Model']} | Norm: {x['Normalization']} | Val: {x['Validation']}",
                                       axis=1).tolist()
        if combo_list:
            selected_combo_name = st.selectbox("Select combination for Matrix:", combo_list, key="cm_tab2")
            selected_row = filtered_df.iloc[combo_list.index(selected_combo_name)]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("True Positive (TP)", int(selected_row['TP']))
                st.metric("True Negative (TN)", int(selected_row['TN']))
                st.metric("False Positive (FP)", int(selected_row['FP']))
                st.metric("False Negative (FN)", int(selected_row['FN']))
            with c2:
                cm_matrix = [[int(selected_row['TN']), int(selected_row['FP'])],
                             [int(selected_row['FN']), int(selected_row['TP'])]]
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"],
                            yticklabels=["True: 0", "True: 1"])
                ax_cm.set_title(f"CM: {selected_row['Model']} ({selected_row['Normalization']})")
                st.pyplot(fig_cm)
    else:
        st.warning("No data found.")

# ==========================================
# TAB 3: TIMELINE GRAPHS
# ==========================================
with tab3:
    st.header("Prediction Graphs & Local Threshold Selection")
    if not df_all_preds.empty:
        local_proba_threshold = st.slider("Select Probability Threshold (T) for this Graph:", 0.0, 1.0, 0.50, 0.05)
        st.markdown("---")

        graph_filter = df_all_preds.copy()
        if val_secim != "All": graph_filter = graph_filter[graph_filter["Validation"] == val_secim]
        if norm_secim != "All": graph_filter = graph_filter[graph_filter["Normalization"] == norm_secim]
        if model_secim != "All": graph_filter = graph_filter[graph_filter["Model"] == model_secim]

        available_combos = graph_filter[["Model", "Normalization", "Validation"]].drop_duplicates()
        if len(available_combos) > 1:
            combo_names = available_combos.apply(lambda x: f"{x['Model']} | {x['Normalization']} | {x['Validation']}",
                                                 axis=1).tolist()
            picked_combo = st.selectbox("Visualization Model:", combo_names)
            m_val, n_val, v_val = picked_combo.split(" | ")
            plot_df = graph_filter[(graph_filter["Model"] == m_val) & (graph_filter["Normalization"] == n_val) & (
                        graph_filter["Validation"] == v_val)]
        elif len(available_combos) == 1:
            plot_df = graph_filter
        else:
            plot_df = pd.DataFrame()

        if not plot_df.empty:
            data_nos = sorted(plot_df["Data_No"].unique())
            secili_data_no = st.selectbox("Select Machine (Data_No):", data_nos)
            demo_data = plot_df[plot_df["Data_No"] == secili_data_no].copy()
            demo_data["dynamic_warning"] = (demo_data["pred_proba"] >= local_proba_threshold).astype(int)

            fig_tl, ax_tl = plt.subplots(figsize=(12, 4))
            ax_tl.plot(demo_data["Time"], demo_data["pred_proba"], label="Prediction Probability", color="blue",
                       alpha=0.5)
            ax_tl.plot(demo_data["Time"], demo_data["warning_flag"], label="True Warning Flag", color="orange",
                       linewidth=2.5)
            ax_tl.axhline(local_proba_threshold, linestyle="--", color="red",
                          label=f"Threshold (T={local_proba_threshold})")

            alarms = demo_data[demo_data["dynamic_warning"] == 1]
            if not alarms.empty: ax_tl.scatter(alarms["Time"], alarms["pred_proba"], color="red", marker="x", s=20,
                                               label="Alarm Triggered")

            ax_tl.set_title(f"Timeline for Machine {secili_data_no} (Model: {plot_df['Model'].iloc[0]})")
            ax_tl.set_xlabel("Time (Cycles)")
            ax_tl.set_ylabel("Probability / Status")
            ax_tl.legend(loc='upper left', bbox_to_anchor=(1, 1))
            st.pyplot(fig_tl)

            true_t = demo_data.loc[demo_data["warning_flag"] == 1, "Time"]
            pred_t = demo_data.loc[demo_data["dynamic_warning"] == 1, "Time"]
            if not true_t.empty and not pred_t.empty:
                delta = pred_t.iloc[0] - true_t.iloc[0]
                if delta < 0:
                    st.success(f"✅ Early Warning: Model detected failure {abs(delta):.1f} cycles BEFORE actual event.")
                else:
                    st.error(f"⚠️ Late Warning: Model detected failure {delta:.1f} cycles AFTER actual event.")
        else:
            st.warning("No prediction data matches the current filters.")
    else:
        st.error("All_Model_Predictions.csv not found.")

# ==========================================
# TAB 4: SCORE
# ==========================================
with tab4:
    st.header("Asymmetric Scoring and Alarm Optimization")
    if not df_scores.empty:
        st.markdown("### 🎚️ Tolerance Window Setting")
        if "Tolerance" in df_scores.columns:
            tol_list = sorted(list(df_scores["Tolerance"].unique()))
            selected_tol = st.select_slider("Select Tolerance Window Size (± Cycles):", options=tol_list,
                                            value=tol_list[len(tol_list) // 2])
            sc_filter = df_scores[df_scores["Tolerance"] == selected_tol].copy()
        else:
            st.warning("Tolerance column not found in CSV.")
            sc_filter = df_scores.copy()

        st.markdown("---")
        if val_secim != "All": sc_filter = sc_filter[sc_filter["Validation"] == val_secim]
        if norm_secim != "All": sc_filter = sc_filter[sc_filter["Normalization"] == norm_secim]
        if model_secim != "All": sc_filter = sc_filter[sc_filter["Model"] == model_secim]

        col1, col2 = st.columns(2)
        with col1:
            t_secim = st.selectbox("Select Probability Threshold (T):",
                                   ["All"] + sorted(list(df_scores["Threshold"].unique())))
        with col2:
            h_secim = st.selectbox("Select Hits (K):", ["All"] + sorted(list(df_scores["Consecutive_Hits"].unique())))

        if t_secim != "All": sc_filter = sc_filter[sc_filter["Threshold"] == t_secim]
        if h_secim != "All": sc_filter = sc_filter[sc_filter["Consecutive_Hits"] == h_secim]

        st.dataframe(sc_filter.sort_values(by="Score", ascending=True), use_container_width=True)

        if not sc_filter.empty:
            best = sc_filter.sort_values(by="Score", ascending=True).iloc[0]
            st.success(f"🏆 Best Result: {best['Model']} | Norm: {best['Normalization']} | Score: {best['Score']:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Penalty Score", f"{best['Score']:.2f}")
            m2.metric("Missed Failures", int(best['Missed_Alerts']))
            m3.metric("Tolerable Early Rate", best['Tol_Early_Rate'])
            m4.metric("Tolerable Late Rate", best['Tol_Late_Rate'])
    else:
        st.warning("Model_Skorlari_GridSearch.csv not found.")

# ==========================================
# TAB 5: MOVING BRACKET (FULL DETAILS)
# ==========================================
with tab5:
    st.header("Moving Bracket Evaluation")
    st.info(
        f"Metrics below are evaluated strictly on the critical window (M={WINDOW_M}, N={WINDOW_N}). Easy early predictions are excluded.")

    if not filtered_mb_df.empty:
        mb_cols = ["Validation", "Normalization", "Model", "MB_Accuracy", "MB_F1", "MB_Precision", "MB_Recall"]
        cols_to_show = [c for c in mb_cols if c in filtered_mb_df.columns]
        st.dataframe(filtered_mb_df[cols_to_show].sort_values(by="MB_F1", ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("🧩 Detailed Per-Machine Analysis")

        combo_list_mb = filtered_mb_df.apply(
            lambda x: f"{x['Model']} | Norm: {x['Normalization']} | Val: {x['Validation']}", axis=1).tolist()
        if combo_list_mb:
            selected_combo_mb = st.selectbox("Select model combination for detailed analysis:", combo_list_mb,
                                             key="detailed_tab5")
            selected_row_mb = filtered_mb_df.iloc[combo_list_mb.index(selected_combo_mb)]

            # 1. Confusion Matrix (Overall)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("True Positive (TP)", int(selected_row_mb['TP']))
                st.metric("True Negative (TN)", int(selected_row_mb['TN']))
                st.metric("False Positive (FP)", int(selected_row_mb['FP']))
                st.metric("False Negative (FN)", int(selected_row_mb['FN']))
            with c2:
                cm_matrix_mb = [[int(selected_row_mb['TN']), int(selected_row_mb['FP'])],
                                [int(selected_row_mb['FN']), int(selected_row_mb['TP'])]]
                fig_cm_mb, ax_cm_mb = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_matrix_mb, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"],
                            yticklabels=["True: 0", "True: 1"])
                ax_cm_mb.set_title(f"Overall CM: {selected_row_mb['Model']}")
                st.pyplot(fig_cm_mb)

            st.markdown("---")

            # Extract Model Info from Selection
            m_str, n_str, v_str = selected_combo_mb.split(" | ")
            m_val = m_str.strip()
            n_val = n_str.replace("Norm: ", "").strip()
            v_val = v_str.replace("Val: ", "").strip()

            # Process Data Dynamically for Selected Model
            model_preds = df_all_preds[(df_all_preds["Model"] == m_val) &
                                       (df_all_preds["Normalization"] == n_val) &
                                       (df_all_preds["Validation"] == v_val)].copy()

            if not model_preds.empty:
                # Calculate original lengths before windowing
                group_lengths = model_preds.groupby("Data_No").size().reset_index(name="group_total_rows_before_window")

                # SAFELY Apply moving bracket logic (avoiding Pandas apply KeyError)
                limited_frames = []
                for _, g in model_preds.sort_values(["Data_No", "Time"]).groupby("Data_No"):
                    limited_frames.append(select_moving_bracket(g, WINDOW_M, WINDOW_N))

                if limited_frames:
                    limited_preds = pd.concat(limited_frames, ignore_index=True)
                else:
                    limited_preds = pd.DataFrame(columns=model_preds.columns)

                if not limited_preds.empty:
                    limited_preds["y_pred"] = (limited_preds["pred_proba"] >= 0.5).astype(int)

                    # --- TABLE 1: Rows Kept ---
                    st.markdown("#### 1. Rows Kept per Data_No")
                    rows_kept = limited_preds.groupby("Data_No").agg(
                        rows_kept=("Data_No", "size"),
                        time_min=("Time", "min"),
                        time_max=("Time", "max")
                    ).reset_index()
                    rows_kept["requested_rows"] = WINDOW_M - WINDOW_N
                    st.dataframe(rows_kept, use_container_width=True)

                    group_metric_rows = []
                    group_range_rows = []

                    for data_no, g in limited_preds.groupby("Data_No"):
                        y_true = g["warning_flag"].values
                        y_pred = g["y_pred"].values

                        acc = accuracy_score(y_true, y_pred)
                        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                                           zero_division=0)
                        pos_rate = np.mean(y_true)

                        true_alert_idx = g.loc[g["warning_flag"] == 1, "Time"]
                        true_alert_t = float(true_alert_idx.iloc[0]) if len(true_alert_idx) > 0 else np.nan

                        pred_alert_idx = g.loc[g["y_pred"] == 1, "Time"]
                        pred_alert_t = float(pred_alert_idx.iloc[0]) if len(pred_alert_idx) > 0 else np.nan

                        total_len = \
                        group_lengths.loc[group_lengths["Data_No"] == data_no, "group_total_rows_before_window"].iloc[0]
                        actual_rows = len(g)

                        group_metric_rows.append({
                            "Data_No": data_no, "n_rows": actual_rows, "positive_rate": pos_rate,
                            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
                        })

                        delta = pred_alert_t - true_alert_t if (
                                    pd.notna(true_alert_t) and pd.notna(pred_alert_t)) else np.nan

                        effective_start = min(WINDOW_M, total_len) if actual_rows > 0 else np.nan
                        effective_stop = min(WINDOW_N, total_len) if actual_rows > 0 else np.nan

                        group_range_rows.append({
                            "Data_No": data_no, "total_rows_before_window": total_len,
                            "actual_start_last": effective_start, "actual_end_last_exclusive": effective_stop,
                            "actual_rows_kept": actual_rows, "true_alert_time": true_alert_t,
                            "pred_alert_time": pred_alert_t, "delta_time(pred-true)": delta,
                            "mean_pred_proba": g["pred_proba"].mean(), "max_pred_proba": g["pred_proba"].max(),
                            "min_pred_proba": g["pred_proba"].min()
                        })

                    # --- TABLE 2: Metrics per Data_No ---
                    st.markdown("#### 2. Per-Machine Metrics")
                    st.dataframe(pd.DataFrame(group_metric_rows), use_container_width=True)

                    # --- TABLE 3: Window & Alert Times per Data_No ---
                    st.markdown("#### 3. Per-Machine Window & Range Metrics")
                    st.dataframe(pd.DataFrame(group_range_rows), use_container_width=True)
                else:
                    st.warning("No data remains after applying the Moving Bracket window.")
            else:
                st.warning("No prediction details found for the selected model.")
    else:
        st.warning("Moving_Bracket_Metrics.csv not found. Please run the Jupyter Notebook update.")