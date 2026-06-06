import streamlit as st
import pandas as pd

# Import modules
from data_loader import load_data, get_subset
from model import build_model
from optimiser import solve_model, extract_solution, build_doctor_schedule
from visualisation import create_heatmap, create_expected_load_graph, create_idle_graph, create_overload_graph, create_delay_distribution


# ===============================
# INIT DATA (LOAD ONCE)
# ===============================
@st.cache_data
def get_data():
    return load_data()


df_global = get_data()


# ===============================
# PAGE 1 — DATA INPUT
# ===============================
def data_input_page():

    st.title("📊 Data Input")

    df = df_global

    st.write("### Raw Data")
    st.dataframe(df.head())

    # -------------------------
    # SESSION STATE INIT
    # -------------------------
    if "n_patients" not in st.session_state:
        st.session_state.n_patients = 70

    # -------------------------
    # INPUT (STATE-BOUND PROPERLY)
    # -------------------------
    st.number_input(
        "Number of patients",
        min_value=10,
        max_value=len(df),
        step=1,
        key="n_patients"
    )

    # ALWAYS derive from session state (single source of truth)
    n = st.session_state.n_patients

    # -------------------------
    # SUBSET (ONLY UPDATE WHEN N CHANGES)
    # -------------------------
    # First time initialisation
    if "subset" not in st.session_state:
        st.session_state.subset = get_subset(df, n)
        st.session_state.subset_n = n

    # Only regenerate if user changes n
    elif st.session_state.subset_n != n:
        st.session_state.subset = get_subset(df, n)
        st.session_state.subset_n = n

    subset = st.session_state.subset

    st.write("### Selected Appointments")
    st.dataframe(subset[["AppointmentID", "no_show_prob"]])
    st.info(f"Current dataset contains {len(subset)} appointments.")


# ===============================
# PAGE 2 — OPTIMISATION
# ===============================
def optimisation_page():

    st.title("⚙️ Optimisation")

    if "subset" not in st.session_state:
        st.warning("Go to Data Input first.")
        return

    df = st.session_state["subset"]

    # ---------------------------
    # USER INPUTS (PERSISTENT KEYED STATE)
    # ---------------------------
    num_doctors = st.slider("Doctors", 1, 10, 2, key="num_doctors")
    doctors = [f"D{i+1}" for i in range(num_doctors)]

    overbooking = st.slider("Overbooking", 1, 5, 2, key="overbooking")
    load_cap = st.slider("Load Cap", 1.0, 2.0, 1.2, key="load_cap")

    st.subheader("Weights")

    w_wait = st.slider("Waiting Weight", 0.0, 5.0, 1.0, key="w_wait")
    w_idle = st.slider("Idle Weight", 0.0, 5.0, 1.0, key="w_idle")
    w_overtime = st.slider("Overtime Weight", 0.0, 10.0, 1.0, key="w_overtime")
    w_unscheduled = st.slider("Unscheduled Penalty", 0.0, 20.0, 1.0, key="w_unscheduled")
    w_risk = st.slider("Risk Weight", 0.0, 5.0, 1.0, key="w_risk")

    # ---------------------------
    # RUN MODEL
    # ---------------------------
    if st.button("🚀 Run Optimisation"):

        time_slots = list(range(24))
        start_time = pd.Timestamp("08:00")

        slot_labels = [
            (start_time + pd.Timedelta(minutes=10*t)).strftime("%H:%M")
            for t in time_slots
        ]

        params = {
            "overbooking": overbooking,
            "load_cap": load_cap,
            "w_wait": w_wait,
            "w_idle": w_idle,
            "w_overtime": w_overtime,
            "w_unscheduled": w_unscheduled,
            "w_risk": w_risk
        }

        model, x, u, idle, overtime = build_model(df, doctors, time_slots, params)

        if not solve_model(model):
            st.error("No optimal solution found.")
            return

        schedule_df, unscheduled = extract_solution(
            model, x, u, doctors, time_slots, df, slot_labels
        )

        # =========================
        # PERFORMANCE METRICS
        # =========================

        total_idle = sum(idle[d, t].X for d in doctors for t in time_slots)
        total_overtime = sum(overtime[d, t].X for d in doctors for t in time_slots)
        total_delay_penalty = schedule_df["TimeSlot"].sum()
        scheduled_patients = len(schedule_df)

        avg_underutilisation = total_idle / (len(doctors) * len(time_slots))
        avg_overload = total_overtime / (len(doctors) * len(time_slots))
        avg_delay_penalty = schedule_df["TimeSlot"].mean()

        st.success(f"✨ Objective Value: {model.objVal:.4f}")
        st.write(f"❗ Unscheduled Patients: {len(unscheduled)}")

        st.subheader("📋 Schedule Table")
        st.dataframe(schedule_df)

        st.subheader("🩺 Doctor Appointment Schedule")
        doctor_schedule_df = build_doctor_schedule(schedule_df)
        st.dataframe(doctor_schedule_df)

        st.subheader("📊 Heatmap")
        fig = create_heatmap(schedule_df, doctors, slot_labels)
        st.pyplot(fig)

        st.subheader("📈 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Delay Penalty", f"{avg_delay_penalty:.2f}")
        col2.metric("Average Expected Underutilisation", f"{avg_underutilisation:.2f}")
        col3.metric("Average Expected Overload", f"{avg_overload:.2f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Delay Penalty", f"{total_delay_penalty:.2f}")
        col2.metric("Total Expected Underutilisation", f"{total_idle:.2f}")
        col3.metric("Total Expected Overload", f"{total_overtime:.2f}")

        st.subheader("📈 Expected Load by Time Slot")
        fig_load = create_expected_load_graph(schedule_df, df, doctors, time_slots)
        st.pyplot(fig_load)

        st.subheader("📉 Underutilisation by Time Slot")
        fig_idle = create_idle_graph(idle, doctors, time_slots)
        st.pyplot(fig_idle)

        st.subheader("📈 Overload by Time Slot")
        fig_overload = create_overload_graph(overtime, doctors, time_slots)
        st.pyplot(fig_overload)

        st.subheader("⏳ Delay Distribution")
        fig_delay = create_delay_distribution(schedule_df)
        st.pyplot(fig_delay)

# ===============================
# MAIN NAVIGATION
# ===============================
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Data Input", "Optimisation"])

if page == "Data Input":
    data_input_page()
else:
    optimisation_page()