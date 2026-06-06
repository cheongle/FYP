import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap(schedule_df, doctors, slot_labels):

    heatmap_data = pd.DataFrame(
        0, index=doctors, columns=slot_labels
    )

    for _, row in schedule_df.iterrows():
        heatmap_data.loc[row["Doctor"], row["Time"]] += 1

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    plt.xticks(rotation=45)
    plt.title("Optimised Schedule")

    return fig

def create_expected_load_graph(schedule_df, df, doctors, time_slots):

    # AppointmentID -> show probability
    show_prob = {
        row["AppointmentID"]: 1 - row["no_show_prob"]
        for _, row in df.iterrows()
    }

    load_data = []

    for d in doctors:

        loads = []

        for t in time_slots:

            patients = schedule_df[
                (schedule_df["Doctor"] == d)
                &
                (schedule_df["TimeSlot"] == t)
            ]["AppointmentID"]

            load = sum(
                show_prob[p]
                for p in patients
            )

            loads.append(load)

        load_data.append((d, loads))

    fig, ax = plt.subplots(figsize=(12,5))

    for doctor, loads in load_data:

        ax.plot(
            time_slots,
            loads,
            marker="o",
            label=doctor
        )

    ax.axhline(
        y=1,
        linestyle="--",
        alpha=0.5,
        label="Capacity"
    )

    ax.set_title("Expected Load by Time Slot")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Expected Load")
    ax.legend()

    return fig

def create_idle_graph(idle, doctors, time_slots):

    idle_by_slot = []

    for t in time_slots:

        total_idle = sum(
            idle[d,t].X
            for d in doctors
        )

        idle_by_slot.append(total_idle)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        time_slots,
        idle_by_slot,
        marker="o"
    )

    ax.set_title("Total Expected Underutilisation by Time Slot")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Underutilisation")

    return fig

def create_overload_graph(
    overtime,
    doctors,
    time_slots
):

    overload_by_slot = []

    for t in time_slots:

        total_overload = sum(
            overtime[d,t].X
            for d in doctors
        )

        overload_by_slot.append(total_overload)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        time_slots,
        overload_by_slot,
        marker="o"
    )

    ax.set_title("Total Expected Overload by Time Slot")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Overload")

    return fig

def create_delay_distribution(schedule_df):

    fig, ax = plt.subplots(figsize=(10, 4))

    doctors = schedule_df["Doctor"].unique()

    for doctor in doctors:

        subset = schedule_df[
            schedule_df["Doctor"] == doctor
        ]

        ax.hist(
            subset["TimeSlot"],
            bins=24,
            alpha=0.5,
            label=doctor
        )

    ax.set_title("Patient Delay Distribution by Doctor")
    ax.set_xlabel("Assigned Time Slot")
    ax.set_ylabel("Number of Patients")
    ax.legend()

    return fig