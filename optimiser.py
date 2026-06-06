import pandas as pd

def solve_model(model):
    model.setParam("Presolve", 2)
    model.setParam("Symmetry", 2)
    model.setParam("MIPFocus", 3)
    model.setParam("Threads", 0)
    model.setParam("MIPGap", 0.005)
    model.optimize()
    return model.status == 2  # OPTIMAL

def extract_solution(model, x, u, doctors, time_slots, df, slot_labels):

    appointments = df.index.tolist()

    schedule = []
    unscheduled = []

    for i in appointments:

        # Pull AppointmentID directly from dataframe
        appointment_id = df.loc[i, "AppointmentID"]

        if u[i].X > 0.5:
            unscheduled.append(appointment_id)

        else:
            for d in doctors:
                for t in time_slots:
                    if x[i,d,t].X > 0.5:
                        schedule.append([
                            appointment_id, d, t, slot_labels[t]
                        ])

    schedule_df = pd.DataFrame(
        schedule,
        columns=["AppointmentID", "Doctor", "TimeSlot", "Time"]
    )

    return schedule_df, unscheduled

def build_doctor_schedule(schedule_df):

    # Sort by doctor and timeslot
    grouped = (
        schedule_df
        .sort_values(["Doctor", "TimeSlot"])
        .groupby("Doctor")["AppointmentID"]
        .apply(list)
        .reset_index()
    )

    # Convert list to string
    grouped["Appointments"] = grouped["AppointmentID"].apply(
        lambda x: ", ".join(x)
    )

    return grouped[["Doctor", "Appointments"]]