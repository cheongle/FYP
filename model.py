import gurobipy as gp
from gurobipy import GRB

def build_model(df, doctors, time_slots, params):

    appointments = df.index.tolist()
    p = df['no_show_prob'].to_dict()
    show_prob = {i: 1 - p[i] for i in appointments}

    model = gp.Model("Scheduling")

    x = model.addVars(appointments, doctors, time_slots,
                      vtype=GRB.BINARY, name="x")

    u = model.addVars(appointments, vtype=GRB.BINARY, name="u")

    idle = model.addVars(doctors, time_slots, lb=0)
    overtime = model.addVars(doctors, time_slots, lb=0)

    # Constraints
    for i in appointments:
        model.addConstr(
            gp.quicksum(x[i,d,t] for d in doctors for t in time_slots) + u[i] == 1
        )

    for d in doctors:
        for t in time_slots:
            model.addConstr(
                gp.quicksum(x[i,d,t] for i in appointments) <= params["overbooking"]
            )

    for d in doctors:
        for t in time_slots[:-1]:
            model.addConstr(
                gp.quicksum(x[i,d,t] for i in appointments)
                <= gp.quicksum(x[i,d,t+1] for i in appointments) + 1
            )

    expected_load = {}
    for d in doctors:
        for t in time_slots:
            expected_load[d,t] = gp.quicksum(
                show_prob[i] * x[i,d,t] for i in appointments
            )

    for d in doctors:
        for t in time_slots:
            model.addConstr(idle[d,t] >= 1 - expected_load[d,t])
            model.addConstr(overtime[d,t] >= expected_load[d,t] - 1)
            model.addConstr(expected_load[d,t] <= params["load_cap"])

    waiting = gp.quicksum(
        (t / len(time_slots)) * show_prob[i] * x[i,d,t]
        for i in appointments for d in doctors for t in time_slots
    )

    risk = gp.quicksum(
        expected_load[d,t] ** 2
        for d in doctors for t in time_slots
    )

    model.setObjective(
        params["w_wait"] * waiting +
        params["w_idle"] * gp.quicksum(idle[d,t] for d in doctors for t in time_slots) +
        params["w_overtime"] * gp.quicksum(overtime[d,t] for d in doctors for t in time_slots) +
        params["w_unscheduled"] * gp.quicksum(u[i] for i in appointments) +
        params["w_risk"] * risk,
        GRB.MINIMIZE
    )

    return model, x, u, idle, overtime