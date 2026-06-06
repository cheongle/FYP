import pandas as pd

def load_data(path="no_show_prob.csv"):
    df = pd.read_csv(path)
    
    # Create a clean, explicit ID
    df = df.reset_index(drop=True)
    df["AppointmentID"] = ["A" + str(i+1) for i in range(len(df))]

    return df


def get_subset(df, n):
    subset = df.head(n).copy()
    return subset