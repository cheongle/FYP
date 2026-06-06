import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("clean_noshow.csv", parse_dates=['scheduledday', 'appointmentday'])

# Define days and cap extreme lead times
df['lead_time_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
df['lead_time_days'] = df['lead_time_days'].clip(lower=0, upper=60)
df['lead_time_days'] = df['lead_time_days'] / 30

# Weekday patterns
df['appointment_weekday'] = df['appointmentday'].dt.weekday

# Weekend indicator
df['is_weekend'] = df['appointment_weekday'].isin([5, 6]).astype(int)

# Cyclical encoding (for ML smoothness)
df['weekday_sin'] = np.sin(2 * np.pi * df['appointment_weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['appointment_weekday'] / 7)
df = df.drop(columns=['appointment_weekday'])

# Define health complexity
df['health_complexity'] = (
    df['hipertension'] +
    df['diabetes'] +
    df['alcoholism'] +
    df['handcap']
)

# Define demographics
def age_group(age):
    if age < 18:
        return 0
    elif age < 35:
        return 1
    elif age < 60:
        return 2
    else:
        return 3

df['age_group'] = df['age'].apply(age_group)

# Define commitment and socioeconomic proxies
df['engagement_score'] = df['sms_received'] + df['scholarship']

# Save engineered dataset
df.to_csv("featured_noshow.csv", index=False)
print("Feature engineering complete. Saved as featured_noshow.csv")