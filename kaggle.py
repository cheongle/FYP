import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("joniarroba/noshowappointments")
print("Path to dataset files:", path)

# Check dataset
print("Files in dataset folder:")
print(os.listdir(path))

# Load into pandas
file_path = os.path.join(path, "KaggleV2-May-2016.csv")
df = pd.read_csv(file_path)

print(df.head())
print(df.shape)

# Clean data
# Standardise column names 
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace('-', '_')
)

# Convert datetime columns
df['scheduledday'] = pd.to_datetime(df['scheduledday'], errors='coerce')
df['appointmentday'] = pd.to_datetime(df['appointmentday'], errors='coerce')

# Convert target
df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})

# Encode gender
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# Disregard unnecessary columns
df = df.drop(columns=['patientid', 'appointmentid', 'neighbourhood'])

# Handle missing values
print(df.isnull().sum())
df = df.dropna()

# Fixing unrealistic ages
df = df[(df['age'] >= 0) & (df['age'] <= 100)]

# Remove impossible cases
df['lead_time_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
df = df[df['lead_time_days'] >= 0]

# Clean binary columns
binary_cols = [
    'scholarship', 'hipertension',
    'diabetes', 'alcoholism',
    'sms_received'
]

for col in binary_cols:
    df[col] = df[col].astype(int)

# Fix handcap column
df = df[(df['handcap'] >= 0) & (df['handcap'] <= 4)]

# Remove duplicates
df = df.drop_duplicates()

# Sanity check
print(df.info())
print(df.describe())
print("Dataset loaded with shape:", df.shape)
print("Columns:", df.columns.tolist())

# Save cleaned dataset
df.to_csv("clean_noshow.csv", index=False)