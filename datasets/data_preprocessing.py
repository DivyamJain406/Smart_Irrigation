import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("processed_data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

dataset_path = "dataset/smart_agriculture_dataset.csv"

df = pd.read_csv(dataset_path)


print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


# ==========================================
# SELECT IMPORTANT COLUMNS
# ==========================================

# MODIFY THESE COLUMN NAMES
# ACCORDING TO YOUR DATASET

selected_columns = [
    "temperature",
    "humidity",
    "rainfall"
]

df = df[selected_columns]


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df = df.dropna()


print("\nDataset Shape After Cleaning:")
print(df.shape)


# ==========================================
# RENAME COLUMNS
# ==========================================

df.rename(
    columns={

        "temperature": "temperature",
        "humidity": "soil_moisture",
        "rainfall": "rainfall"

    },
    inplace=True
)


# ==========================================
# SAVE CLEANED DATASET
# ==========================================

cleaned_dataset_path = (
    "processed_data/cleaned_irrigation_data.csv"
)

df.to_csv(
    cleaned_dataset_path,
    index=False
)

print("\nCleaned dataset saved.")


# ==========================================
# DATA VISUALIZATION
# ==========================================

# 1. Temperature Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["temperature"],
    kde=True
)

plt.title("Temperature Distribution")

plt.savefig(
    "visualizations/temperature_distribution.png"
)

plt.close()


# ==========================================

# 2. Soil Moisture Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["soil_moisture"],
    kde=True
)

plt.title("Soil Moisture Distribution")

plt.savefig(
    "visualizations/soil_moisture_distribution.png"
)

plt.close()


# ==========================================

# 3. Rainfall Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["rainfall"],
    kde=True
)

plt.title("Rainfall Distribution")

plt.savefig(
    "visualizations/rainfall_distribution.png"
)

plt.close()


# ==========================================

# 4. Correlation Heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.savefig(
    "visualizations/correlation_heatmap.png"
)

plt.close()


print("\nVisualizations generated successfully.")