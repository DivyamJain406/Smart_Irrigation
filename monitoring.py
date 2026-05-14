import pandas as pd


# ======================================
# LOAD PREDICTION LOGS
# ======================================

log_file = "logs/predictions.log"


try:

    with open(log_file, "r") as file:

        logs = file.readlines()

except FileNotFoundError:

    print("Prediction log file not found.")
    exit()


# ======================================
# EXTRACT STATES
# ======================================

soil_values = []
temperature_values = []
rainfall_values = []


for line in logs:

    try:

        state_part = line.split("State: ")[1]

        state_text = state_part.split(" | ")[0]

        state = eval(state_text)

        soil_values.append(state[0])
        temperature_values.append(state[1])
        rainfall_values.append(state[2])

    except:

        continue


# ======================================
# CALCULATE AVERAGES
# ======================================

avg_soil = sum(soil_values) / len(soil_values)
avg_temp = sum(temperature_values) / len(temperature_values)
avg_rain = sum(rainfall_values) / len(rainfall_values)


# ======================================
# PRINT MONITORING METRICS
# ======================================

print("\n===== MONITORING REPORT =====\n")

print(f"Average Soil Moisture : {avg_soil:.2f}")
print(f"Average Temperature   : {avg_temp:.2f}")
print(f"Average Rainfall      : {avg_rain:.2f}")


# ======================================
# SIMPLE DRIFT DETECTION
# ======================================

if avg_soil > 8:

    print("\nWARNING: Soil moisture drift detected!")

if avg_temp > 35:

    print("\nWARNING: Temperature drift detected!")

if avg_rain > 8:

    print("\nWARNING: Rainfall drift detected!")


print("\nMonitoring completed.")