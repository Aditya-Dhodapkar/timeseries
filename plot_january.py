import numpy as np
import matplotlib.pyplot as plt
import os
from src.folderconstants import output_folder

dataset = "temp_ghi"  # Change this if your dataset name is different
folder = os.path.join(output_folder, dataset)

# Load hourly data
year1_hourly = np.load(os.path.join(folder, 'year1_hourly.npy'))
year2_hourly = np.load(os.path.join(folder, 'year2_hourly.npy'))
year3_predictions = np.load(os.path.join(folder, 'year3_predictions_hourly.npy'))

# Extract January 1-2 (first 48 hours)
jan_1_2_year1 = year1_hourly[:48]
jan_1_2_year2 = year2_hourly[:48]
jan_1_2_year3 = year3_predictions[:48]

# Create x-axis (hours 0-47)
hours = np.arange(48)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(hours, jan_1_2_year1, label="Year 1 Actual", alpha=0.7)
plt.plot(hours, jan_1_2_year2, label="Year 2 Actual", alpha=0.7)
plt.plot(hours, jan_1_2_year3, label="Year 3 Predicted", alpha=0.7)
plt.xlabel("Hour (January 1-2)")
plt.ylabel("Load")
plt.title("January 1-2: Hourly Load Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plots/january_1_2_comparison.png")
plt.close()

print("January 1-2 comparison plot saved to plots/january_1_2_comparison.png")