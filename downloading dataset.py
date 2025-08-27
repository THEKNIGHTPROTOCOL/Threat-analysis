# Cell 2: Download the data and check it
import kagglehub
import os
import pandas as pd

# This will now automatically use your D: drive location!
print("Downloading dataset to D: drive... This may take a moment.")
dataset_path = kagglehub.dataset_download("marcodena/mobile-phone-activity")
print("Download complete!")
print("Dataset is located at:", dataset_path)

# List the files to confirm the download
print("\nFiles downloaded:")
for file_name in os.listdir(dataset_path):
    print(f" - {file_name}")

# (Optional) Read and display a bit of the data to confirm it works
data_file = os.path.join(dataset_path, "mobile_activity.csv") # Change this if the file name is different

if os.path.exists(data_file):
    print(f"\nReading the main data file...")
    df = pd.read_csv(data_file)
    print(df.head())
else:
    print("Main CSV file not found. Please check the printed list for the correct file name.")
