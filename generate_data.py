import pandas as pd
import numpy as np

# Generate random dataset
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "User_ID": range(1, n + 1),
    "Daily_Usage_Minutes": np.random.randint(30, 600, n),
    "App_Open_Count": np.random.randint(5, 150, n),
    "Data_Used_MB": np.random.randint(50, 5000, n),
    "Calls_Made": np.random.randint(0, 50, n),
    "Messages_Sent": np.random.randint(0, 200, n),
    "Device_Type": np.random.choice(["Android", "iOS"], n)
})

df.to_csv("mobile_activity.csv", index=False)
print("✅ mobile_activity.csv generated!")
