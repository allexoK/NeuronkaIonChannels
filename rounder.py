import pandas as pd

df = pd.read_csv("submission")
df["time"] = df["time"].round(digits = 4)
print(df)