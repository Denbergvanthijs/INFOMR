import pandas as pd

df = pd.read_csv('csvs/feature_extraction.csv')
df = df.fillna(0)

# Save the updated DataFrame back to a CSV file
df.to_csv('csvs/feature_extraction.csv', index=False)