import pandas as pd
import numpy as np


df = pd.read_csv('csvs/feature_extraction.csv')

# Access the specific column you want to analyze
area = df[' surface_area']  # Replace 'YourColumnName' with the actual column name
volume = df[' volume']  # Replace 'YourColumnName' with the actual column name
compactness = df[' compactness']  # Replace 'YourColumnName' with the actual column name
diameter = df[' diameter']  # Replace 'YourColumnName' with the actual column name
convexity = df[' convexity']  # Replace 'YourColumnName' with the actual column name
eccentricity = df[' eccentricity']  # Replace 'YourColumnName' with the actual column name


print(f'Max area: {np.max(area)}   Min area: {np.min(area)}')
print(f'Max volume: {np.max(volume)}   Min volume: {np.min(volume)}')
print(f'Max compactness: {np.max(compactness)}   Min compactness: {np.min(compactness)}')
print(f'Max diameter: {np.max(diameter)}   Min diameter: {np.min(diameter)}')
print(f'Max convexity: {np.max(convexity)}   Min convexity: {np.min(convexity)}')
print(f'Max eccentricity: {np.max(eccentricity)}   Min eccentricity: {np.min(eccentricity)}')