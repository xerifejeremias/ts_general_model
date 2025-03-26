import pandas as pd
from feature_engineering import apply_pipeline_feature_engineering

# Read the CSV file into a DataFrame
df = pd.read_csv("data.csv")

# Iterate through groups based on 'market_code' and 'product_code'
for (market_code, product_code), group in df.groupby(["market_code", "product_code"]):
    # Generate new features for the current group
    new_features = apply_pipeline_feature_engineering(group)
    # Add the new features back to the original DataFrame using the group index
    df.loc[group.index, new_features.columns] = new_features

# Print the first few rows of the DataFrame to verify the result
print(df.head())
