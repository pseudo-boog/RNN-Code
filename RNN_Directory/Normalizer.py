"""This file is for normalizing the data for the RNN model."""
import os
import pandas as pd

def normalize_and_save_to_training(start_iteration, end_iteration, mechanism_prefixes):
    for i in range(start_iteration, end_iteration + 1):
        for j in range(2):
            for prefix in mechanism_prefixes:
                test_path = f'/Users/dylanpyle/VsCode/RNN_Code/DATA/Training/{prefix}/rct_{i}/exp_{j}.csv'
                train_path = f'/Users/dylanpyle/VsCode/RNN_Code/DATA/test/{prefix}/rct_{i}/exp_{j}.csv'

                if not os.path.exists(test_path):
                    print(f"File not found: {test_path}. Skipping.")
                    continue

                try:
                    df = pd.read_csv(test_path, skiprows=1, header=None, usecols=[0, 2])
                    df.columns = ['A', 'P']
                    df = df.dropna()

                    max_A = df['A'].max()
                    max_P = df['P'].max()

                    if max_A == 0 or pd.isna(max_A) or max_P == 0 or pd.isna(max_P):
                        print(f"Warning: Max A or P is zero or NaN in {test_path}. Skipping.")
                        continue

                    df['A'] = df['A'] / max_A
                    df['P'] = df['P'] / max_A

                    # Ensure correct order: A, then P
                    df = df[['A', 'P']]

                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(train_path), exist_ok=True)

                    # Save normalized file
                    df.to_csv(train_path, index=False, header=False)
                    print(f"Saved normalized file: {train_path}")

                except Exception as e:
                    print(f"Error processing {test_path}: {e}. Skipping.")
                    continue

# Example usage
mechanism_prefixes = ["M1_si"]
start_iteration = 0
end_iteration = 10500
normalize_and_save_to_training(start_iteration, end_iteration, mechanism_prefixes)
