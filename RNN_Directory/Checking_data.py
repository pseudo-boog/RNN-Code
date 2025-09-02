import pickle
import numpy as np
import os

# === Path to your training x2 file ===
path = "/Users/dylanpyle/VsCode/RNN_Code/DATA/TESTING/x2_train_M1_M20_train_val_test_set.pkl"

# === Load file ===
print(f"Loading {path} ...")
with open(path, "rb") as f:
    x2_train = pickle.load(f)

# === Inspect type ===
print("\nType:", type(x2_train))

if isinstance(x2_train, dict):
    print("\nTop-level keys:", list(x2_train.keys())[:10])
    for k, v in x2_train.items():
        if isinstance(v, dict):
            print(f"\nKey {k} contains inner dict with keys:", list(v.keys()))
            for ik, iv in v.items():
                if isinstance(iv, np.ndarray):
                    print(f"    Inner key {ik}: shape {iv.shape}, dtype {iv.dtype}")
                else:
                    print(f"    Inner key {ik}: type {type(iv)}")
        elif isinstance(v, np.ndarray):
            print(f"Key {k}: shape {v.shape}, dtype {v.dtype}")
        else:
            print(f"Key {k}: type {type(v)}")

elif isinstance(x2_train, np.ndarray):
    print("Shape:", x2_train.shape)
    print("Dtype:", x2_train.dtype)
    # Show first sample's first few timesteps
    if x2_train.ndim == 3:
        print("\nFirst sample (first 5 timesteps):")
        print(x2_train[0, :5, :])
    elif x2_train.ndim == 2:
        print("\nFirst sample (first 10 features):")
        print(x2_train[0, :10])
else:
    print("Unknown type:", type(x2_train))

print("\nInspection complete.")
