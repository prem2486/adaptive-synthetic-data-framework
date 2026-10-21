
import time
import pandas as pd

from engine.tabddpm import generate_tabddpm

# Small mixed-type dataset for a smoke test
df = pd.DataFrame({
    "age": [22, 25, 28, 31, 35, 39, 42, 46,
            24, 29, 33, 37, 41, 45, 27, 34,
            38, 43, 48, 30, 26, 32, 40, 44],
    "income": [25000, 28000, 31000, 35000, 42000, 46000,
               51000, 58000, 27000, 32000, 38000, 44000,
               50000, 56000, 30000, 40000, 45000, 53000,
               61000, 36000, 29000, 39000, 49000, 55000],
    "category": ["A", "B", "A", "C", "B", "A",
                 "C", "B", "A", "C", "B", "A",
                 "B", "C", "A", "B", "C", "A",
                 "B", "C", "A", "B", "C", "A"],
})

start = time.time()

synthetic_df = generate_tabddpm(
    reference_df=df,
    n_samples=5,
    seed=42,
    n_iter=5,
    batch_size=16,
    device="cpu",
)

print("\nGenerated synthetic data:")
print(synthetic_df)
print("\nShape:", synthetic_df.shape)
print("Columns match:", list(synthetic_df.columns) == list(df.columns))
print("Elapsed seconds:", round(time.time() - start, 2))

assert synthetic_df.shape == (5, 3)
assert list(synthetic_df.columns) == list(df.columns)

print("\nSMOKE TEST PASSED")
