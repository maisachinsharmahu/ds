import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
n = 15
ward = pd.DataFrame(
    {
        "patient": [f"P{100 + i}" for i in range(n)],
        "ward": rng.choice(["Cardio", "Ortho", "Neuro"], n),
        "age": rng.integers(22, 80, n).astype(float),
        "stay_days": rng.integers(1, 15, n).astype(float),
        "bill": np.round(rng.normal(42000, 15000, n), -2),
    }
)
ward.loc[[3, 9], "age"] = np.nan
ward.loc[6, "bill"] = np.nan

print("shape", ward.shape, "| dtypes:", dict(ward.dtypes.astype(str)))
print(ward.head(4).to_string(index=False))
print(ward.tail(2).to_string(index=False))

print("\nmissing:", ward.isna().sum()[ward.isna().sum() > 0].to_dict())
print(ward.describe().round(1).to_string())

ward["age"] = (
    ward.groupby("ward")["age"]
    .transform(lambda s: s.fillna(s.mean()))
    .round(0)
)
ward["bill"] = ward["bill"].fillna(ward["bill"].median())
ward["per_day"] = (ward["bill"] / ward["stay_days"]).round(0)
ward["senior"] = ward["age"] >= 60

print("\nafter cleaning, missing:", int(ward.isna().sum().sum()))
print(ward.iloc[:5, [0, 1, 2, 5, 6]].to_string(index=False))

print(
    "\nseniors:",
    int(ward.senior.sum()),
    "| average bill of seniors",
    round(ward[ward.senior].bill.mean()),
    "| others",
    round(ward[~ward.senior].bill.mean()),
)
print(
    "longest stays:",
    ward.nlargest(3, "stay_days")[
        ["patient", "ward", "stay_days"]
    ].to_dict("records"),
)
print(
    "query ward == 'Cardio' and bill > 40000:",
    ward.query("ward == 'Cardio' and bill > 40000").patient.tolist(),
)
print("ward counts:", ward.ward.value_counts().to_dict())

wide = ward.pivot_table(
    index="ward", values=["bill", "stay_days"], aggfunc="mean"
).round(1)

print(wide.to_string())
print(
    "column types after astype:",
    ward.astype({"age": int}).dtypes["age"],
)
print(
    "duplicated rows:",
    int(ward.duplicated().sum()),
    "| unique wards:",
    ward.ward.nunique(),
)
