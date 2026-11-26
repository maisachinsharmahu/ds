# Experiment 1: Describing, Viewing and Manipulating Data

> **File:** `exp1.py`  |  **Tools:** Python 3, pandas, NumPy, Matplotlib

## 🎯 Aim

To describe, view and manipulate data with pandas.

## 📘 Theory

Before any model is built, the data must be looked at. A pandas DataFrame holds a table whose columns are Series. The shape, the first and last rows (head, tail), the column types and the count of non-null values (info) and the numeric summary (describe) give a first picture of what the table contains and what is missing.

Manipulation then prepares the table: gaps are filled (here with the mean of the ward, or with the median), new columns are derived, rows are chosen by condition, by query() or by position, and values are summarised with a pivot table. The sample is a table of 15 hospital patients.

- **describe()**: Count, mean, spread and quartiles of each numeric column.
- **transform()**: Applies a group function and keeps the shape of the table.
- **query()**: Selects rows with a text condition.
- **pivot_table()**: Summary of a value for each group.

## 🪜 Algorithm / Steps

1. Generate a table of 15 patients with two missing ages and one missing bill.
2. View its shape, types, head, tail, the missing values and describe().
3. Fill the ages with the mean of the ward and the bill with the median; derive the cost per day and a senior flag.
4. Select rows by flag, by nlargest and by query, and count the wards.
5. Make a pivot table of the mean bill and stay for each ward and check duplicates.

## 💻 Code

```python
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
```

## 🖥️ Output

```text
shape (15, 5) | dtypes: {'patient': 'str', 'ward': 'str', 'age': 'float64', 'stay_days': 'float64', 'bill': 'float64'}
patient   ward  age  stay_days    bill
   P100 Cardio 28.0        6.0 42000.0
   P101  Neuro 62.0       10.0 39800.0
   P102  Ortho 77.0       11.0 55400.0
   P103  Ortho  NaN        4.0 43900.0
patient  ward  age  stay_days    bill
   P113 Ortho 78.0       10.0 47900.0
   P114 Neuro 71.0       11.0 31900.0

missing: {'age': 2, 'bill': 1}
        age  stay_days     bill
count  13.0       15.0     14.0
mean   60.9        7.5  43321.4
std    17.9        3.9  11889.1
min    28.0        1.0  24300.0
25%    60.0        4.5  33825.0
50%    65.0        7.0  42950.0
75%    73.0       10.5  49725.0
max    79.0       14.0  65800.0

after cleaning, missing: 0
patient   ward  age  per_day  senior
   P100 Cardio 28.0   7000.0   False
   P101  Neuro 62.0   3980.0    True
   P102  Ortho 77.0   5036.0    True
   P103  Ortho 66.0  10975.0    True
   P104  Ortho 61.0  16450.0    True

seniors: 12 | average bill of seniors 43742 | others 41517
longest stays: [{'patient': 'P107', 'ward': 'Cardio', 'stay_days': 14.0}, {'patient': 'P110', 'ward': 'Ortho', 'stay_days': 13.0}, {'patient': 'P102', 'ward': 'Ortho', 'stay_days': 11.0}]
query ward == 'Cardio' and bill > 40000: ['P100', 'P107']
ward counts: {'Ortho': 7, 'Cardio': 4, 'Neuro': 4}
           bill  stay_days
ward                      
Cardio  40025.0        6.8
Neuro   37975.0        7.8
Ortho   48207.1        7.7
column types after astype: int64
duplicated rows: 0 | unique wards: 3
```

## ✅ Result

The table has 15 rows and 5 columns, two ages and one bill were missing (the mean age of the 13 known patients is 60.9 and the mean bill 43321.4). After filling them no value is missing. 12 patients are seniors (age 60 or more) with an average bill of 43742 against 41517 for the others. The longest stays are P107 (14 days), P110 and P102. Cardio patients with a bill above 40000 are P100 and P107. Ortho is the largest ward (7 patients) and also the most expensive (mean bill 48207.1).

## 🎤 Viva Questions

**Q. What is the difference between a Series and a DataFrame?**  
A. A Series is a one-dimensional labelled array; a DataFrame is a two-dimensional table of labelled rows and columns whose columns are Series.

**Q. What is the difference between loc and iloc?**  
A. loc selects by labels, iloc selects by integer positions.

**Q. What does describe() give?**  
A. The count, mean, standard deviation, minimum, quartiles and maximum of the numeric columns.

**Q. How are missing values found and handled?**  
A. isna() finds them; dropna() removes them and fillna() fills them, for example with the median or the mean.

**Q. How is a new column derived?**  
A. By assigning an expression of other columns, such as df['amount'] = df['qty'] * df['price'].

---
[⬅ Back to index](README.md)
