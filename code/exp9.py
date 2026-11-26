import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mall = pd.read_csv("data/mall.csv")
mall.columns = ["id", "gender", "age", "income", "score"]
mall["band"] = pd.cut(
    mall.age,
    [17, 29, 44, 60, 80],
    labels=["18-29", "30-44", "45-60", "61+"],
)

print("customers:", len(mall))

print("\n1. by gender")
print(
    mall.groupby("gender")
    .agg(
        n=("id", "count"),
        age=("age", "mean"),
        income=("income", "mean"),
        score=("score", "mean"),
    )
    .round(2)
    .to_string()
)

print("\n2. by age band, several functions")
print(
    mall.groupby("band", observed=True)
    .score.agg(["count", "mean", "median", "std", "min", "max"])
    .round(1)
    .to_string()
)

print("\n3. pivot of mean score: age band x gender")
print(
    mall.pivot_table(
        index="band",
        columns="gender",
        values="score",
        aggfunc="mean",
        observed=True,
    )
    .round(1)
    .to_string()
)

print("\n4. share of high spenders (score > 60) per band")
print(
    (
        mall.assign(high=mall.score > 60)
        .groupby("band", observed=True)
        .high.mean()
        * 100
    )
    .round(1)
    .to_string()
)

mall["income_rank"] = (
    mall.groupby("band", observed=True).income.rank(pct=True).round(2)
)
mall["vs_band"] = (
    mall.score
    - mall.groupby("band", observed=True).score.transform("mean")
).round(1)

print("\n5. rank and deviation inside the group")
print(
    mall[["id", "band", "income", "income_rank", "score", "vs_band"]]
    .head(4)
    .to_string(index=False)
)

print("\n6. income quartiles and mean score")

mall["iq"] = pd.qcut(mall.income, 4, labels=["Q1", "Q2", "Q3", "Q4"])

print(
    mall.groupby("iq", observed=True)
    .agg(
        low=("income", "min"),
        high=("income", "max"),
        score=("score", "mean"),
    )
    .round(1)
    .to_string()
)

print("\n7. bands with a mean score above 45 (filter)")

keep = mall.groupby("band", observed=True).filter(
    lambda g: g.score.mean() > 45
)

print(keep.band.value_counts().sort_index().to_dict())

print("\n8. cumulative income of the five best spenders per gender")

top = (
    mall.sort_values("score", ascending=False)
    .groupby("gender")
    .head(5)
    .sort_values(["gender", "score"], ascending=[True, False])
)
top["running"] = top.groupby("gender").income.cumsum()

print(
    top[["gender", "score", "income", "running"]].to_string(index=False)
)

print(
    "\n9. resample like grouping with rolling mean of score sorted by income:",
    mall.sort_values("income")
    .score.rolling(20)
    .mean()
    .dropna()
    .iloc[[0, 50, 100, 170]]
    .round(1)
    .tolist(),
)
print("10. crosstab with margins")
print(pd.crosstab(mall.band, mall.iq, margins=True).to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
mall.groupby("band", observed=True).score.mean().plot.bar(
    ax=ax[0], color="grey", rot=0
)
ax[0].set_title("Mean score by age band")
mall.pivot_table(
    index="iq",
    columns="gender",
    values="score",
    aggfunc="mean",
    observed=True,
).plot.bar(ax=ax[1], color=["lightgrey", "black"], rot=0)
ax[1].set_title("Mean score by income quartile")
pd.crosstab(mall.band, mall.iq).plot.bar(
    stacked=True, ax=ax[2], colormap="Greys", rot=0
)
ax[2].set_title("Customers per band and quartile")
plt.tight_layout()
plt.show()
