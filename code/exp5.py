import json, os, sqlite3, pickle
import pandas as pd
import matplotlib.pyplot as plt

mall = pd.read_csv("data/mall.csv")
mall.columns = ["id", "gender", "age", "income", "score"]
os.makedirs("out/store", exist_ok=True)
mall.to_csv("out/store/mall.csv", index=False)
mall.to_json("out/store/mall.json", orient="table")
mall.to_pickle("out/store/mall.pkl")
mall.to_csv("out/store/mall.tsv", index=False, sep="\t")
con = sqlite3.connect("out/store/mall.db")
mall.to_sql("customer", con, if_exists="replace", index=False)
for f in sorted(os.listdir("out/store")):
    print(f"{f:<12} {os.path.getsize('out/store/' + f) / 1024:7.1f} KB")

print("\nreading back")

rt = {
    "csv": pd.read_csv("out/store/mall.csv"),
    "json": pd.read_json("out/store/mall.json", orient="table"),
    "pickle": pd.read_pickle("out/store/mall.pkl"),
    "tsv": pd.read_csv("out/store/mall.tsv", sep="\t"),
    "sql": pd.read_sql("select * from customer", con),
}
for k, v in rt.items():
    print(
        f"  {k:<7} shape {v.shape}  same income column: {bool((v.income.values == mall.income.values).all())}"
    )

print("\nSQL")
print(
    pd.read_sql(
        "select gender, count(*) n, round(avg(age),1) age, round(avg(income),1) income, round(avg(score),1) score from customer group by gender",
        con,
    ).to_string(index=False)
)
print(
    pd.read_sql(
        "select case when age < 30 then 'under 30' when age < 50 then '30-49' else '50+' end band, count(*) n, round(avg(score),1) score from customer group by band order by band",
        con,
    ).to_string(index=False)
)
print(
    pd.read_sql(
        "select id, age, income, score from customer where score > 90 order by income desc limit 3",
        con,
    ).to_string(index=False)
)

con.execute("create index idx_income on customer(income)")

print(
    "rows with income between 60 and 80:",
    con.execute(
        "select count(*) from customer where income between 60 and 80"
    ).fetchone()[0],
)

con.close()

with open("out/store/summary.json", "w") as f:
    json.dump(
        {
            "customers": len(mall),
            "mean_income": round(mall.income.mean(), 2),
            "top_score": int(mall.score.max()),
        },
        f,
    )

print("\nsummary file:", open("out/store/summary.json").read())
print(
    "income and score correlation:",
    round(mall.income.corr(mall.score), 3),
    "| age and score:",
    round(mall.age.corr(mall.score), 3),
)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
mall.age.plot.hist(bins=12, ax=ax[0], color="grey", edgecolor="white")
ax[0].set_title("Age")
ax[1].scatter(
    mall.income,
    mall.score,
    c=mall.age,
    cmap="Greys",
    edgecolor="black",
    s=22,
)
ax[1].set_xlabel("income")
ax[1].set_ylabel("score")
ax[1].set_title("Income and score (shade = age)")
mall.groupby("gender")[["income", "score"]].mean().plot.bar(
    ax=ax[2], color=["lightgrey", "black"], rot=0
)
ax[2].set_title("Mean income and score by gender")
plt.tight_layout()
plt.show()
