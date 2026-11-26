import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

por = pd.read_csv("data/student-por.csv", sep=";")
g = por[["G1", "G2", "G3", "absences", "age"]]

print("students:", len(por))
print(g.describe().round(2).to_string())

print("\nextra measures")

ex = pd.DataFrame(
    {
        "median": g.median(),
        "mode": g.mode().iloc[0],
        "variance": g.var(),
        "IQR": g.quantile(0.75) - g.quantile(0.25),
        "skew": g.skew(),
        "kurt": g.kurt(),
        "cv %": g.std() / g.mean() * 100,
    }
)

print(ex.round(2).to_string())

q1, q3 = g.absences.quantile([0.25, 0.75])

print(
    "\nabsences: Q1",
    q1,
    "Q3",
    q3,
    "upper fence",
    q3 + 1.5 * (q3 - q1),
    "| outliers",
    int((g.absences > q3 + 1.5 * (q3 - q1)).sum()),
)
print(
    "grades of zero in G3:",
    int((por.G3 == 0).sum()),
    "| the final grade is at most",
    int(por.G3.max()),
)

print("\nPearson correlation")
print(g.corr().round(3).to_string())
print("Spearman G1-G3:", round(stats.spearmanr(g.G1, g.G3)[0], 3))

low, hi = stats.t.interval(
    0.95, len(g) - 1, g.G3.mean(), stats.sem(g.G3)
)

print(
    f"\n95 % confidence interval of the mean G3: {low:.2f} to {hi:.2f}"
)
print(
    "mean G3 by sex:",
    por.groupby("sex").G3.mean().round(2).to_dict(),
    "| t-test p =",
    f'{stats.ttest_ind(por[por.sex == "F"].G3, por[por.sex == "M"].G3).pvalue:.4f}',
)
print(
    "G1 against G3 paired t-test p =",
    f"{stats.ttest_rel(g.G1, g.G3).pvalue:.2e}",
    "| mean change",
    round((g.G3 - g.G1).mean(), 2),
)
print(
    "one-way ANOVA of G3 across study time: F =",
    round(
        stats.f_oneway(
            *[x.G3 for _, x in por.groupby("studytime")]
        ).statistic,
        2,
    ),
    ", p =",
    f"{stats.f_oneway(*[x.G3 for _, x in por.groupby('studytime')]).pvalue:.2e}",
)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].hist(g.G3, bins=range(0, 22), color="grey", edgecolor="white")
ax[0, 0].set_title("Final grade")
por.boxplot(column="G3", by="studytime", ax=ax[0, 1])
ax[0, 1].set_title("G3 by study time")
plt.sca(ax[0, 1])
plt.suptitle("")
ax[1, 0].scatter(g.G1, g.G3, s=10, color="grey", alpha=0.6)
ax[1, 0].plot([0, 20], [0, 20], "k--")
ax[1, 0].set_title("G1 against G3")
im = ax[1, 1].imshow(g.corr(), cmap="Greys")
ax[1, 1].set_xticks(range(5))
ax[1, 1].set_xticklabels(g.columns, fontsize=7)
ax[1, 1].set_yticks(range(5))
ax[1, 1].set_yticklabels(g.columns, fontsize=7)
plt.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_title("Correlation")
plt.tight_layout()
plt.show()
