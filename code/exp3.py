import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

por = pd.read_csv("data/student-por.csv", sep=";")
por["result"] = np.where(por.G3 >= 10, "Pass", "Fail")


def report(a, b, data=por):
    t = pd.crosstab(data[a], data[b])
    chi2, p, dof, e = stats.chi2_contingency(t)
    print(
        f"{a:>9} x {b:<8} chi2 {chi2:8.2f}  dof {dof}  p {p:.3g}  {'dependent' if p < .05 else 'independent'}"
    )
    return t, chi2, p, dof, e


print("Portuguese course data,", len(por), "students")

for a, b in [
    ("school", "result"),
    ("sex", "result"),
    ("internet", "result"),
    ("romantic", "result"),
    ("paid", "result"),
    ("address", "school"),
    ("famsup", "schoolsup"),
]:
    report(a, b)

t, chi2, p, dof, e = report("studytime", "result")

print("\nstudy time against result")
print(t.to_string())
print(
    "expected\n"
    + pd.DataFrame(
        e.round(1), index=t.index, columns=t.columns
    ).to_string()
)
print("cells with expected below 5:", int((e < 5).sum()))
print(
    "critical value at 1 %:",
    round(stats.chi2.ppf(0.99, dof), 3),
    "| at 5 %:",
    round(stats.chi2.ppf(0.95, dof), 3),
)

die = np.random.default_rng(8).integers(1, 7, 300)
obs = np.bincount(die)[1:]
g = stats.chisquare(obs)

print(
    "\n300 rolls of a die:",
    obs.tolist(),
    "-> chi2",
    round(g.statistic, 2),
    "p",
    round(g.pvalue, 3),
)

loaded = np.array([30, 35, 40, 45, 50, 100])
g2 = stats.chisquare(loaded)

print(
    "a loaded die",
    loaded.tolist(),
    "-> chi2",
    round(g2.statistic, 2),
    "p",
    f"{g2.pvalue:.2e}",
)

m = stats.chi2_contingency(pd.crosstab(por.school, por.result))

print(
    "\n2 x 2 school against result: with Yates correction chi2",
    round(m[0], 3),
    ", odds ratio",
    round(
        stats.fisher_exact(pd.crosstab(por.school, por.result))[0], 3
    ),
    ", Fisher p",
    f"{stats.fisher_exact(pd.crosstab(por.school, por.result))[1]:.2e}",
)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
(t.div(t.sum(1), axis=0) * 100).plot.bar(
    stacked=True, ax=ax[0], color=["black", "lightgrey"], rot=0
)
ax[0].set_title("Result by study time (%)")
ax[1].bar(range(1, 7), obs, color="grey")
ax[1].axhline(50, color="black", ls="--")
ax[1].set_title("300 die rolls, expected 50 each")
plt.tight_layout()
plt.show()
