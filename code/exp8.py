import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

mall = pd.read_csv("data/mall.csv")
mall.columns = ["id", "gender", "age", "income", "score"]
por = pd.read_csv("data/student-por.csv", sep=";")

print("Part A: predict the final grade G3 of", len(por), "students")

cols = [
    "G1",
    "G2",
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "freetime",
    "goout",
    "health",
]
X, y = por[cols], por.G3
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=3
)
for name, c in [
    ("G2 only", ["G2"]),
    ("G1 and G2", ["G1", "G2"]),
    ("all ten", cols),
    ("without G1, G2", cols[2:]),
]:
    m = LinearRegression().fit(Xtr[c], ytr)
    p = m.predict(Xte[c])
    print(
        f"  {name:<15} test R2 {r2_score(yte, p):.3f}   RMSE {np.sqrt(mean_squared_error(yte, p)):.2f}"
    )
m = LinearRegression().fit(Xtr[["G1", "G2"]], ytr)

print(
    f"  G3 = {m.intercept_:.2f} + {m.coef_[0]:.3f} G1 + {m.coef_[1]:.3f} G2"
)

full = LinearRegression().fit(Xtr, ytr)

print(
    "  weights of the ten columns:",
    {c: round(float(w), 2) for c, w in zip(cols, full.coef_)},
)
print(
    "  cross validated R2 (5 folds):",
    round(cross_val_score(LinearRegression(), X, y, cv=5).mean(), 3),
)

r = yte - full.predict(Xte)

print(
    "  residual mean",
    round(r.mean(), 3),
    "| largest miss",
    round(r.abs().max(), 2),
    "| share within 1 grade point",
    round((r.abs() <= 1).mean() * 100, 1),
    "%",
)

print(
    "\nPart B: spending score of",
    len(mall),
    "mall customers from the age",
)

a = mall[["age"]]
s = mall.score
Atr, Ate, str_, ste = train_test_split(
    a, s, test_size=0.25, random_state=1
)
for deg in (1, 2, 3, 5, 9):
    pm = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(
        Atr, str_
    )
    print(
        f"  degree {deg}: train R2 {pm.score(Atr, str_):.3f}   test R2 {pm.score(Ate, ste):.3f}"
    )
rg = make_pipeline(PolynomialFeatures(9), Ridge(alpha=1.0)).fit(
    Atr, str_
)

print(
    "  degree 9 with ridge penalty: test R2",
    round(rg.score(Ate, ste), 3),
)
print(
    "  correlation of age and score:",
    round(a.age.corr(s), 3),
    "| with income:",
    round(mall.income.corr(s), 3),
)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
pg = (
    LinearRegression()
    .fit(Xtr[["G1", "G2"]], ytr)
    .predict(Xte[["G1", "G2"]])
)
ax[0].scatter(yte, pg, s=12, color="grey")
ax[0].plot([0, 20], [0, 20], "k--")
ax[0].set_title("Actual against predicted G3")
ax[1].scatter(pg, yte - pg, s=12, color="grey")
ax[1].axhline(0, color="black")
ax[1].set_title("Residuals")
xs = np.linspace(18, 70, 200).reshape(-1, 1)
ax[2].scatter(a, s, s=10, color="lightgrey")
for deg, ls in ((1, "-"), (3, "--"), (9, ":")):
    ax[2].plot(
        xs,
        make_pipeline(PolynomialFeatures(deg), LinearRegression())
        .fit(Atr, str_)
        .predict(xs),
        "k",
        ls=ls,
        label=f"degree {deg}",
    )
ax[2].set_ylim(0, 105)
ax[2].legend()
ax[2].set_title("Score against age")
plt.tight_layout()
plt.show()
