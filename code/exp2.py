import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(5)
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
x = np.linspace(0, 8, 400)
for lam in (0.5, 1.0, 2.0):
    ax[0, 0].plot(
        x, stats.expon.pdf(x, scale=1 / lam), label=f"rate {lam}"
    )
ax[0, 0].set_title("Exponential")
ax[0, 0].legend()
k = np.arange(0, 20)
for m in (2, 5, 9):
    ax[0, 1].plot(k, stats.poisson.pmf(k, m), "o-", label=f"mean {m}")
ax[0, 1].set_title("Poisson")
ax[0, 1].legend()
u = rng.uniform(0, 1, (4000, 12)).sum(axis=1)
ax[1, 0].hist(u, bins=40, density=True, color="grey", edgecolor="white")
xs = np.linspace(u.min(), u.max(), 200)
ax[1, 0].plot(xs, stats.norm.pdf(xs, 6, 1), "k")
ax[1, 0].set_title("Sum of 12 uniform numbers")
d = rng.normal(100, 15, 3000)
ax[1, 1].hist(
    d, bins=40, density=True, color="lightgrey", edgecolor="grey"
)
xx = np.linspace(40, 160, 300)
ax[1, 1].plot(xx, stats.norm.pdf(xx, 100, 15), "k")
ax[1, 1].fill_between(
    xx[(xx > 85) & (xx < 115)],
    stats.norm.pdf(xx[(xx > 85) & (xx < 115)], 100, 15),
    color="grey",
    alpha=0.6,
)
ax[1, 1].set_title("IQ scores, mean 100 and sd 15")
plt.tight_layout()
plt.show()

print(
    "exponential with rate 0.5: mean",
    stats.expon.mean(scale=2),
    "| P(X > 3) =",
    round(stats.expon.sf(3, scale=2), 4),
    "| median",
    round(stats.expon.median(scale=2), 3),
)
print(
    "poisson with mean 5: P(X = 5) =",
    round(stats.poisson.pmf(5, 5), 4),
    "| P(X <= 3) =",
    round(stats.poisson.cdf(3, 5), 4),
    "| P(X >= 8) =",
    round(stats.poisson.sf(7, 5), 4),
)
print(
    "uniform(0, 1) sums of 12: mean",
    round(u.mean(), 3),
    "variance",
    round(u.var(), 3),
    "(theory 6 and 1), skewness",
    round(stats.skew(u), 3),
)
print(
    "normality check of the sums: Shapiro p =",
    round(stats.shapiro(u[:500]).pvalue, 3),
)
print(
    "IQ between 85 and 115:",
    round(
        stats.norm.cdf(115, 100, 15) - stats.norm.cdf(85, 100, 15), 4
    ),
)
print(
    "IQ above 130:",
    round(stats.norm.sf(130, 100, 15), 4),
    "| 99th percentile:",
    round(stats.norm.ppf(0.99, 100, 15), 1),
)
print(
    "simulated share above 130:",
    round((d > 130).mean(), 4),
    "| sample mean",
    round(d.mean(), 2),
    "sd",
    round(d.std(ddof=1), 2),
)
print(
    "kurtosis of normal sample:",
    round(stats.kurtosis(d), 3),
    "| skewness:",
    round(stats.skew(d), 3),
)

z = (115 - 100) / 15

print(
    "z-score of 115:",
    z,
    "| area to its left:",
    round(stats.norm.cdf(z), 4),
)
print(
    "sf of exponential at its mean:",
    round(stats.expon.sf(2, scale=2), 4),
    "(= e^-1 =",
    round(np.exp(-1), 4),
    ")",
)
