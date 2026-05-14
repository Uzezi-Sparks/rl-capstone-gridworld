import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "reports/evidence/results_summary.csv"
)

plt.figure()

plt.bar(
    df["Run"],
    df["Score"]
)

plt.xlabel(
    "Experiment Run"
)

plt.ylabel(
    "Mean Return"
)

plt.title(
    "V3 Hyperparameter Search Results"
)

plt.savefig(
    "reports/figures/v3_results.png"
)

print(
    "Saved: reports/figures/v3_results.png"
)

plt.show()