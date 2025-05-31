import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

algorithms = ["kNN", "PSO+kNN"]
k_values = [1, 3, 5, 10, 15, 20]

df_list = []

for alg in algorithms:
    for k in k_values:
        csv_path = f"results_{alg}_k={k}.csv"
        temp_df = pd.read_csv(csv_path)
        df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df["k"] = df["k"].astype(int)
df["accuracy"] = df["accuracy"].astype(float)
df["IDR"] = df["IDR"].astype(float)


# 1. Plot binary (all_attacks) charts
binary_df = df[df["attack_type"] == "all_attacks"]
acc_pivot = binary_df.pivot(index="k", columns="algorithm", values="accuracy")
idr_pivot = binary_df.pivot(index="k", columns="algorithm", values="IDR")

# 1.1 Accuracy
plt.figure()
for alg in acc_pivot.columns:
    plt.plot(
        acc_pivot.index,
        acc_pivot[alg],
        marker="o",
        label=alg
    )
plt.xticks(acc_pivot.index)
plt.xlabel("k (number of neighbours)")
plt.ylabel("Accuracy (attack vs. normal)")
plt.title("Binary (all_attacks) Accuracy vs. k")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# 1.2 IDR
plt.figure()
for alg in idr_pivot.columns:
    plt.plot(
        idr_pivot.index,
        idr_pivot[alg],
        marker="o",
        label=alg
    )
plt.xticks(idr_pivot.index)
plt.xlabel("k (number of neighbours)")
plt.ylabel("IDR (attack vs. normal)")
plt.title("Binary (all_attacks) IDR vs. k")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# 2. Plot per-attack charts
attack_types = sorted(df["attack_type"].unique())
if "all_attacks" in attack_types:
    attack_types.remove("all_attacks")

n_attacks = len(attack_types)
n_columns = 2
n_rows = int(np.ceil(n_attacks / n_columns))

# 2.1 Accuracy
fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 4 * n_rows), squeeze=False)
for idx, attack in enumerate(attack_types):
    row = idx // n_columns
    column = idx % n_columns
    ax = axes[row][column]

    sub_df = df[df["attack_type"] == attack]
    pivot_acc = sub_df.pivot(index="k", columns="algorithm", values="accuracy")

    for alg in pivot_acc.columns:
        ax.plot(
            pivot_acc.index,
            pivot_acc[alg],
            marker="o",
            label=alg
        )

    ax.set_title(f"Accuracy vs. k\n(attack = {attack})")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    if idx == 0:
        ax.legend()

for idx in range(n_attacks, n_rows * n_columns):
    fig.delaxes(axes[idx // n_columns][idx % n_columns])

plt.tight_layout()
plt.show()

# 2.2 IDR
fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 4 * n_rows), squeeze=False)
for idx, attack in enumerate(attack_types):
    row = idx // n_columns
    column = idx % n_columns
    ax = axes[row][column]

    sub_df = df[df["attack_type"] == attack]
    pivot_idr = sub_df.pivot(index="k", columns="algorithm", values="IDR")

    for alg in pivot_idr.columns:
        ax.plot(
            pivot_idr.index,
            pivot_idr[alg],
            marker="o",
            label=alg
        )

    ax.set_title(f"IDR vs. k\n(attack = {attack})")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.set_xlabel("k")
    ax.set_ylabel("IDR")
    ax.grid(True, linestyle="--", alpha=0.4)
    if idx == 0:
        ax.legend()

for idx in range(n_attacks, n_rows * n_columns):
    fig.delaxes(axes[idx // n_columns][idx % n_columns])

plt.tight_layout()
plt.show()

# 3. Bar charts
k_values = [5, 10, 15]
ks = sorted(df["k"].unique())
 
# 3.1 Accuracy
for current_k in ks:
    sub_df = df[df["k"] == current_k]
    pivot_acc = (
        sub_df[sub_df["attack_type"].isin(attack_types)]
        .pivot(index="attack_type", columns="algorithm", values="accuracy")
        .reindex(attack_types)
        .fillna(0)
        * 100.0
    )
    N = len(attack_types)
    ind = np.arange(N)
    width = 0.1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        ind,
        pivot_acc["kNN"],
        width,
        label="kNN"
    )
    ax.bar(
        ind + width,
        pivot_acc["PSO+kNN"],
        width,
        label="PSO+kNN"
    )
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per‐Attack Accuracy (k = {current_k})\n(kNN vs. PSO+kNN)")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(attack_types, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(99.5, 100)
    ax.set_yticks(np.arange(99.5, 100.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# 3.2 IDR
for current_k in ks:
    sub_df = df[df["k"] == current_k]
    pivot_idr = (
        sub_df[sub_df["attack_type"].isin(attack_types)]
        .pivot(index="attack_type", columns="algorithm", values="IDR")
        .reindex(attack_types)
        .fillna(0)
        * 100.0
    )
    N = len(attack_types)
    ind = np.arange(N)
    width = 0.1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        ind,
        pivot_idr["kNN"],
        width,
        label="kNN"
    )
    ax.bar(
        ind + width,
        pivot_idr["PSO+kNN"],
        width,
        label="PSO+kNN"
    )
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per‐Attack Accuracy (k = {current_k})\n(kNN vs. PSO+kNN)")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(attack_types, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(60, 100)
    ax.set_yticks(np.arange(60, 105, 5))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()