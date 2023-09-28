#!/usr/bin/env python
"""Basic plotting utility of collectives benchmark results.


To generate the allreduce data, see `benchmark_collectives.sh`


Requirements: `pip install matplotlib seaborn pandas`
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import torch

plt.rcParams["lines.markersize"] = 20

sns.set(rc={"figure.figsize": (15.7, 8.27)})
sns.set_theme(style="ticks", palette="pastel")
sns.set_context("paper")
sns.set(font_scale=2)


DEVICE = os.environ.get("DEVICE", "cpu")
BENCHMARK = os.environ.get("BENCH", "allreduce")
DTYPE = os.environ.get("HCCL_DTYPE", "float16")
PT_VER = torch.__version__

# WORLD_SIZES must be in sync with `benchmark_all.sh` range
WORLD_SIZES = [2, 3, 4, 5, 6, 7, 8]
col_name = "World Size"

Y_LABEL = "Elapsed Time (us)"


def main():
    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency", col_name])
    df.rename(columns={"avg_latency": Y_LABEL}, inplace=True)

    for world_size in WORLD_SIZES:
        s = str(world_size)
        try:
            local = pd.read_csv(f"osu_{BENCHMARK}-{DEVICE}-{DTYPE}-{s}.csv")
            local[col_name] = s
            df = pd.concat([df, local], axis=0)
        except FileNotFoundError as err:
            print(f"Error: {err}")

    df[col_name] = pd.Categorical(df[col_name])

    sns.despine(right=True)
    ax = sns.lineplot(
        x="size_in_bytes",
        y="avg_latency",
        hue=col_name,
        sizes=(20, 200),
        legend=False,
        data=df,
    )

    ax = sns.scatterplot(
        x="size_in_bytes",
        y="avg_latency",
        hue=col_name,
        sizes=(20, 200),
        legend="auto",
        data=df,
    )

    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.legend(markerscale=2)
    ax.set_xlabel("Message length (bytes)")
    ax.set_ylabel("Average Latency (us)")
    title = f"OS-MPI {BENCHMARK} benchmark\n (Device: {DEVICE} | dtype: {DTYPE}"
    title += f" | PT: {PT_VER}"
    title += ")"
    ax.set_title(title)

    plt.savefig(f"plot-{BENCHMARK}-{DEVICE}-{DTYPE}.png")


if __name__ == "__main__":
    sys.exit(main())  # noqa
