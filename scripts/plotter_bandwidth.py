#!/usr/bin/env python

"""Basic plotting utility of bandwidth benchmark.


To generate the allreduce data, see `benchmark_latency_bandwidth.sh`


Requirements: `pip install matplotlib seaborn pandas`
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

plt.rcParams["lines.markersize"] = 20

sns.set(rc={"figure.figsize": (15.7, 8.27)})
sns.set_theme(style="ticks", palette="pastel")
sns.set_context("paper")
sns.set(font_scale=2)


DEVICE = os.environ.get("DEVICE", "cpu")
DTYPE = os.environ.get("HCCL_DTYPE", "float16")
PT_VER = torch.__version__

X_LABEL = "size_in_bytes"
Y_LABEL = "bw_mb_per_sec"


def main():
    df = pd.read_csv(f"osu_bandwidth-{DEVICE}-{DTYPE}-2.csv")
    df.rename(
        columns={"size_in_bytes": X_LABEL, "bw_mb_per_sec": Y_LABEL}, inplace=True
    )

    sns.despine(right=True)
    ax = sns.lineplot(
        x=X_LABEL,
        y=Y_LABEL,
        sizes=(20, 200),
        legend=False,
        data=df,
    )

    ax = sns.scatterplot(
        x=X_LABEL,
        y=Y_LABEL,
        sizes=(20, 200),
        legend="auto",
        data=df,
    )

    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.set_xlabel("Message length (bytes)")
    ax.set_ylabel("Bandwidth (MB/s)")
    title = f"OSU-MPI Bandwidth benchmark\n (Device: {DEVICE} | dtype: {DTYPE}"
    title += f" | PT: {PT_VER}"
    title += ")"
    ax.set_title(title)

    plt.savefig(f"plot-bandwidth-{DEVICE}-{DTYPE}.png")


if __name__ == "__main__":
    sys.exit(main())  # noqa
