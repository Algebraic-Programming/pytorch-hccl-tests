"""Basic plotting utility of benchmark results. Allreduce is tested for now.


To generate the allreduce data.

#!/bin/bash
DTYPES="int float16 float32"
for SIZE in {2..8}
do
    for DTYPE in ${DTYPES}
    do
        echo "Allreduce (size: ${SIZE} | dtype: ${DTYPE})"
        make allreduce -e WORLD_SIZE=${SIZE} HCCL_DTYPE=${DTYPE}
    done
done
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams["lines.markersize"] = 20

sns.set(rc={"figure.figsize": (15.7, 8.27)})
sns.set_theme(style="ticks", palette="pastel")
sns.set_context("paper")
sns.set(font_scale=2)


DEVICE = "cpu"
BENCHMARK = "allreduce"
WORLD_SIZES = [2, 3, 4, 5, 6, 7, 8]
col_name = "World Size"


def main():
    df = pd.DataFrame()
    for world_size in WORLD_SIZES:
        s = str(world_size)
        local = pd.read_csv(f"osu_{BENCHMARK}-{DEVICE}-{s}.csv")
        local[col_name] = s
        df = pd.concat([df, local], axis=0)
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
    ax.set_ylabel("Average Latency")
    ax.set_title(f"OS-MPI average {BENCHMARK}")

    plt.savefig(f"plot_{BENCHMARK}.png")


if __name__ == "__main__":
    sys.exit(main())  # noqa
