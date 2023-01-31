import codecs
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_df(df: pd.DataFrame, title:str = None, file_path:str = "output.png"):
    plt.clf()
    sns.set_style("whitegrid")
    sns.barplot(x="model_size", y="mean_rank", data=df, color="darkblue").set_title(title)
    plt.xlabel("number of models")
    plt.ylabel("mean rank")
    plt.ylim(0,10.2)
    plt.savefig(fname=file_path)
    plt.show()


def read_data(filename: str) -> Tuple[pd.DataFrame, list, list]:
    model_sizes = []
    mean_ranks = []
    results = []
    with codecs.open(filename=filename, mode="r", encoding="utf-8") as f:
        results = json.load(f)

    for result in results:
        ranks = result['recommendation_ranks']
        mean_rank = np.mean(ranks)
        no_models = result['no_models']
        mean_ranks.append(mean_rank)
        model_sizes.append(no_models)

    return pd.DataFrame({"model_size": model_sizes, "mean_rank": mean_ranks}), model_sizes, mean_ranks


def plot_lines(model_sizes, mean_ranks):
    plt.clf()
    plt.plot(model_sizes, mean_ranks)
    plt.show()


def plot_bars(model_sizes, mean_ranks):
    plt.clf()
    plt.bar(model_sizes, mean_ranks)
    plt.show()


if __name__ == '__main__':
    filenames = [x for x in os.listdir("results") if os.path.isfile(os.path.join("results", x)) and "all" not in x]
    timestamp_to_process = filenames[-1].split("_")[0]
    filenames = [x for x in filenames if x.startswith(timestamp_to_process)]
    for filename in filenames:
        df, model_sizes, mean_ranks = read_data("results/" + filename)
        #plot_bars([str(x) for x in model_sizes], mean_ranks)
        #plot_lines(model_sizes, mean_ranks)
        visualize_df(df, title=filename[:-5], file_path=filename[:-5] + ".pdf")