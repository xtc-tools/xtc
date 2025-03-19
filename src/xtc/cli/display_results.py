#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""
Display histogram of results for a list of .csv files
"""

import argparse
import logging
import csv
import random
import numpy as np
from types import SimpleNamespace as ns
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def read_results_csv(fname, Xcol="X", Ycol="Y", backend=None):
    X, Y = [], []
    with open(fname, newline="") as infile:
        reader = csv.reader(infile, delimiter=";")
        for idx, row in enumerate(reader):
            if idx == 0:
                X_idx = row.index(Xcol)
                Y_idx = row.index(Ycol)
                if backend is not None:
                    backend_idx = row.index("backend")
            else:
                if backend is None or row[backend_idx] == backend:
                    # eval(row[X_idx], {}, {})
                    X.append(row[X_idx])
                    Y.append(eval(row[Y_idx], {}, {}))
    return np.array(X), np.array(Y)


def read_inputs(args):
    results = []
    for idx, inp in enumerate(args.inputs):
        spec_map = {
            0: None,
            1: f"res_{idx}",
            2: "X",
            3: "Y",
            4: None,
        }
        spec_map.update({k: v for k, v in enumerate(inp.split(":"))})
        fname, label, Xcol, Ycol, backend = list(spec_map.values())
        X, Y = read_results_csv(fname, Xcol=Xcol, Ycol=Ycol, backend=backend)
        results.append(ns(X=X, Y=Y, label=label))
    return results


def draw_pmf(ax, Y, bins=20, label=None):
    ax.hist(Y, bins=bins, label=label, histtype="step", alpha=0.8)


def draw_cdf(ax, Y, bins=20, label=None):
    plt.hist(
        Y,
        bins=bins,
        density=True,
        cumulative=True,
        label=label,
        histtype="step",
        alpha=0.8,
    )


def save_fig(fname):
    fig = plt.gcf()
    dpi = fig.dpi
    size = fig.get_size_inches() * dpi
    width = 1024
    height = size[1] / size[0] * width
    fig.set_size_inches(width / dpi, height / dpi)
    plt.savefig(fname, dpi=dpi)


def display_results(results, args):
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    for res in results:
        draw_pmf(axs[0], res.Y, label=res.label)
    axs[0].legend()
    axs[0].set_title("Peak performance distribution")

    for res in results:
        draw_cdf(axs[1], res.Y, label=res.label)
    axs[1].legend()
    axs[1].set_title("Peak performance cumulative distribution")
    if args.title:
        fig.suptitle(args.title)

    plt.tight_layout()
    if args.output:
        save_fig(args.output)
    if args.show:
        plt.show()


def display(args):
    results = read_inputs(args)
    display_results(results, args)


def main():
    parser = argparse.ArgumentParser(
        description="Autotune Matmult",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--title", type=str, help="Figure title")
    parser.add_argument("--output", type=str, help="Save figure to file")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show figure",
    )
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="debug mode"
    )
    parser.add_argument("inputs", nargs="+", help="input csv files")
    args = parser.parse_args()

    logging.basicConfig()
    logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    display(args)


if __name__ == "__main__":
    main()
