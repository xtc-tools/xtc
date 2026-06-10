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
import numpy as np
from types import SimpleNamespace as ns
from pathlib import Path
from typing import Sequence, Any
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def read_results_csv(
    fname: str | Path | None,
    Xcol: str | None = "X",
    Ycol: str | None = "Y",
    backend: str | None = None,
):
    assert fname is not None
    assert Xcol is not None
    assert Ycol is not None
    X, Y = [], []
    with open(fname, newline="") as infile:
        reader = csv.reader(infile, delimiter=",")
        backend_idx, X_idx, Y_idx = 0, 0, 0
        for idx, row in enumerate(reader):
            if idx == 0:
                X_idx = row.index(Xcol)
                Y_idx = row.index(Ycol)
                if backend is not None:
                    backend_idx = row.index("backend")
            else:
                if backend is None or row[backend_idx] == backend:
                    X.append(row[X_idx])
                    Y.append(eval(row[Y_idx], {}, {}))
    return np.array(X), np.array(Y)


def read_inputs(args: ns):
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


def draw_pmf(ax: Any, Y: Sequence[float], bins: int = 20, label: str | None = None):
    ax.hist(Y, bins=bins, label=label, histtype="step", alpha=0.8)


def draw_cdf(ax: Any, Y: Sequence[float], bins: int = 20, label: str | None = None):
    ax.hist(
        Y,
        bins=bins,
        density=True,
        cumulative=True,
        label=label,
        histtype="step",
        alpha=0.8,
    )


def draw_cor(
    ax: Any,
    Yref: Sequence[float],
    Y: Sequence[float],
    ref_label: str | None = None,
    label: str | None = None,
):
    ax.scatter(
        Yref,
        Y,
        label=label,
    )
    if ref_label:
        ax.set_xlabel(ref_label)


def save_fig(fname: str | Path):
    fig = plt.gcf()
    dpi = fig.dpi
    size = fig.get_size_inches() * dpi
    width = 1024
    height = size[1] / size[0] * width
    fig.set_size_inches(width / dpi, height / dpi)
    plt.savefig(fname, dpi=dpi)


def display_results(results: Sequence[ns], args: ns):
    num_figs = sum([bool(opt) for opt in [args.pmf, args.cdf, args.cor]])
    if num_figs == 0:
        return
    fig, axs = plt.subplots(num_figs, 1, figsize=(6, 3 * num_figs))
    if num_figs == 1:
        axs = [axs]
    idx = 0
    axes = ns()
    for type in ["pmf", "cdf", "cor"]:
        if not getattr(args, type):
            continue
        setattr(axes, type, axs[idx])
        idx += 1

    if args.pmf:
        for res in results:
            draw_pmf(axes.pmf, res.Y, label=res.label)
        axes.pmf.legend()
        axes.pmf.set_title("Peak performance distribution")

    if args.cdf:
        for res in results:
            draw_cdf(axes.cdf, res.Y, label=res.label)
        axes.cdf.legend()
        axes.cdf.set_title("Peak performance cumulative distribution")

    if args.cor:
        assert len(results) >= 2
        ref = results[0]
        for res in results[1:]:
            print("XXX", len(ref.Y), len(res.Y))
            draw_cor(axes.cor, ref.Y, res.Y, ref_label=ref.label, label=res.label)
        axes.cor.legend(loc="upper left")
        axes.cor.set_title("Peak performance correlation")

    if args.title:
        fig.suptitle(args.title)

    plt.tight_layout()
    if args.output:
        save_fig(args.output)
    if args.show:
        plt.show()


def display(args: ns):
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
        "--pmf", action=argparse.BooleanOptionalAction, default=True, help="draw PMF"
    )
    parser.add_argument(
        "--cdf", action=argparse.BooleanOptionalAction, default=True, help="draw CDF"
    )
    parser.add_argument(
        "--cor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw correlation",
    )
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

    display(ns(**vars(args)))


if __name__ == "__main__":
    main()
