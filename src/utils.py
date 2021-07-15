#! /usr/bin/env python

"""
Convenience utilities for the package.
"""

import typing as tp
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from src.types import Array, DataFrame, GenericType


@tp.overload
def z_score(x: Array, axis: tp.Union[tp.Literal[0], tp.Literal[1]]) -> Array:
    ...


@tp.overload
def z_score(x: DataFrame, axis: tp.Union[tp.Literal[0], tp.Literal[1]]) -> DataFrame:
    ...


def z_score(
    x: tp.Union[Array, DataFrame], axis: tp.Union[tp.Literal[0], tp.Literal[1]] = 0
) -> tp.Union[Array, DataFrame]:
    """
    Standardize and center an array or dataframe.

    Parameters
    ----------
    x :
        A numpy array or pandas DataFrame.

    axis :
        Axis across which to compute - 0 == rows, 1 == columns.
        This effectively calculates a column-wise (0) or row-wise (1) Z-score.
    """
    return (x - x.mean(axis=axis)) / x.std(axis=axis)


def minmax_scale(x: GenericType) -> GenericType:
    """
    Scale array to 0-1 range.

    x: np.ndarray
        Array to scale
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - x.min()) / (x.max() - x.min())


def close_plots(func) -> tp.Callable:
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs) -> None:
        func(*args, **kwargs)
        plt.close("all")

    return close
