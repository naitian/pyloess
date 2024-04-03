"""Fast, vectorized n-dimensional LOESS.

This LOESS implementation is fast and supports local polynomial regressions.

Implementation based on
https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm
"""
import numpy as np

def loess(x, y, eval_x=None, degree=2, span=0.75):
    """Fit LOESS model.

    Parameters
    ----------
    x : ndarray
        The x-values of the observed points.
    y : ndarray
        The y-values of the observed points.
    eval_x : ndarray, optional
        The x-values at which to evaluate the fitted regression.
    degree : int
        The degree of polynomial to be used.
    span : float
        Between 0 and 1. The size of the neighborhood as a fraction of the total points.

    Returns
    -------
    ndarray
        A two-dimensional numpy array if eval_x is not provided, otherwise a
        one-dimensional numpy array. In the case of a two-dimensional array,
        the first column is the sorted x values provided as input, and the
        second column is the predicted y-values from evaluating the LOESS
        regression at the corresponding x value. The one-dimensional array
        contains the predicted y-values corresponding to the values in eval_x.
    """
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    return_x = eval_x is None
    eval_x = x if eval_x is None else eval_x
    num_eval = len(eval_x)

    dists = np.abs(eval_x[:, None] - x)
    relevant_inds = np.argsort(dists, axis=1)[:, : int(np.ceil(span * len(x)))]
    dists = dists[np.arange(num_eval)[:, None], relevant_inds]

    # Use tricubic weighting
    normed_dists = dists / np.max(dists, axis=1, keepdims=True)
    weights = np.clip((1 - normed_dists**3) ** 3, 0, 1)
    indexer = np.arange(num_eval)[:, None]
    subset_x = np.tile(x, (num_eval, 1))[indexer, relevant_inds]
    subset_y = np.tile(y, (num_eval, 1))[indexer, relevant_inds]

    # Prepare x values for polynomial regression
    subset_x = np.stack([subset_x**i for i in range(degree + 1)], axis=-1)
    eval_x = np.stack([eval_x**i for i in range(degree + 1)], axis=-1)

    # Solve the regression
    betas = (
        np.linalg.inv(subset_x.transpose(0, 2, 1) * weights[:, None, :] @ subset_x)
        @ subset_x.transpose(0, 2, 1)
        * weights[:, None, :]
        @ subset_y[:, :, np.newaxis]
    ).squeeze()

    yhat = np.sum(betas * eval_x, axis=-1)

    if return_x:
        return np.stack((eval_x[:, 1], yhat), axis=1)
    return yhat

