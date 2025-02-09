#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
import scipy
from scipy.stats import norm, bernoulli, uniform, multivariate_normal, binom
from scipy.special import expit, logit
from sklearn import metrics


def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
                   figsize=None, s=4, xlabel=None, ylabel=None):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_bars(data, figsize=None, tick_gap=1, series=None, title=None,
              xlabel=None, ylabel=None, std=None, rotation: int = 45):
    plt.figure(figsize=figsize)
    # x = np.arange(len(data))
    # x = 0.5 + np.arange(len(data))
    # plt.bar(x, data, width=0.7)
    # x = data.index-0.5
    x = data.index
    plt.bar(x, data, width=0.7, yerr=std)
    # plt.bar(x, data, width=0.7)
    if series is not None:
        # plt.plot(series.index-0.5, series, color='tab:orange')
        plt.plot(series.index, series, color='tab:orange')
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=rotation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle=':')
    plt.tight_layout()


def plot_pred_scatter(y_true, y_pred, figsize=None, print_metrics=True):
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=0.1)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.grid(linestyle=':')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()

    if print_metrics:
        print(f'R2: {metrics.r2_score(y_true, y_pred):.2f}')
        print(f'MAE: {metrics.mean_absolute_error(y_true, y_pred):.2f}')


def generate_data(size=100, seed=None):
    # Seed the numpy RNG
    np.random.seed(seed=seed)
    # =======================================================================
    # Causal model
    # =======================================================================

    # X_i = observable input
    # Z_i = latent variable
    # Y = output

    # X_0, X_1, Z_0 = -> X_2 (mediator)
    # X_2, X_3 -> Y
    # Z_1 -> X_0 (confounder)
    # Z_1 -> X_4 (confounder)
    # X_5 (same distribution as Y)

    # Generate Z_1 (confounder)
    z1_dist = norm(loc=0, scale=1)
    z1 = z1_dist.rvs(size)

    # Generate X_0 (mediated cause, normal)
    x0_dist = norm(loc=0, scale=1)
    x0 = x0_dist.rvs(size) + z1

    # Generate X_1 (mediated cause, log-normal)
    x1_dist = norm(loc=0, scale=1)
    x1 = np.exp(x1_dist.rvs(size))

    # Generate Z_0 (hidden cause, normal)
    z0_dist = norm(loc=0, scale=1.5)
    z0 = z0_dist.rvs(size)

    # Generate X_2 (mediator)
    x2 = 0.5 * (x0 + x1) + z0

    # Generate X_3 (direct cause)
    x3_dist = bernoulli(p=0.6)
    x3 = x3_dist.rvs(size)

    # Output
    y_p_logit = (-1 + 2 * x3) * x2 + 0.4 * x0 - 0.4 * x1
    y_p_logit = y_p_logit - 0.3 * y_p_logit.mean()
    y_dist = bernoulli(p=expit(y_p_logit))
    y = y_dist.rvs(size)

    # Generate X_4 (confounder-linked variable)
    x4_dist = norm(loc=0, scale=1)
    x4 = x4_dist.rvs(size) + z1

    # Generate multiple, sparsely correlated, 0-1 variables
    n_sd = 5
    x_sd_vals = []
    x_sd_names = []
    pwgt = norm(loc=0, scale=1).rvs(size=(n_sd, n_sd)) # random linear dependency weights
    pmsk = bernoulli(p=0.5).rvs(size=(n_sd, n_sd)) # random mask
    pmsk = np.tril(pmsk) # make the mask triangular, so that dependncies are acyclic
    pwgt = (pwgt * pmsk) # normalize
    Z = pwgt.sum(axis=1).reshape(-1, 1) # normalization constants
    Z[Z == 0] = 1 # avoid division by zero in the next normalization
    pwgt = pwgt / Z
    for j in range(n_sd): # generate values for all variables
        tmp_logit = uniform(loc=-1.5, scale=1.5).rvs(size)
        tmp_logit += np.sum(x_sd_vals[k] * pwgt[j, k] for k in range(j))
        tmp_vals = bernoulli(p=expit(tmp_logit)).rvs(size)
        x_sd_vals.append(tmp_vals)
        x_sd_names.append(f'X{5+j}')

    # Generate multiple, sparsely correlated, normal variables
    n_nd = 5
    x_nd_vals = []
    x_nd_names = [f'X{5+n_sd+j}' for j in range(n_nd)]
    sqs_dense = norm(loc=0, scale=1).rvs(size=(n_nd, n_nd)) # dense pre-signa matrix
    sqs_mask = bernoulli(p=0.2).rvs(size=(n_sd, n_sd)) # sparse mask
    sqs = sqs_dense * sqs_mask # apply mask
    sqs[range(n_nd), range(n_nd)] = 1 # ensure non-zero diagonal
    sigma = sqs.T @ sqs # positive definite matrix
    # sigma /= sigma[range(n_nd), range(n_nd)] # rescale to obtain a covariance matrix
    means = norm(loc=0, scale=1).rvs(n_nd)
    x_nd_vals = multivariate_normal(mean=means, cov=sigma).rvs(size)

    # Concatenate all inputs data
    in_vals = np.vstack([x0, x1, x2, x3, x4] + x_sd_vals).T
    in_vals = np.hstack((in_vals, x_nd_vals))
    in_names = ['X0', 'X1', 'X2', 'X3', 'X4'] + x_sd_names + x_nd_names
    # Shuffle columns
    cidx = list(range(in_vals.shape[1]))
    np.random.shuffle(cidx) # TODO re-enable after debugging
    in_vals = in_vals[:, cidx]

    # Hide the input names and build a name map
    in_names = [in_names[i] for i in cidx]
    cnames = [f'u{i}' for i in range(in_vals.shape[1])] # TODO re-enable after debugging
    # cnames = list(in_names)
    name_map = {u:n for u, n in zip(cnames, in_names)}

    # Prepare the result dataframe
    cnames.append('y')
    data = np.hstack([in_vals, y.reshape(-1, 1)])
    res = pd.DataFrame(columns=cnames, data=data)
    return res, name_map


def binomial_plot(p, n, l_alpha=None, r_alpha=None,
                  l_color='tab:green', r_color='tab:green',
                  figsize=None, **kw_args):
    # Define the input range
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    cdf = binom.cdf(x, n, p)

    # Identify low- and high- section
    if l_alpha is not None:
        l_sep = int(np.argwhere(cdf <= l_alpha)[-1])
        l_x = x[:l_sep+1]
        l_pmf = pmf[:l_sep+1]
    if r_alpha is not None:
        r_sep = int(np.argwhere(cdf >= 1-r_alpha)[0])
        r_x = x[r_sep:]
        r_pmf = pmf[r_sep:]

    # Build a figure
    plt.figure(figsize=figsize)
    plt.plot(x, pmf)
    if l_alpha:
        plt.plot(l_x, l_pmf, color=l_color, lw=5)
    if r_alpha:
        plt.plot(r_x, r_pmf, color=r_color, lw=5)
    plt.xlabel(f'number of observed events')
    plt.ylabel(f'probability')
    plt.grid(':')

