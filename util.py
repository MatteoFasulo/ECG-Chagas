#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
import scipy
from scipy.stats import norm, bernoulli, uniform, multivariate_normal, binom
from scipy.special import expit, logit
import keras
from keras import layers
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


def build_nn_model(input_shape, output_shape, hidden, output_activation='linear'):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    model_out = layers.Dense(output_shape, activation=output_activation)(x)
    model = keras.Model(model_in, model_out)
    return model


# def train_nn_model(model, X, y, loss,
#         verbose=0, patience=10,
#         validation_split=0.0, **fit_params):
#     # Compile the model
#     model.compile(optimizer='Adam', loss=loss)
#     # Build the early stop callback
#     cb = []
#     if validation_split > 0:
#         cb += [callbacks.EarlyStopping(patience=patience,
#             restore_best_weights=True)]
#     # Train the model
#     history = model.fit(X, y, callbacks=cb,
#             validation_split=validation_split,
#             verbose=verbose, **fit_params)
#     return history


# def plot_nn_model(model):
#     return keras.utils.plot_model(model, show_shapes=True,
#             show_layer_names=True, rankdir='LR', show_layer_activations=True)


# def plot_training_history(history=None,
#         figsize=None, print_final_scores=True):
#     plt.figure(figsize=figsize)
#     for metric in history.history.keys():
#         plt.plot(history.history[metric], label=metric)
#     # if 'val_loss' in history.history.keys():
#     #     plt.plot(history.history['val_loss'], label='val. loss')
#     if len(history.history.keys()) > 0:
#         plt.legend()
#     plt.xlabel('epochs')
#     plt.grid(linestyle=':')
#     plt.tight_layout()
#     plt.show()
#     if print_final_scores:
#         trl = history.history["loss"][-1]
#         s = f'Final loss: {trl:.4f} (training)'
#         if 'val_loss' in history.history:
#             vll = history.history["val_loss"][-1]
#             s += f', {vll:.4f} (validation)'
#         print(s)



# def load_ed_data(data_file):
#     # Read the CSV file
#     data = pd.read_csv(data_file, sep=';', parse_dates=[2, 3])
#     # Remove the "Flow" column
#     f_cols = [c for c in data.columns if c != 'Flow']
#     data = data[f_cols]
#     # Convert a few fields to categorical format
#     data['Code'] = data['Code'].astype('category')
#     data['Outcome'] = data['Outcome'].astype('category')
#     # Sort by triage time
#     data.sort_values(by='Triage', inplace=True)
#     # Discard the firl
#     return data


def plot_bars(data, figsize=None, tick_gap=1, series=None, title=None,
              xlabel=None, ylabel=None, std=None):
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
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle=':')
    plt.tight_layout()



# def build_nn_poisson_model(input_shape, hidden, rate_guess=1):
#     model_in = keras.Input(shape=input_shape, dtype='float32')
#     x = model_in
#     for h in hidden:
#         x = layers.Dense(h, activation='relu')(x)
#     log_rate = layers.Dense(1, activation='linear')(x)
#     lf = lambda t: tfp.distributions.Poisson(rate=rate_guess * tf.math.exp(t))
#     model_out = tfp.layers.DistributionLambda(lf)(log_rate)
#     model = keras.Model(model_in, model_out)
#     return model


# def build_nn_normal_model(input_shape, hidden, stddev_guess=1):
#     model_in = keras.Input(shape=input_shape, dtype='float32')
#     x = model_in
#     for h in hidden:
#         x = layers.Dense(h, activation='relu')(x)
#     mu_logsigma = layers.Dense(2, activation='linear')(x)
#     lf = lambda t: tfp.distributions.Normal(loc=t[:, :1], scale=tf.math.exp(t[:, 1:]))
#     model_out = tfp.layers.DistributionLambda(lf)(mu_logsigma)
#     model = keras.Model(model_in, model_out)
#     return model


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


# def load_cmapss_data(data_folder):
#     # Read the CSV files
#     fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
#     cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
#     datalist = []
#     nmcn = 0
#     for fstem in fnames:
#         # Read data
#         dfname = os.path.join(f'{data_folder}', f'{fstem}.txt')
#         data = pd.read_csv(dfname, sep=' ', header=None)
#         # Drop the last two columns (parsing errors)
#         data.drop(columns=[26, 27], inplace=True)
#         # Replace column names
#         data.columns = cols
#         # Add the data source
#         data['src'] = fstem
#         # Shift the machine numbers
#         data['machine'] += nmcn
#         nmcn += len(data['machine'].unique())
#         # Generate RUL data
#         cnts = data.groupby('machine')[['cycle']].count()
#         cnts.columns = ['ftime']
#         data = data.join(cnts, on='machine')
#         data['rul'] = data['ftime'] - data['cycle']
#         data.drop(columns=['ftime'], inplace=True)
#         # Store in the list
#         datalist.append(data)
#     # Concatenate
#     data = pd.concat(datalist)
#     # Put the 'src' field at the beginning and keep 'rul' at the end
#     data = data[['src'] + cols + ['rul']]
#     # data.columns = cols
#     return data


# def split_by_field(data, field):
#     res = {}
#     for fval, gdata in data.groupby(field):
#         res[fval] = gdata
#     return res


# def partition_by_machine(data, tr_machines):
#     # Separate
#     tr_machines = set(tr_machines)
#     tr_list, ts_list = [], []
#     for mcn, gdata in data.groupby('machine'):
#         if mcn in tr_machines:
#             tr_list.append(gdata)
#         else:
#             ts_list.append(gdata)
#     # Collate again
#     tr_data = pd.concat(tr_list)
#     ts_data = pd.concat(ts_list)
#     return tr_data, ts_data


# def random_censoring(data, rel_censoring_lb, seed=None):
#     # seed the RNG
#     np.random.seed(seed)
#     # Process all experiments
#     tr_list = []
#     for _, gdata in data.groupby('machine'):
#         # sample a censoring point
#         sep = int(len(gdata) * np.random.rand())
#         tr_list.append(gdata.iloc[:sep])
#     # Collate again
#     res = pd.concat(tr_list)
#     return res


# class RULCostModel:
#     def __init__(self, maintenance_cost, safe_interval=0):
#         self.maintenance_cost = maintenance_cost
#         self.safe_interval = safe_interval

#     def cost(self, machine, pred, thr, return_margin=False):
#         # Merge machine and prediction data
#         tmp = np.array([machine, pred]).T
#         tmp = pd.DataFrame(data=tmp,
#                            columns=['machine', 'pred'])
#         # Cost computation
#         cost = 0
#         nfails = 0
#         slack = 0
#         for mcn, gtmp in tmp.groupby('machine'):
#             idx = np.nonzero(gtmp['pred'].values < thr)[0]
#             if len(idx) == 0:
#                 cost += self.maintenance_cost
#                 nfails += 1
#             else:
#                 cost -= max(0, idx[0] - self.safe_interval)
#                 slack += len(gtmp) - idx[0]
#         if not return_margin:
#             return cost
#         else:
#             return cost, nfails, slack


# def plot_rul(pred=None, target=None,
#         stddev=None,
#         q1_3=None,
#         same_scale=True,
#         figsize=None,
#         title=None,
#         xlabel=None,
#         ylabel=None):
#     plt.figure(figsize=figsize)
#     if target is not None:
#         plt.plot(range(len(target)), target, label='target',
#                 color='tab:orange')
#     if pred is not None:
#         if same_scale or target is None:
#             ax = plt.gca()
#         else:
#             ax = plt.gca().twinx()
#         ax.plot(range(len(pred)), pred, label='pred',
#                 color='tab:blue')
#         if stddev is not None:
#             ax.fill_between(range(len(pred)),
#                     pred-stddev, pred+stddev,
#                     alpha=0.3, color='tab:blue', label='+/- std')
#         if q1_3 is not None:
#             ax.fill_between(range(len(pred)),
#                     q1_3[0], q1_3[1],
#                     alpha=0.3, color='tab:blue', label='1st/3rd quartile')
#     plt.legend()
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(linestyle=':')
#     plt.tight_layout()


# def predict_cf(model, ref_sample, columns, values):
#     # Replicate the reference sample
#     df = np.repeat(ref_sample.values.reshape(1, -1), len(values), axis=0)
#     # Replace the columns
#     df = pd.DataFrame(columns=ref_sample.index, data=df)
#     df[columns] = values
#     # Compute predictions
#     pred = model.predict(df.astype('float64'), verbose=0)
#     # Add index and return
#     return pd.Series(data=pred.ravel())


# def rolling_survival_cmapss(hazard_model, data, look_up_window):
#     # Repeat the rows in the dataframe
#     rdata = data.loc[data.index.repeat(len(look_up_window))]
#     # Add cycle offsets
#     rdata['cycle'] += np.tile(look_up_window, len(data))
#     # Compute hazard values
#     hazard = hazard_model.predict(rdata, verbose=0)
#     # Reshape hazard values (one row per original sample)
#     hazard = hazard.reshape(len(data), len(look_up_window))
#     # Compute survival
#     survival = np.cumprod(1-hazard, axis=1)
#     # Prepare a dataframe and return
#     return pd.DataFrame(index=data.index, data=survival, columns=look_up_window)

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

