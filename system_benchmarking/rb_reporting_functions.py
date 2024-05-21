# Copyright 2024 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Functions for plotting and reporting 2Q RB data. '''

import pandas as pd
import numpy as np

from scipy.stats import sem
import matplotlib.pyplot as plt

from rb_analysis_functions import exponential_with_asymptote, convert_metrics

zone_labels_3 = {
    '0, 1': 'DG01',
    '2, 3': 'DG02',
    '4, 5': 'DG03',
    '6, 7': 'DG04',
    '0': 'DG01-left',
    '1': 'DG01-right',
    '2': 'DG02-left',
    '3': 'DG02-right',
    '4': 'DG03-left',
    '5': 'DG03-right',
    '6': 'DG04-left',
    '7': 'DG04-right',
}

def errorbar_plot(fid_info: dict,
                  data: dict,
                  log_scale=False,
                  savename=None):
    ''' Plot survival probability and fitting. '''

    if len(fid_info) > 10:
        cmap = plt.cm.turbo
        color_list = [cmap(i) for i in range(0, cmap.N, cmap.N//len(fid_info))]
    else:
        color_list = [plt.get_cmap("tab10").colors[i] for i in range(10)]

    xvals = [int(l) for l in data['sequence_info']]
    xrange = np.arange(np.min(xvals), np.max(xvals)+1)

    fig, ax = plt.subplots()
    legend = []
    for i, (q, metric_params) in enumerate(fid_info.items()):
        nqubits = len(q.split(', '))
        asympt = 1/2**nqubits

        A, r = convert_metrics(metric_params, nqubits)
        survival_fit = exponential_with_asymptote(
            xrange,
            A,
            r,
            asympt
        )
        ax.plot(xrange, survival_fit, "-", color=color_list[i])

        for length in xvals:
            surv = [
                val/data['shots']
                for val in data['survival'][q][str(length)].values()
            ]
            ax.errorbar(
                length,
                np.mean(surv),
                yerr=sem(surv),
                fmt="o",
                markersize=5,
                capsize=3,
                ecolor=color_list[i],
                markerfacecolor=[1, 1, 1],
                markeredgecolor=color_list[i],
            )
        legend.append(str(q))

    ax.grid(True, axis="both", linestyle="--")
    ax.set_xlabel("Sequence length (number of Cliffords)")
    ax.set_ylabel("Avg. Survival")

    legend = [zone_labels_3[key] for key in legend]


    ax.legend(legend)
    if log_scale:
        ax.set_xscale("log")

    if savename:
        fig.savefig(savename + '.svg', format='svg')


def report(fid_info: dict, 
           boot_info: dict):
    ''' Returns DataFrame containing summary of results. '''

    column0 = 'RB Intercept'
    column1 = 'Avg. infidelity'

    try:
        fid_info = {zone_labels_3[key]: fid for key, fid in fid_info.items()}
        boot_info = {zone_labels_3[key]: fid for key, fid in boot_info.items()}
    except KeyError:
        pass

    df1 = pd.DataFrame.from_dict(fid_info).transpose()
    df1.rename(columns={0: column0, 1: column1}, inplace=True)

    boot_new = {
        qname: {
            f'{column0} uncertainty': (qvals['intercept upper'] - qvals['intercept lower'])/2,
            f'{column1} uncertainty': (qvals['rate upper'] - qvals['rate lower'])/2
        }
        for qname, qvals in boot_info.items()
    }
    df2 = pd.DataFrame.from_dict(boot_new).transpose()
    
    result = pd.concat([df1, df2], axis=1).reindex(df1.index)
    result.rename(columns={result.columns[0]: 'Qubits'})
    result = result[[column1, f'{column1} uncertainty', column0, f'{column0} uncertainty']]
    result.loc['Mean'] = result.mean()

    # change uncertainties to standard error in means
    result[f'{column0} uncertainty']['Mean'] = avg_uncertainty(
        result[f'{column0} uncertainty'].head(len(result[f'{column0} uncertainty']) - 1).to_list()
    )
    result[f'{column1} uncertainty']['Mean'] = avg_uncertainty(
        result[f'{column1} uncertainty'].head(len(result[f'{column1} uncertainty']) - 1).to_list()
    )
    result[column1] = result[column1].map(lambda x: 1 - x)
    pd.set_option('display.float_format', lambda x: '%.3E' % x)
    result[column0] = result[column0].map(lambda x: 1 - x)

    return result


def avg_uncertainty(uncertainty_list):

    out = np.sqrt(sum(
        unc**2 
        for unc in uncertainty_list
    )) / len(uncertainty_list)

    return out