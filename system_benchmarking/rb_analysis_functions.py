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

''' Functions for loading 2Q RB data. '''

import json
import pathlib

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import  erf


def load_data(file_name):
    
    data_dir = pathlib.Path.cwd().parent.joinpath('system_benchmarking/data')

    with open(data_dir.joinpath(file_name), 'r') as f:
        data = json.load(f)

    return data


def rb_analysis(file_name: str):
    ''' Analyze RB data and return DataFrame of results. '''

    data = load_data(file_name)

    fit_info = {}
    boot_info = {}
    for q in data['survival']:
        xvals = list(data['survival'][q].keys())
        yvals = [
            np.mean(list(vals.values()))/data['shots']
            for vals in data['survival'][q].values()
        ]
        fit_info[q] = expoential_fit(
            xvals, 
            yvals,
            len(q.split(',')),
        )
        boot_info[q] = bootstrap(
            data['survival'][q], 
            data['shots'], 
            len(q.split(',')),
        )

    return fit_info, boot_info


def rb_analysis_combined(file_list):
    ''' Analyze RB data combined for all qubits. '''

    data_list = []
    for file in file_list:
        data_list.append(load_data(file))

    combined_data = {
        qubits: {
            length: {} for length in vals
        } 
        for qubits, vals in data_list[0]['survival'].items()
    }
    for data in data_list:
        for qubits, vals in data['survival'].items():
            for length, reps in vals.items():
                for val in reps.values():
                    combined_data[qubits][length][len(combined_data[qubits][length]) + 1] = val
    fit_info = {}
    boot_info = {}
    qlist = list(combined_data.keys())
    xvals = list(combined_data[qlist[0]])
    ylist = {}
    for x in xvals:
        i = 0
        ylist[x] = {}
        for q in qlist:
            for val in combined_data[q][x].values():
                ylist[x][str(i)] = val
                i += 1

    yvals = [
        np.mean(list(vals.values()))/data['shots']
        for vals in ylist.values()
    ]
    fit_info = expoential_fit(
        xvals, 
        yvals,
        len(qlist[0].split(',')),
    )
    boot_info = bootstrap(
        ylist, 
        data_list[0]['shots'],  # all datasets had same number of shots 
        len(qlist[0].split(',')),
    )
    return 1 - fit_info[1], (boot_info['rate upper'] - boot_info['rate lower'])/2


def expoential_fit(seq_lengths: list,
                   survival_means: list,
                   nqubits: int,
                   initial_guess: list = None):
    ''' Fits survival to exponential decay with asymoptote. '''


    asympt =  1/2**nqubits
    if not initial_guess:
        initial_guess = [1 - 1/2**nqubits, 0.99]

    fit_function = lambda x, A, r: exponential_with_asymptote(x, A, r, asympt)
    bounds = [
        tuple([0]*len(initial_guess)),
        tuple([1]*len(initial_guess))
    ]
    fit_res = curve_fit(
        fit_function,
        seq_lengths,
        survival_means,
        initial_guess,
        bounds=bounds
    )
    metrics = convert_params(
        fit_res[0],
        nqubits,
    )
    return metrics


def convert_params(fit_params,
                   nqubits):
    ''' Convert to standard metrics. '''

    if nqubits == 2:
        ntq = 1.5
    else:
        ntq = 1

    out = [
        fit_params[0] + 1/2**nqubits,
        ((2**nqubits - 1) * fit_params[1] ** (1 / ntq) + 1)/2**nqubits
    ]

    return out


def convert_metrics(metrics_params,
                    nqubits,
                    data_type: str = 'survival'):
    ''' Convert to standard metrics. '''

    if nqubits == 2:
        ntq = 1.5
    else:
        ntq = 1

    out = [
        metrics_params[0] - 1/2**nqubits,
        ((2**nqubits * metrics_params[1] - 1)/(2**nqubits - 1))**ntq
    ]

    return out


def exponential_with_asymptote(seq_len: list, 
                               A: float,
                               r: float,
                               asympt: int):
    ''' Calculate the residuals for 0th order RB survival equation. '''

    survival_prob = A * (r ** seq_len) + asympt

    return survival_prob


def bootstrap(survival,
              shots,
              nqubits,
              resamples: int = 1000):
    ''' Semi-parameteric bootstrap RB data. '''

    xvals = [int(m) for m in survival]
    reps = len(survival[str(xvals[0])])
    boot_sample = {'intercept': [], 'rate': []}
    for _ in range(resamples):
        resampled_reps = np.random.choice(
            np.arange(reps),
            size=(len(xvals), reps),
            replace=True
        )
        sampled_vals = [
            [
                survival[str(l)][str(r)]/shots
                for r in resampled_reps[i]
            ]
            for i, l in enumerate(xvals)
        ]
        resampled_vals = np.random.binomial(
            shots, 
            sampled_vals
        )/shots
        yvals = [
            np.mean(vals)
            for vals in resampled_vals
        ]
        metrics = expoential_fit(
            xvals, 
            yvals,
            nqubits,
        )
        boot_sample['intercept'].append(metrics[0])
        boot_sample['rate'].append(metrics[1])

    thresh = 1/2 + erf(1/np.sqrt(2))/2
    uncertainty = {}
    for param, vals in boot_sample.items():
        uncertainty[param + ' lower'] = (
            2*np.mean(vals) - np.quantile(vals, thresh)
        )
        uncertainty[param + ' upper'] = (
            2*np.mean(vals) - np.quantile(vals, 1-thresh)
        )

    return uncertainty
