# Copyright 2024 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import choices
import numpy as np
from scipy.optimize import curve_fit
from typing import Optional

def XEB(exp_counts, ideal_probs, N):
    shots = sum(exp_counts.values())
    avg = 0
    for bs in list(exp_counts.keys()):
        avg += exp_counts[bs] * ideal_probs[bs]
    avg /= shots
        
    return (2**N) * avg - 1

def XEB_bootstrap_uncertainty(xeb_emp, counts, probs, n_resamples):
    bs_with_corresp_probs = []
    for i, counter in enumerate(counts):
        for bs in counter.keys():
            for _ in range(counter[bs]):
                bs_with_corresp_probs.append((bs, probs[i][bs]))
    num_total_shots = len(bs_with_corresp_probs)

    N = len(list(counts[0].keys())[0])
    xeb_list = []
    for _ in range(n_resamples):
        bootstrapped_shots_with_probs = choices(bs_with_corresp_probs, k=num_total_shots)
        avg = 0
        for bs_prob in bootstrapped_shots_with_probs:
            avg += bs_prob[1]
        avg /= num_total_shots
        
        xeb = (2**N) * avg - 1
        xeb_list.append(xeb)

    lower = 2*xeb_emp - np.quantile(xeb_list,1-0.1587)
    upper = 2*xeb_emp - np.quantile(xeb_list, 0.1587)
    return xeb_emp - lower, upper - xeb_emp

def return_probability(bitstrings_with_ideal_bitstrings):
    correct = 0
    nshots = len(bitstrings_with_ideal_bitstrings)
    for bs_with_ideal in bitstrings_with_ideal_bitstrings:
        if bs_with_ideal[0] == bs_with_ideal[1]:
            correct += 1
    return correct / nshots

def bs_with_ideal(counts, ideals):
    bitstrings_with_ideal_bitstrings = []
    for i, counter in enumerate(counts):
        for bs in counter.keys():
            for _ in range(counter[bs]):
                bitstrings_with_ideal_bitstrings.append((bs, ideals[i]))
    return bitstrings_with_ideal_bitstrings

def MB_bootstrap_uncertainty(mb_bitstrings_with_ideal_bitstrings, n_resamples):
    empirical_mean = return_probability(mb_bitstrings_with_ideal_bitstrings)
    n_total_shots = len(mb_bitstrings_with_ideal_bitstrings)
    resampled_values = []
    for _ in range(n_resamples):
        resampled_counts = choices(mb_bitstrings_with_ideal_bitstrings,k = n_total_shots)
        prob = return_probability(resampled_counts)
        resampled_values.append(prob)

    lower = 2*empirical_mean - np.quantile(resampled_values,1-0.1587)
    upper = 2*empirical_mean - np.quantile(resampled_values, 0.1587)
    return empirical_mean - lower, upper - empirical_mean

def return_probability_averaged(bitstrings_with_ideal_bitstrings,N):
    correct_list = {i:0 for i in range(N)}
    nshots = len(bitstrings_with_ideal_bitstrings)
    for bs_with_ideal in bitstrings_with_ideal_bitstrings:
        bs = bs_with_ideal[0]
        ideal = bs_with_ideal[1]
        for i in range(N):
            if bs[i] == ideal[i]:
                correct_list[i] += 1
    returns = {i:(correct_list[i] / nshots) for i in range(N)}
    return np.mean(list(returns.values()))

def transport_1qrb_bootstrap_uncertainty_resampled_jobs(counts_list, ideals_list, n_resamples, seq_lengths, N):
    empirical_means = []
    resampled_value_lists = [[] for _ in range(len(counts_list))]
    A_values = []
    mem_values = []
    for j,counts in enumerate(counts_list):
        ideals = ideals_list[j]

        t1qrb_bitstrings_with_ideal_bitstrings = []
        for i, counter in enumerate(counts):
            for bs in counter.keys():
                for _ in range(counter[bs]):
                    t1qrb_bitstrings_with_ideal_bitstrings.append((bs, ideals[i]))

        empirical_mean = return_probability_averaged(t1qrb_bitstrings_with_ideal_bitstrings, N)
        empirical_means.append(empirical_mean)
        n_total_shots = len(t1qrb_bitstrings_with_ideal_bitstrings)
        for _ in range(n_resamples):
            #just resample shots
            resampled_counts = choices(t1qrb_bitstrings_with_ideal_bitstrings,k = n_total_shots)
            prob = return_probability_averaged(resampled_counts,N)
            resampled_value_lists[j].append(prob)

    for resample in range(len(resampled_value_lists[0])):
        t1qrb_returns = [resampled_value_lists[i][resample] for i in range(len(resampled_value_lists))]
        A, mem_fid = exponential_fit(seq_lengths,t1qrb_returns,1)
        A_values.append(A)
        mem_values.append(mem_fid)

    empirical_A, empirical_mem = exponential_fit(seq_lengths,empirical_means,1)

    lowers = [empirical_means[i] - (2*empirical_means[i] - np.quantile(resampled_value_lists[i],1-0.1587)) for i in range(len(resampled_value_lists))]
    uppers = [empirical_means[i] - np.quantile(resampled_value_lists[i], 0.1587) for i in range(len(resampled_value_lists))]
    t1qrb_uncertainty_pairs = [(lowers[i], uppers[i]) for i in range(len(lowers))]

    return [(empirical_A,np.std(A_values)), (empirical_mem, np.std(mem_values)), empirical_means, t1qrb_uncertainty_pairs]


def exponential_fit(seq_lengths: list,
                   survival_means: list,
                   nqubits: int,
                   initial_guess: Optional[list] = None,
                   verbose: Optional[bool] = False):
    ''' Fits survival to exponential decay with asymptote. '''

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
    if verbose:
        print(fit_res)
    metrics = convert_params(
        fit_res[0],
        nqubits
    )
    return metrics

def convert_params(fit_params,
                   nqubits):
    ''' Convert to standard metrics. '''

    out = [
            fit_params[0] + 1/2**nqubits,
            ((2**nqubits - 1) * fit_params[1]  + 1)/2**nqubits
        ]

    return out
def exponential_with_asymptote(seq_len: list, 
                               A: float,
                               r: float,
                               asympt: int):
    ''' Calculate the residuals for 0th order RB survival equation. '''

    survival_prob = A * (r ** seq_len) + asympt

    return survival_prob

def logistic(x,A,x0,k):
    return A / (1 + np.exp(-k*(x-x0)))

def logistic_err(x,A,x0,k,Aerr,x0err,kerr):
    return np.sqrt((Aerr*logistic(x,A,x0,k)/A)**2+(kerr*A*np.exp(-k*(x-x0))*(x-x0)/(1+np.exp(-k*(x-x0)))**2)**2+(x0err*A*k*np.exp(-k*(x-x0))/(1+np.exp(-k*(x-x0)))**2)**2)

def logistic_fit(Nrange, mem_errs, t1qrb_uncerts):
    popt, pcov =curve_fit(logistic, Nrange, mem_errs, sigma = t1qrb_uncerts,p0=[.0005,18,0.1])
    perrs=np.sqrt(np.diag(pcov))
    return popt, perrs

def gate_counting(N, d, eff_tq_process_fid, spam_fid, eff_depth_shift):
    return ((eff_tq_process_fid)**((N/2)*(d-eff_depth_shift))) * (spam_fid)**N 

def gate_counting_uncert_Nscan(tq_fid, tq_uncert, spam_fid, spam_uncert, eff_tq_depth_shift, popt, perrs, gc, Nrange_interp):
    return [np.sqrt((spam_uncert*N*gc[i]/spam_fid)**2 + (effective_2Q_uncert(tq_uncert, logistic_err(N,popt[0],popt[1],popt[2],perrs[0],perrs[1],perrs[2]))*(N*(12-eff_tq_depth_shift)/2)*gc[i]/effective_2Q(tq_fid, mem_err(N,popt)))**2) for i,N in enumerate(Nrange_interp)]

def gate_counting_uncert_dscan(tq_fid, tq_uncert, spam_fid, spam_uncert, eff_tq_depth_shift, mem_err, mem_err_uncert, N, gc, drange_interp):
    return [np.sqrt((spam_uncert*N*gc[i]/spam_fid)**2 + (effective_2Q_uncert(tq_uncert, mem_err_uncert)*(N*(d-eff_tq_depth_shift)/2)*gc[i]/effective_2Q(tq_fid, mem_err))**2) for i,d in enumerate(drange_interp)]

def mem_err(N, popt):
    return logistic(N,popt[0],popt[1],popt[2])

def effective_2Q(tq_fid, memory_err):
    return 1- ((5/4)*(1-tq_fid) + 2*(3/2)*memory_err)

def effective_2Q_uncert(tq_uncert, mem_uncert):
    return np.sqrt((25/16)*tq_uncert**2+9*mem_uncert**2)