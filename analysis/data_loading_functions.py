# Copyright 2024 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import json
import ast
import numpy as np
from analysis_functions import *

def load_data(file_path):
    
    data_dir = pathlib.Path.cwd().parent

    with open(data_dir.joinpath(file_path), 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def load_XEB_results(Nd_pairs_xeb, n_resamples):
    probabilities = dict()
    xeb_counts = dict()
    xeb = dict()
    xeb_uncertainties = dict()

    for i, Nd in enumerate(Nd_pairs_xeb):
        N = Nd[0]
        if N <= 40:
            d = Nd[1]
            probabilities[(N,d)] = []
            xeb_counts[(N,d)] = []
            for r in range(1,51):
                filename=f'amplitudes/N_scan_depth12/N{N}_d{d}_XEB/N{N}_d{d}_r{r}_XEB_amplitudes.json'
                amps_json = load_data(filename)
                filename = f'results/N_scan_depth12/N{N}_d{d}_XEB/N{N}_d{d}_r{r}_XEB_counts.json'
                counts_json = load_data(filename)
                amp_dict = {ast.literal_eval(k):np.array(v,dtype='complex128').item() for k,v in amps_json.items()}
                counts_dict = {ast.literal_eval(k):v for k,v in counts_json.items()}
                prob_dict = {k:abs(v)**2 for k,v in amp_dict.items()}
                xeb[(N,d,r)]= XEB(counts_dict, prob_dict, N)
                probabilities[(N,d)].append(prob_dict)
                xeb_counts[(N,d)].append(counts_dict)
            xeb[(N,d)] = np.mean([xeb[(N,d,r)] for r in range(1,51)])
            xeb_uncertainties[(N,d)] = XEB_bootstrap_uncertainty(xeb[(N,d)],xeb_counts[(N,d)],probabilities[(N,d)],n_resamples)
    return xeb, xeb_uncertainties

def load_XEB_results_N40_verification(Nd_pairs_xeb_N40_verification, n_resamples):
    probabilities_N40_verification = dict()
    xeb_counts_N40_verification = dict()
    xeb_N40_verification = dict()
    xeb_uncertainties_N40_verification = dict()
    for i, Nd in enumerate(Nd_pairs_xeb_N40_verification):
        N = Nd[0]
        d = Nd[1]
        probabilities_N40_verification[(N,d)] = []
        xeb_counts_N40_verification[(N,d)] = []
        for r in range(1,51):
            filename=f'amplitudes/N40_verification/N{N}_d{d}_XEB/N{N}_d{d}_r{r}_XEB_amplitudes.json'
            amps_json = load_data(filename)
            filename = f'results/N40_verification/N{N}_d{d}_XEB/N{N}_d{d}_r{r}_XEB_counts.json'
            counts_json = load_data(filename)
            amp_dict = {ast.literal_eval(k):np.array(v,dtype='complex128').item() for k,v in amps_json.items()}
            counts_dict = {ast.literal_eval(k):v for k,v in counts_json.items()}
            prob_dict = {k:abs(v)**2 for k,v in amp_dict.items()}
            xeb_N40_verification[(N,d,r)] = XEB(counts_dict, prob_dict, N)
            probabilities_N40_verification[(N,d)].append(prob_dict)
            xeb_counts_N40_verification[(N,d)].append(counts_dict)
        xeb_N40_verification[(N,d)] = np.mean([xeb_N40_verification[(N,d,r)] for r in range(1,51)])
        xeb_uncertainties_N40_verification[(N,d)] = XEB_bootstrap_uncertainty(xeb_N40_verification[(N,d)],xeb_counts_N40_verification[(N,d)],probabilities_N40_verification[(N,d)],n_resamples)
    return xeb_N40_verification, xeb_uncertainties_N40_verification

def load_MB_results(Nd_pairs_mb, n_resamples):
    mb_counts = dict()
    mb_fidelity = dict()
    mb_uncertainties = dict()
    mb_ideals = dict()
    for i, Nd in enumerate(Nd_pairs_mb):
        N = Nd[0]
        d = Nd[1]
        mb_counts[(N,d)] = []
        mb_ideals[(N,d)] = []
        for r in range(1,51):
            if N < 56:
                filename = f'results/N_scan_depth12/N{N}_d{d}_MB/N{N}_d{d}_r{r}_MB_counts.json'
                counts_json = load_data(filename)
                filename = f'results/N_scan_depth12/N{N}_d{d}_MB/N{N}_d{d}_r{r}_MB_ideal_bitstring.json'
                ideal = tuple(load_data(filename))
            else:
                filename = f'results/N56_depths/N{N}_d{d}_MB/N{N}_d{d}_r{r}_MB_counts.json'
                counts_json = load_data(filename)
                filename = f'results/N56_depths/N{N}_d{d}_MB/N{N}_d{d}_r{r}_MB_ideal_bitstring.json'
                ideal = tuple(load_data(filename))
                
            mb_ideals[(N,d)].append(ideal)
            counts_dict = {ast.literal_eval(k):v for k,v in counts_json.items()}
            mb_counts[(N,d)].append(counts_dict)
        mb_fidelity[(N,d)] = return_probability(bs_with_ideal(mb_counts[(N,d)], mb_ideals[(N,d)]))
        mb_uncertainties[(N,d)] = MB_bootstrap_uncertainty(bs_with_ideal(mb_counts[(N,d)], mb_ideals[(N,d)]), n_resamples)
    return mb_fidelity, mb_uncertainties

def load_MB_results_N40_verification(Nd_pairs_mb_N40_verification, n_resamples):
    mb_counts_N40_verification = dict()
    mb_ideals_N40_verification = dict()
    mb_fidelity_N40_verification = dict()
    mb_uncertainties_N40_verification = dict()
    for i, Nd in enumerate(Nd_pairs_mb_N40_verification):
        N = Nd[0]
        d = Nd[1]
        mb_counts_N40_verification[(N,d)] = []
        mb_ideals_N40_verification[(N,d)] = []
        for r in range(1,51):
            filename = f'results/N40_verification/N{N}_d{d}_MB/N{N}_d{d}_r{r}_MB_counts.json'
            counts_json = load_data(filename)
            filename = f'results/N40_verification/N{N}_d{d}_MB/N{N}_d{d}_r{r}_MB_ideal_bitstring.json'
            ideal = tuple(load_data(filename))
            mb_ideals_N40_verification[(N,d)].append(ideal)
            counts_dict = {ast.literal_eval(k):v for k,v in counts_json.items()}
            mb_counts_N40_verification[(N,d)].append(counts_dict)
        mb_fidelity_N40_verification[(N,d)] = return_probability(bs_with_ideal(mb_counts_N40_verification[(N,d)], mb_ideals_N40_verification[(N,d)]))
        mb_uncertainties_N40_verification[(N,d)] = MB_bootstrap_uncertainty(bs_with_ideal(mb_counts_N40_verification[(N,d)], mb_ideals_N40_verification[(N,d)]), n_resamples)
    return mb_fidelity_N40_verification, mb_uncertainties_N40_verification


def load_t1qrb_results(Nd_pairs_t1qrb, n_resamples, t1qrb_seq_lengths):
    t1qrb_counts = dict()
    t1qrb_survival = dict()
    t1qrb_uncertainties = dict()
    t1qrb_ideals = dict()

    for i, Nd in enumerate(Nd_pairs_t1qrb):
        N = Nd[0]
        d = Nd[1]
        t1qrb_counts[(N,d)] = []
        t1qrb_ideals[(N,d)] = []
        for r in range(1,11):
            if N < 56:
                filename = f'results/N_scan_depth12/N{N}_d{d}_Transport_1QRB/N{N}_d{d}_r{r}_Transport_1QRB_counts.json'
                counts_json = load_data(filename)
                filename = f'results/N_scan_depth12/N{N}_d{d}_Transport_1QRB/N{N}_d{d}_r{r}_Transport_1QRB_ideal_bitstring.json'
                ideal = tuple(load_data(filename))
            else:
                filename = f'results/N56_depths/N{N}_d{d}_Transport_1QRB/N{N}_d{d}_r{r}_Transport_1QRB_counts.json'
                counts_json = load_data(filename)
                filename = f'results/N56_depths/N{N}_d{d}_Transport_1QRB/N{N}_d{d}_r{r}_Transport_1QRB_ideal_bitstring.json'
                ideal = tuple(load_data(filename))
                
            t1qrb_ideals[(N,d)].append(ideal)
            counts_dict = {ast.literal_eval(k):v for k,v in counts_json.items()}
            t1qrb_counts[(N,d)].append(counts_dict)
        t1qrb_survival[(N,d)] = return_probability_averaged(bs_with_ideal(t1qrb_counts[(N,d)], t1qrb_ideals[(N,d)]),N)
        if d == t1qrb_seq_lengths[-1]:
            t1qrb_uncertainties[N] = transport_1qrb_bootstrap_uncertainty_resampled_jobs([t1qrb_counts[(N,d)] for d in t1qrb_seq_lengths], [t1qrb_ideals[(N,d)] for d in t1qrb_seq_lengths], n_resamples, t1qrb_seq_lengths, N)
    
    return t1qrb_survival, t1qrb_uncertainties


def load_t1qrb_results_N40_verification(Nd_pairs_t1qrb_N40_verification, n_resamples, t1qrb_seq_lengths):
    t1qrb_counts_N40_verification = dict()
    t1qrb_ideals_N40_verification = dict()
    t1qrb_survival_N40_verification = dict()
    t1qrb_uncertainties_N40_verification = dict()

    for i, Nd in enumerate(Nd_pairs_t1qrb_N40_verification):
        N = Nd[0]
        d = Nd[1]
        t1qrb_counts_N40_verification[(N,d)] = []
        t1qrb_ideals_N40_verification[(N,d)] = []
        for r in range(1,11):
            filename = f'results/N40_verification/N{N}_d{d}_Transport_1QRB/N{N}_d{d}_r{r}_Transport_1QRB_counts.json'
            counts_json = load_data(filename)
            filename = f'results/N40_verification/N{N}_d{d}_Transport_1QRB/N{N}_d{d}_r{r}_Transport_1QRB_ideal_bitstring.json'
            ideal = tuple(load_data(filename))
            t1qrb_ideals_N40_verification[(N,d)].append(ideal)
            counts_dict = {ast.literal_eval(k):v for k,v in counts_json.items()}
            t1qrb_counts_N40_verification[(N,d)].append(counts_dict)
        t1qrb_survival_N40_verification[(N,d)] = return_probability_averaged(bs_with_ideal(t1qrb_counts_N40_verification[(N,d)], t1qrb_ideals_N40_verification[(N,d)]),N)
        if d == t1qrb_seq_lengths[-1]:
            t1qrb_uncertainties_N40_verification[N] = transport_1qrb_bootstrap_uncertainty_resampled_jobs([t1qrb_counts_N40_verification[(N,d)] for d in t1qrb_seq_lengths], [t1qrb_ideals_N40_verification[(N,d)] for d  in t1qrb_seq_lengths], n_resamples, t1qrb_seq_lengths, N)
    
    return t1qrb_survival_N40_verification, t1qrb_uncertainties_N40_verification
