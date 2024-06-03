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

from analysis_functions import *
import matplotlib as mpl
mpl.rcParams.update({'errorbar.capsize' : 3})
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rcParams.update({'font.size': 14})
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_t1qrb_survival_curves(t1qrb_survival, t1qrb_uncertainties, t1qrb_seq_lengths, t1qrb_seq_lengths_interp, Nrange):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    cmap = mpl.colormaps.get_cmap('viridis')
    color_list = [cmap(0.2+0.5*(n-16)/(40)) for n in Nrange]

    for i, N in enumerate(Nrange):
        ax.errorbar(t1qrb_seq_lengths,[t1qrb_survival[(N,d)] for d in t1qrb_seq_lengths], yerr = np.array(t1qrb_uncertainties[N][3]).transpose(),fmt="o",color=color_list[i], markerfacecolor=[1,1,1],markeredgecolor=color_list[i])
        
    for i, N in enumerate(Nrange):
        ax.plot(t1qrb_seq_lengths_interp,exponential_with_asymptote(t1qrb_seq_lengths_interp,t1qrb_uncertainties[N][0][0]-1/2,2*t1qrb_uncertainties[N][1][0]-1,1/2),color=color_list[i], markerfacecolor=[1,1,1],markeredgecolor=color_list[i],label = f'N={N}')

    ax.grid(visible=True, axis="both", linestyle="--")
    ax.set_xlabel("Sequence length (number of blocks)")
    ax.set_ylabel("Avg. survival")
    plt.xticks(t1qrb_seq_lengths)
    plt.ylim([0.94, 1.002])
    plt.legend()
    plt.show()

def plot_t1qrb_survival_curve_N40_verification(t1qrb_survival_N40_verification, t1qrb_uncertainties_N40_verification, t1qrb_seq_lengths, t1qrb_seq_lengths_interp):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    color_list = [plt.get_cmap("tab10").colors[i] for i in range(10)]

    ax.errorbar(t1qrb_seq_lengths,[t1qrb_survival_N40_verification[(40,d)] for d in t1qrb_seq_lengths], yerr = np.array(t1qrb_uncertainties_N40_verification[40][3]).transpose(),fmt="o", markerfacecolor=[1,1,1])
    ax.plot(t1qrb_seq_lengths_interp,exponential_with_asymptote(t1qrb_seq_lengths_interp,t1qrb_uncertainties_N40_verification[40][0][0]-1/2,2*t1qrb_uncertainties_N40_verification[40][1][0]-1,1/2),color=color_list[0], markerfacecolor=[1,1,1],markeredgecolor=color_list[0],label = "N=40 verification")
    ax.grid(visible=True, axis="both", linestyle="--")
    ax.set_xlabel("Sequence length (number of blocks)")
    ax.set_ylabel("Avg. survival")
    plt.xticks(t1qrb_seq_lengths)
    plt.ylim([0.94, 1.002])
    plt.legend()
    plt.show()

def plot_mem_error_and_logistic(t1qrb_uncertainties, Nrange, Nrange_interp):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')

    mem_errs = [1-t1qrb_uncertainties[N][1][0] for N in Nrange]
    mem_err_uncerts = [t1qrb_uncertainties[N][1][1] for N in Nrange]
    popt, perrs = logistic_fit(Nrange, mem_errs, mem_err_uncerts)
    ax.plot(Nrange_interp,1e4*logistic(Nrange_interp,popt[0],popt[1],popt[2]),color='black')
    ax.errorbar(Nrange,1e4*np.array(mem_errs), yerr = 1e4*np.array(mem_err_uncerts).transpose(),fmt="o", markerfacecolor=[1,1,1],label = "TSQRB survival")

    ax.grid(visible=True, axis="both", linestyle="--")
    ax.set_xlabel("Number of qubits, N")
    ax.set_ylabel(r"Memory error per qubit per layer ($\times 10^{-4}$)")
    plt.xticks(Nrange)
    plt.ylim([-.1,10])
    plt.show()
    return popt, perrs, mem_errs, mem_err_uncerts


def plot_depth12_data(xeb, xeb_uncertainties, mb_fidelity, mb_uncertainties, gc_depth12, gc_depth12_lower, gc_depth12_upper, Nrange, Nrange_interp):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    color_list = [plt.get_cmap("Dark2").colors[i] for i in range(7)]

    ax.plot(Nrange_interp,gc_depth12, "-", color='black', label = '$F_{GC}$ (gate counting)')
    ax.errorbar(Nrange[:4],[xeb[(N,12)] for N in Nrange[:4]], yerr = np.array([xeb_uncertainties[(N,12)] for N in Nrange[:4]]).transpose() ,fmt="o",color=color_list[1], markerfacecolor=[1,1,1],markeredgecolor=color_list[1],label = "$F_{XEB}$ (linear cross-entropy)")
    ax.errorbar(Nrange,[mb_fidelity[(N,12)] for N in Nrange], yerr = np.array([mb_uncertainties[(N,12)] for N in Nrange]).transpose(),fmt="o",color=color_list[0], markerfacecolor=[1,1,1],zorder=10,markeredgecolor=color_list[0],label = "$F_{MB}$ (MB return probability)")
    ax.fill_between(Nrange_interp, gc_depth12_lower, gc_depth12_upper, alpha = 0.3, color='grey')     
    ax.grid(visible=True, axis="both", linestyle="--")
    ax.set_xlabel("Number of qubits, N")
    ax.set_ylabel("Fidelity estimate")
    plt.xticks(Nrange)
    plt.legend()
    plt.title('d = 12')
    plt.ylim([0.01, 0.9])
    plt.show()

def plot_N56_data(mb_fidelity, mb_uncertainties, gc_N56, gc_N56_lower, gc_N56_upper, spam_fid, spam_uncert, drange, drange_interp):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    rgb_tuples = [(0.4588973877326169, 0.6791204256957736,0.8886379767392907),(0.6532460537228691, 0.47360381303444854,0.7423175066609748),(0.8002298773104436, 0.43601105321386224, 0.6262037532988443),(0.9108684639751516, 0.6297202346256765, 0.5199328466751674),(0.942417154475004, 0.8094639603666023, 0.48640451246247773),(0.8774495485276863, 0.8377864733546613, 0.5363325097376662),(0.640616457823911, 0.629736334022707, 0.5477426349428957),(0.3632787874974958, 0.3557898867323215, 0.35797929042428006), (0.2412007815655248, 0.23161827390816775, 0.23441958857592546), (0.18688993133389536, 0.176376006346306, 0.17944955191354484),(0.14939975360681784, 0.1382428815635377, 0.14150434510224663), (0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444),(0.133532, 0.122103, 0.125444)]
    cm1 = LinearSegmentedColormap.from_list('',rgb_tuples[4:])
    cm1.set_bad('#FFFFFF')
    color_list = [plt.get_cmap("Dark2").colors[i] for i in range(7)]
    alpha = 0.85

    mb24_d12_estimate = np.sqrt(mb_fidelity[(56,24)]/(spam_fid)**56)*(spam_fid)**56 
    mb24_d12_lower_unc = np.sqrt((mb_uncertainties[(56,24)][0]*mb24_d12_estimate/(2*mb_fidelity[(56,24)]))**2+(spam_uncert*28*mb24_d12_estimate / spam_fid)**2)
    mb24_d12_upper_unc = np.sqrt((mb_uncertainties[(56,24)][1]*mb24_d12_estimate/(2*mb_fidelity[(56,24)]))**2+(spam_uncert*28*mb24_d12_estimate / spam_fid)**2)
    ax.plot(drange_interp,gc_N56, "-", color='black', label = '$F_{GC}$ (gate counting)',zorder=2)
    ax.errorbar(drange,[mb_fidelity[(56,d)] for d in drange], yerr = np.array([mb_uncertainties[(56,d)] for d in drange]).transpose(),fmt="o",color=color_list[0], markerfacecolor=[1,1,1],markeredgecolor=color_list[0],label = "$F_{MB}$ (MB return probability)")
    ax.errorbar([12],[mb24_d12_estimate], yerr = np.array([(mb24_d12_lower_unc,mb24_d12_upper_unc)]).transpose(),fmt="o",color=color_list[3], markerfacecolor=[1,1,1],markeredgecolor=color_list[3],label = "$F_{MB}$ (from MB at depth 24)")
    ax.fill_between(drange_interp, gc_N56_lower, gc_N56_upper, alpha = 0.3, color='grey',zorder=2)

    x, y = np.meshgrid(np.linspace(7.5, 20.5, 1000), np.linspace(.01, 0.1, 1000))
    p=ax.fill_between(np.arange(7.5,20.5,0.01),[0.01 for _ in np.arange(7.5,20.5,0.01)],[0.1 for _ in np.arange(7.5,20.5,0.01)],alpha=alpha,color='none')
    gradient = plt.imshow(x,cmap=cm1, aspect='auto', origin='lower',extent = [7.5,20.5,.01,0.1],alpha=alpha)
    gradient.set_clip_path(p.get_paths()[0], transform=plt.gca().transData)

    ax.grid(visible=True, axis="y", linestyle="--")
    ax.set_xlabel("Depth, d", fontsize=14)
    ax.annotate(r'~statevector hardness $\longrightarrow$',(12.1,0.02),xytext=(12.1,0.12), color='black',alpha=1)
    ax.set_ylabel("Fidelity estimate",  fontsize=14)
    ax.axvline(x=12, color='#d3d3d3', ls='--',zorder=1)
    plt.xticks(drange)
    plt.legend()
    plt.title('N = 56')
    plt.ylim([0.01, 0.9])
    ax.set_xlim([7.5,20.5])
    plt.show()

def plot_N40_data(xeb_N40_verification, xeb_uncertainties_N40_verification, mb_fidelity_N40_verification, mb_uncertainties_N40_verification, gc_N40, gc_N40_lower, gc_N40_upper, drange, drange_interp):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    color_list = [plt.get_cmap("Dark2").colors[i] for i in range(7)]

    ax.plot(drange_interp,gc_N40, "-", color='black', label = '$F_{GC}$ (gate counting)')
    ax.errorbar(drange,[xeb_N40_verification[(40,d)] for d in drange], yerr = np.array([xeb_uncertainties_N40_verification[(40,d)] for d in drange]).transpose(),fmt="o",color=color_list[1], markerfacecolor=[1,1,1],markeredgecolor=color_list[1],label = "$F_{XEB}$ (linear cross-entropy)")
    ax.errorbar(drange,[mb_fidelity_N40_verification[(40,d)] for d in drange], yerr = np.array([mb_uncertainties_N40_verification[(40,d)] for d in drange]).transpose(),fmt="o",color=color_list[0], markerfacecolor=[1,1,1],markeredgecolor=color_list[0],label = "$F_{MB}$ (MB return probability)")
    ax.fill_between(drange_interp, gc_N40_lower, gc_N40_upper, alpha = 0.3, color='grey')     
    ax.grid(visible=True, axis="both", linestyle="--")
    ax.set_xlabel("Depth, d", fontsize=14)
    ax.set_ylabel("Fidelity estimate")
    plt.xticks(drange)
    plt.legend()
    plt.title('N = 40')
    plt.ylim([0.01, 0.9])
    plt.show()