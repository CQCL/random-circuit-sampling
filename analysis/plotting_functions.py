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

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams.update({'errorbar.capsize' : 3})
color_list = [plt.get_cmap("tab10").colors[i] for i in range(10)]

def plot_transport_1qrb(seq_lengths, return_probs, return_uncerts, N):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    ax.errorbar(seq_lengths,return_probs[N], yerr = np.array(return_uncerts[N][3]).transpose(),fmt="o",color=color_list[0], markerfacecolor=[1,1,1],markeredgecolor=color_list[0],label = "Survival probability")
    ax.grid(visible=True, axis="both", linestyle="--")
    ax.set_xlabel("Depth, d", fontsize=12)
    ax.set_ylabel("Average survival",  fontsize=12)
    plt.xticks(seq_lengths)
    plt.title('RCS Transport 1QRB N = 16')
    plt.ylim([0.9, 1])
    plt.show()