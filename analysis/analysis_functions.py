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

def gate_counting(N, d, tq_process_fid, mem_process_fid, spam_fid, tq_depth_shift, mem_depth_shift):
    return  tq_process_fid**(N*(d - tq_depth_shift)/2) * mem_process_fid**(N*(d - mem_depth_shift)) * spam_fid**N 

def XEB(exp_counts, ideal_probs, N):
    shots = sum(exp_counts.values())
    avg = 0
    for bs in list(exp_counts.keys()):
        avg += exp_counts[bs] * ideal_probs[bs]
    avg /= shots
        
    return (2**N) * avg - 1