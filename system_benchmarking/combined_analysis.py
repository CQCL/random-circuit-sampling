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

''' Functions for combining analysis across qubits and datasets. '''

import pandas as pd

from rb_analysis_functions import rb_analysis_combined


def combined_report(file_list: list = None):
    ''' Make table of estimates from all methods. '''

    renamed = {
        '2q_cliff_rb_H2-1-N56_2024-05-01_1656.json': '2024-05-01_1656',
        '2q_cliff_rb_H2-1-N56_2024-05-02_0947.json': '2024-05-02_0947', 
        '2q_cliff_rb_H2-1-N56_2024-05-03_1114.json': '2024-05-03_1114',
        '2q_cliff_rb_H2-1-N56_2024-05-07_0830.json': '2024-05-07_0830',
        '2q_cliff_rb_H2-1-N56_2024-05-07_1559.json': '2024-05-07_1559',
        '2q_cliff_rb_H2-1-N56_2024-05-08_1009.json': '2024-05-08_1009',
        '2q_cliff_rb_H2-1-N56_2024-05-09_0814.json': '2024-05-09_0814',
        'combined': 'combined'
    }
    df_raw = extract_parameters(file_list)

    df = {}
    for old_name, new_name in renamed.items():
        if old_name in df_raw:
            df[new_name] = [df_raw[old_name][0], df_raw[old_name][1]]
        else:
            df[new_name] = None
    result = pd.DataFrame.from_dict(df).transpose()
    result.rename(columns={0: '2Q gate infid.', 1: '2Q gate uncertainty'}, inplace=True)
    pd.set_option('display.float_format', lambda x: '%.3E' % x)

    return result


def extract_parameters(file_list: list = None):
    ''' Extract parameters from all tests combined over gate zones. '''

    if file_list is None:
        file_list = [
            '2q_cliff_rb_H2-1-N56_2024-05-01_1656.json',
            '2q_cliff_rb_H2-1-N56_2024-05-02_0947.json',
            '2q_cliff_rb_H2-1-N56_2024-05-03_1114.json',
            '2q_cliff_rb_H2-1-N56_2024-05-07_0830.json',
            '2q_cliff_rb_H2-1-N56_2024-05-07_1559.json',
            '2q_cliff_rb_H2-1-N56_2024-05-08_1009.json',
            '2q_cliff_rb_H2-1-N56_2024-05-09_0814.json',
        ]

    df = {}
    for file in file_list:
        fid, unc = rb_analysis_combined([file])
    
        df[file] = [fid, unc]

    fid, unc = rb_analysis_combined(file_list)
    df['combined'] = [fid, unc]

    return df


def first_sig_fig(val, unc):

    est = '{:.1uE}'.format(ufloat(val, unc))

    vals, pow = est.split('E')

    sig_val = vals.split('+')[0][1:]

    return float(sig_val + 'E' + pow)