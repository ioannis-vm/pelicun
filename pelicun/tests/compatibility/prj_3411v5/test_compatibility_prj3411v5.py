# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of pelicun.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# pelicun. If not, see <http://www.opensource.org/licenses/>.

"""
Verifies backwards compatibility with the following:

    DesignSafe PRJ-3411 > Example01_FEMA_P58_Introduction

There are some changes made to the code and input files, so the output
is not the same with what the original code would produce. We only
want to confirm that executing this code does not raise an error.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pelicun.assessment import Assessment
from pelicun.base import convert_to_MultiIndex
from pelicun.warnings import PelicunWarning


def test_compatibility_DesignSafe_PRJ_3411_Example01() -> None:
    sample_size = 10000
    raw_demands = pd.read_csv(
        'pelicun/tests/compatibility/prj_3411v5/demand_data.csv', index_col=0
    )
    raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
    raw_demands.index.names = ['stripe', 'type', 'loc', 'dir']
    stripe = '3'
    stripe_demands = raw_demands.loc[stripe, :]
    stripe_demands.insert(0, 'Units', '')
    stripe_demands.loc['PFA', 'Units'] = 'g'
    stripe_demands.loc['PID', 'Units'] = 'rad'
    stripe_demands.insert(1, 'Family', '')
    stripe_demands['Family'] = 'lognormal'
    stripe_demands = stripe_demands.rename(columns={'median': 'Theta_0'})  # type: ignore
    stripe_demands = stripe_demands.rename(columns={'log_std': 'Theta_1'})  # type: ignore
    ndims = stripe_demands.shape[0]
    demand_types = stripe_demands.index
    perfect_corr = pd.DataFrame(
        np.ones((ndims, ndims)), columns=demand_types, index=demand_types
    )

    pal = Assessment({'PrintLog': True, 'Seed': 415})
    pal.demand.load_model({'marginals': stripe_demands, 'correlation': perfect_corr})
    pal.demand.generate_sample({'SampleSize': sample_size})

    demand_sample = pal.demand.save_sample()

    delta_y = 0.0075
    pid = demand_sample['PID']  # type: ignore
    rid = pal.demand.estimate_RID(pid, {'yield_drift': delta_y})
    demand_sample_ext = pd.concat([demand_sample, rid], axis=1)  # type: ignore
    sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756]
    demand_sample_ext[('SA_1.13', 0, 1)] = sa_vals[int(stripe) - 1]
    demand_sample_ext.T.insert(0, 'Units', '')
    demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
    demand_sample_ext.loc['Units', ['PID', 'RID']] = 'rad'

    pal.demand.load_sample(demand_sample_ext)

    cmp_marginals = pd.read_csv(
        'pelicun/tests/compatibility/prj_3411v5/CMP_marginals.csv', index_col=0
    )

    pal.stories = 4
    pal.asset.load_cmp_model({'marginals': cmp_marginals})
    pal.asset.generate_cmp_sample()

    cmp_sample = pal.asset.save_cmp_sample()
    assert cmp_sample is not None

    with pytest.warns(PelicunWarning):
        p58_data = pal.get_default_data('fragility_DB_FEMA_P58_2nd')

    cmp_list = cmp_marginals.index.unique().to_numpy()[:-3]
    p58_data_for_this_assessment = p58_data.loc[cmp_list, :].sort_values(
        'Incomplete', ascending=False
    )
    additional_fragility_db = p58_data_for_this_assessment.sort_index()

    p58_metadata = pal.get_default_metadata('fragility_DB_FEMA_P58_2nd')
    assert p58_metadata is not None

    additional_fragility_db.loc[
        ['D.20.22.013a', 'D.20.22.023a', 'D.20.22.023b'],
        [('LS1', 'Theta_1'), ('LS2', 'Theta_1')],
    ] = 0.5
    additional_fragility_db.loc['D.20.31.013b', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.20.61.013b', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.30.31.013i', ('LS1', 'Theta_0')] = 1.5  # g
    additional_fragility_db.loc['D.30.31.013i', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.30.31.023i', ('LS1', 'Theta_0')] = 1.5  # g
    additional_fragility_db.loc['D.30.31.023i', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.30.52.013i', ('LS1', 'Theta_0')] = 1.5  # g
    additional_fragility_db.loc['D.30.52.013i', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db['Incomplete'] = 0

    additional_fragility_db.loc[
        'excessiveRID',
        [
            ('Demand', 'Directional'),
            ('Demand', 'Offset'),
            ('Demand', 'Type'),
            ('Demand', 'Unit'),
        ],
    ] = [1, 0, 'Residual Interstory Drift Ratio', 'rad']
    additional_fragility_db.loc[
        'excessiveRID', [('LS1', 'Family'), ('LS1', 'Theta_0'), ('LS1', 'Theta_1')]
    ] = ['lognormal', 0.01, 0.3]
    additional_fragility_db.loc[
        'irreparable',
        [
            ('Demand', 'Directional'),
            ('Demand', 'Offset'),
            ('Demand', 'Type'),
            ('Demand', 'Unit'),
        ],
    ] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']
    additional_fragility_db.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
    additional_fragility_db.loc[
        'collapse',
        [
            ('Demand', 'Directional'),
            ('Demand', 'Offset'),
            ('Demand', 'Type'),
            ('Demand', 'Unit'),
        ],
    ] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']
    additional_fragility_db.loc[
        'collapse', [('LS1', 'Family'), ('LS1', 'Theta_0'), ('LS1', 'Theta_1')]
    ] = ['lognormal', 1.35, 0.5]
    additional_fragility_db['Incomplete'] = 0

    with pytest.warns(PelicunWarning):
        pal.damage.load_damage_model(
            [additional_fragility_db, 'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv']
        )

    # FEMA P58 uses the following process:
    dmg_process = {
        '1_collapse': {'DS1': 'ALL_NA'},
        '2_excessiveRID': {'DS1': 'irreparable_DS1'},
    }
    pal.damage.calculate(dmg_process=dmg_process)

    damage_sample = pal.damage.save_sample()
    assert damage_sample is not None

    drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
    drivers = drivers[:-3] + drivers[-2:]

    loss_models = cmp_marginals.index.unique().tolist()[:-3]
    loss_models += ['replacement'] * 2
    loss_map = pd.DataFrame(loss_models, columns=['BldgRepair'], index=drivers)

    with pytest.warns(PelicunWarning):
        p58_data = pal.get_default_data('bldg_repair_DB_FEMA_P58_2nd')

    p58_data_for_this_assessment = p58_data.loc[
        loss_map['BldgRepair'].to_numpy()[:-2], :
    ]

    additional_consequences = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(
            [
                ('Incomplete', ''),
                ('Quantity', 'Unit'),
                ('DV', 'Unit'),
                ('DS1', 'Theta_0'),
            ]
        ),
        index=pd.MultiIndex.from_tuples(
            [('replacement', 'Cost'), ('replacement', 'Time')]
        ),
    )
    additional_consequences.loc[('replacement', 'Cost')] = [
        0,
        '1 EA',
        'USD_2011',
        21600000,
    ]
    additional_consequences.loc[('replacement', 'Time')] = [
        0,
        '1 EA',
        'worker_day',
        12500,
    ]

    with pytest.warns(PelicunWarning):
        pal.bldg_repair.load_model(
            [
                additional_consequences,
                'PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv',
            ],
            loss_map,
        )

    pal.bldg_repair.calculate()

    loss_sample = pal.bldg_repair.sample
    assert loss_sample is not None

    with pytest.warns(PelicunWarning):
        agg_df = pal.bldg_repair.aggregate_losses()
    assert agg_df is not None
