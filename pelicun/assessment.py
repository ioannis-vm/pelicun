# -*- coding: utf-8 -*-
#
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
#
# Contributors:
# Adam Zsarnóczay
# John Vouvakis Manousakis

"""
This module has classes and methods that control the performance assessment.

"""

from __future__ import annotations
from typing import Any
import json
import numpy as np
import pandas as pd
from pelicun.base import get
from pelicun import base
from pelicun import file_io
from pelicun import model
from pelicun.base import EDP_to_demand_type

from pelicun.__init__ import __version__ as pelicun_version  # type: ignore


# pylint: disable=consider-using-namedtuple-or-dataclass

default_DBs = {
    'fragility': {
        'FEMA P-58': 'damage_DB_FEMA_P58_2nd.csv',
        'Hazus Earthquake - Buildings': 'damage_DB_Hazus_EQ_bldg.csv',
        'Hazus Earthquake - Stories': 'damage_DB_Hazus_EQ_story.csv',
        'Hazus Earthquake - Transportation': 'damage_DB_Hazus_EQ_trnsp.csv',
        'Hazus Earthquake - Water': 'damage_DB_Hazus_EQ_water.csv',
        'Hazus Hurricane': 'damage_DB_SimCenter_Hazus_HU_bldg.csv',
    },
    'repair': {
        'FEMA P-58': 'loss_repair_DB_FEMA_P58_2nd.csv',
        'Hazus Earthquake - Buildings': 'loss_repair_DB_Hazus_EQ_bldg.csv',
        'Hazus Earthquake - Stories': 'loss_repair_DB_Hazus_EQ_story.csv',
        'Hazus Earthquake - Transportation': 'loss_repair_DB_Hazus_EQ_trnsp.csv',
        'Hazus Hurricane': 'loss_repair_DB_SimCenter_Hazus_HU_bldg.csv',
    },
}

damage_process_approaches = {
    'FEMA P-58': {
        "1_excessive.coll.DEM": {"DS1": "collapse_DS1"},
        "2_collapse": {"DS1": "ALL_NA"},
        "3_excessiveRID": {"DS1": "irreparable_DS1"},
    },
    # TODO: expand with ground failure logic
    'Hazus Earthquake': {
        "1_STR": {"DS5": "collapse_DS1"},
        "2_LF": {"DS5": "collapse_DS1"},
        "3_excessive.coll.DEM": {"DS1": "collapse_DS1"},
        "4_collapse": {"DS1": "ALL_NA"},
        "5_excessiveRID": {"DS1": "irreparable_DS1"},
    },
    'Hazus Hurricane': {},
}

# pylint: enable=consider-using-namedtuple-or-dataclass


class Assessment:
    """
    Base class for scenario-based assessment objects.

    Assessment objects manage the models, data, and calculations in pelicun.

    """

    __slots__: list[str] = [
        'stories',
        'options',
        'unit_conversion_factors',
        'log',
        'demand',
        'asset',
        'damage',
        'loss',
    ]

    def __init__(self, config_options: dict[str, Any] | None = None):
        """
        Initializes an Assessment object.

        Parameters
        ----------
        config_options (Optional[dict]):
            User-specified configuration dictionary.

        """

        self.stories: int | None = None
        self.options = base.Options(config_options, self)
        self.unit_conversion_factors = base.parse_units(self.options.units_file)

        self.log = self.options.log
        self.log.msg(
            f'pelicun {pelicun_version} | \n',
            prepend_timestamp=False,
            prepend_blank_space=False,
        )
        self.log.print_system_info()
        self.log.div()
        self.log.msg('Assessment Started')

        self.demand: model.DemandModel = model.DemandModel(self)
        self.asset: model.AssetModel = model.AssetModel(self)
        self.damage: model.DamageModel = model.DamageModel(self)
        self.loss: model.LossModel = model.LossModel(self)

    @property
    def bldg_repair(self):
        """
        <backwards compatibility>

        Returns
        -------
        model.LossModel
            The loss model.

        """
        self.log.warn(
            '`.bldg_repair` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `.loss` instead.'
        )

        return self.loss

    @property
    def repair(self):
        """
        <backwards compatibility>

        Returns
        -------
        RepairModel_DS
            The damage state-driven component loss model.

        """
        self.log.warn(
            '`.repair` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `.loss` instead.'
        )
        return self.loss

    def get_default_data(self, data_name: str) -> pd.DataFrame:
        """
        Loads a default data file by name and returns it. This method
        is specifically designed to access predefined CSV files from a
        structured directory path related to the SimCenter fragility
        library.

        Parameters
        ----------
        data_name : str
            The name of the CSV file to be loaded, without the '.csv'
            extension. This name is used to construct the full path to
            the file.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the data loaded from the
            specified CSV file.
        """

        # <backwards compatibility>
        if 'fragility_DB' in data_name:
            data_name = data_name.replace('fragility_DB', 'damage_DB')
            self.log.warn(
                '`fragility_DB` is deprecated and will be dropped in '
                'future versions of pelicun. '
                'Please use `damage_DB` instead.'
            )
        if 'bldg_repair_DB' in data_name:
            data_name = data_name.replace('bldg_repair_DB', 'loss_repair_DB')
            self.log.warn(
                '`bldg_repair_DB` is deprecated and will be dropped in '
                'future versions of pelicun. '
                'Please use `loss_repair_DB` instead.'
            )

        data_path = f'{base.pelicun_path}/resources/SimCenterDBDL/{data_name}.csv'

        data = file_io.load_data(
            data_path, None, orientation=1, reindex=False, log=self.log
        )

        assert isinstance(data, pd.DataFrame)
        return data

    def get_default_metadata(self, data_name: str) -> dict:
        """
        Load a default metadata file and pass it to the user.

        Parameters
        ----------
        data_name: string
            Name of the json file to be loaded

        Returns
        -------
        dict
            Default metadata

        """

        # <backwards compatibility>
        if 'fragility_DB' in data_name:
            data_name = data_name.replace('fragility_DB', 'damage_DB')
            self.log.warn(
                '`fragility_DB` is deprecated and will be dropped in '
                'future versions of pelicun. Please use `damage_DB` instead.'
            )
        data_path = f'{base.pelicun_path}/resources/SimCenterDBDL/{data_name}.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def calc_unit_scale_factor(self, unit: str) -> float:
        """
        Determines the scale factor from input unit to the
        corresponding base unit

        Parameters
        ----------
        unit: str
            Either a unit name, or a quantity and a unit name
            separated by a space.
            For example: 'ft' or '100 ft'.

        Returns
        -------
        float
            Scale factor that convert values from unit to base unit

        Raises
        ------
        KeyError
            When an invalid unit is specified
        """

        unit_lst = unit.strip().split(' ')

        # check if there is a quantity specified; if yes, parse it
        if len(unit_lst) > 1:
            unit_count_str, unit_name = unit_lst
            unit_count = float(unit_count_str)

        else:
            unit_count = 1
            unit_name = unit_lst[0]

        try:
            scale_factor = unit_count * self.unit_conversion_factors[unit_name]

        except KeyError as exc:
            raise KeyError(
                f"Specified unit not recognized: {unit_count} {unit_name}"
            ) from exc

        return scale_factor

    def scale_factor(self, unit: str | None) -> float:
        """
        Returns the scale factor of a given unit. If the unit is
        unknown it raises an error. If the unit is None it returns
        1.00.

        Parameters
        ----------
        unit: str
            A unit name.

        Returns
        -------
        float
            Scale factor

        Raises
        ------
        ValueError
            If the unit is unknown.

        """

        if unit is not None:
            if unit in self.unit_conversion_factors:
                scale_factor = self.unit_conversion_factors[unit]

            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            scale_factor = 1.0

        return scale_factor


class GeneralAssessment(Assessment):
    """
    Base class for the assessment objects used in `DL_calculation.py`

    """

    __slots__: list[str] = []

    def calculate_demand(
        self,
        demand_path: str,
        collapse_limits: dict[str, float] | None,
        length_unit: str,
        demand_calibration: dict | None,
        sample_size: int,
        coupled_demands: bool,
        demand_cloning: dict | None,
        residual_drift_inference: dict | None,
    ) -> None:
        """
        Calculates demands.

        Parameters
        ----------
        demand_path : str
            Path to the demand data file.
        collapse_limits : dict[str, float] or None
            Optional dictionary with demand types and their respective
            collapse limits.
        length_unit : str
            Unit of length to be used to add units to the demand data
            if needed.
        demand_calibration : dict or None
            Calibration data for the demand model.
        sample_size : int
            Number of realizations.
        coupled_demands : bool
            Whether to preserve the raw order of the demands.
        demand_cloning : dict or None
            Demand cloning configuration.
        residual_drift_inference : dict or None
            Information for residual drift inference.

        Raises
        ------
        ValueError
            When an unknown residual drift method is specified.

        """

        # Prepare the demand sample input
        demands = self._preprocess_demand(demand_path, collapse_limits, length_unit)

        # Generate demand realizations
        self.demand.load_sample(demands)
        if demand_calibration:
            self.demand.calibrate_model(demand_calibration)
        else:
            self.demand.calibrate_model({"ALL": {"DistributionFamily": "empirical"}})
        self.demand.generate_sample(
            {
                "SampleSize": sample_size,
                'PreserveRawOrder': coupled_demands,
                'DemandCloning': demand_cloning,
            }
        )

        # Add extra information to the demand sample realizations
        self._postprocess_demand(residual_drift_inference)

    def _preprocess_demand(
        self, demand_path: str, collapse_limits: dict | None, length_unit: str
    ) -> pd.DataFrame:
        idx = pd.IndexSlice
        raw_demands = pd.read_csv(demand_path, index_col=0)

        # remove excessive demands that are considered collapses, if needed
        if collapse_limits is not None:
            raw_demands_m = base.convert_to_MultiIndex(raw_demands, axis=1)
            assert isinstance(raw_demands_m, pd.DataFrame)
            raw_demands = raw_demands_m

            if 'Units' in raw_demands.index:
                raw_units = raw_demands.loc['Units', :]
                raw_demands.drop('Units', axis=0, inplace=True)

            else:
                raw_units = None

            DEM_to_drop = np.full(raw_demands.shape[0], False)

            for DEM_type, limit in collapse_limits.items():
                assert isinstance(DEM_type, str)
                assert isinstance(limit, str | float)
                if raw_demands.columns.nlevels == 4:
                    DEM_to_drop += raw_demands.loc[
                        :,  # type: ignore
                        idx[:, DEM_type, :, :],
                    ].max(axis=1) > float(limit)

                else:
                    DEM_to_drop += raw_demands.loc[
                        :,  # type: ignore
                        idx[DEM_type, :, :],
                    ].max(axis=1) > float(limit)

            raw_demands = raw_demands.loc[~DEM_to_drop, :]

            if isinstance(raw_units, pd.Series):
                raw_demands = pd.concat(
                    [raw_demands, raw_units.to_frame().T], axis=0
                )

            self.log.msg(
                f"{np.sum(DEM_to_drop)} realizations removed from the demand "
                f"input because they exceed the collapse limit. The remaining "
                f"sample size: {raw_demands.shape[0]}"
            )

        # add units to the demand data if needed
        if "Units" not in raw_demands.index:
            demands = _add_units(raw_demands, length_unit)

        else:
            demands = raw_demands
        return demands

    def _postprocess_demand(self, residual_drift_inference: dict | None) -> None:

        idx = pd.IndexSlice

        # get the generated demand sample
        demand_sample_tuple = self.demand.save_sample(save_units=True)
        assert demand_sample_tuple is not None
        demand_sample, demand_units = demand_sample_tuple
        assert isinstance(demand_sample, pd.DataFrame)
        assert isinstance(demand_units, pd.Series)

        demand_sample = pd.concat([demand_sample, demand_units.to_frame().T])

        # get residual drift estimates, if needed
        if residual_drift_inference is not None:

            # `method` is guaranteed to exist because it is confirmed when
            # parsing the configuration file.
            rid_inference_method = residual_drift_inference.pop('method')

            if rid_inference_method == 'FEMA P-58':
                RID_list: list[pd.DataFrame] = []
                PID = demand_sample['PID'].copy()
                PID.drop('Units', inplace=True)
                PID = PID.astype(float)

                for direction, delta_yield in residual_drift_inference.items():

                    pids = PID.loc[:, idx[:, direction]]  # type: ignore
                    assert isinstance(pids, pd.DataFrame)
                    RID = self.demand.estimate_RID(
                        pids,
                        {'yield_drift': float(delta_yield)},
                    )

                    RID_list.append(RID)

                RID = pd.concat(RID_list, axis=1)
                RID_units = pd.Series(
                    ['unitless'] * RID.shape[1],
                    index=RID.columns,
                    name='Units',
                )
                RID_sample = pd.concat([RID, RID_units.to_frame().T])
                demand_sample = pd.concat([demand_sample, RID_sample], axis=1)

            else:

                raise ValueError(
                    f'Unknown residual drift inference method: '
                    f'`{rid_inference_method}`.'
                )

        # add a constant one demand
        demand_sample[('ONE', '0', '1')] = np.ones(demand_sample.shape[0])
        demand_sample.loc['Units', ('ONE', '0', '1')] = 'unitless'

        self.demand.load_sample(base.convert_to_SimpleIndex(demand_sample, axis=1))

    def calculate_asset(
        self,
        num_stories: int,
        component_assignment_file: str | None,
        collapse_fragility_demand_type: str | None,
        add_irreparable_damage_columns: bool,
        component_sample_file: str | None,
    ):
        """
        Generates the asset model sample.

        Parameters
        ----------
        num_stories : int
            Number of stories.
        component_assignment_file : str or None
            Path to a component assignment file.
        collapse_fragility_demand_type : str or None
            Optional demand type for the collapse fragility.
        add_irreparable_damage_columns : bool
            Whether to add columns for irreparable damage.
        component_sample_file : str or None
            Optional path to an existing component sample file.

        Raises
        ------
        ValueError
            With invalid combinations of arguments.

        """

        # retrieve the demand sample
        demand_sample = self.demand.save_sample()
        assert isinstance(demand_sample, pd.DataFrame)

        # set the number of stories
        if num_stories:
            self.stories = num_stories

        # We either accept a `component_assignment_file` or a
        # `component_sample_file`, not both.
        if (
            component_assignment_file is not None
            and component_sample_file is not None
        ):
            raise ValueError(
                'Both `component_assignment_file` and '
                '`component_sample_file` are provided. '
                'Please provide only one.'
            )

        # load a component model and generate a sample
        if component_assignment_file is not None:
            cmp_marginals = pd.read_csv(
                component_assignment_file,
                index_col=0,
                encoding_errors='replace',
            )

            self._add_extra_cmp_marginals(
                cmp_marginals,
                demand_sample,
                collapse_fragility_demand_type,
                add_irreparable_damage_columns,
            )

            # load component model and generate sample
            self.asset.load_cmp_model({'marginals': cmp_marginals})
            self.asset.generate_cmp_sample()

        # alternatively, load an existing quantity sample from a file
        if component_sample_file is not None:
            self.asset.load_cmp_sample(component_sample_file)

    def _add_extra_cmp_marginals(
        self,
        cmp_marginals: pd.DataFrame,
        demand_sample: pd.DataFrame,
        collapse_fragility_demand_type: str | None,
        add_irreparable_damage_columns: bool,
    ) -> None:
        DEM_types = demand_sample.columns.unique(level=0)

        # add component(s) to support collapse calculation
        if collapse_fragility_demand_type is not None:
            if not collapse_fragility_demand_type.startswith('SA'):
                # we need story-specific collapse assessment
                # (otherwise we have a global demand and evaluate
                # collapse directly, so this code should be skipped)

                if collapse_fragility_demand_type in DEM_types:
                    # excessive coll_DEM is added on every floor
                    # to detect large RIDs
                    cmp_marginals.loc['excessive.coll.DEM', 'Units'] = 'ea'

                    locs = demand_sample[
                        collapse_fragility_demand_type  # type: ignore
                    ].columns.unique(level=0)
                    cmp_marginals.loc['excessive.coll.DEM', 'Location'] = ','.join(
                        locs
                    )

                    dirs = demand_sample[
                        collapse_fragility_demand_type  # type: ignore
                    ].columns.unique(level=1)
                    cmp_marginals.loc['excessive.coll.DEM', 'Direction'] = ','.join(
                        dirs
                    )

                    cmp_marginals.loc['excessive.coll.DEM', 'Theta_0'] = 1.0

                else:

                    self.log.warn(
                        f'No {collapse_fragility_demand_type} '
                        f'among available demands. Collapse cannot '
                        f'be evaluated.'
                    )

        # always add a component to support basic collapse calculation
        cmp_marginals.loc['collapse', 'Units'] = 'ea'
        cmp_marginals.loc['collapse', 'Location'] = 0
        cmp_marginals.loc['collapse', 'Direction'] = 1
        cmp_marginals.loc['collapse', 'Theta_0'] = 1.0

        # add components to support irreparable damage calculation
        if add_irreparable_damage_columns:
            if 'RID' in DEM_types:
                # excessive RID is added on every floor to detect large RIDs
                cmp_marginals.loc['excessiveRID', 'Units'] = 'ea'

                locs = demand_sample['RID'].columns.unique(level=0)
                cmp_marginals.loc['excessiveRID', 'Location'] = ','.join(locs)

                dirs = demand_sample['RID'].columns.unique(level=1)
                cmp_marginals.loc['excessiveRID', 'Direction'] = ','.join(dirs)

                cmp_marginals.loc['excessiveRID', 'Theta_0'] = 1.0

                # irreparable is a global component to recognize is any of the
                # excessive RIDs were triggered
                cmp_marginals.loc['irreparable', 'Units'] = 'ea'
                cmp_marginals.loc['irreparable', 'Location'] = 0
                cmp_marginals.loc['irreparable', 'Direction'] = 1
                cmp_marginals.loc['irreparable', 'Theta_0'] = 1.0

            else:
                self.log.warn(
                    'No residual interstory drift ratio among '
                    'available demands. Irreparable damage cannot be '
                    'evaluated.'
                )

    def calculate_damage(
        self,
        length_unit: str,
        component_database: str,
        component_database_path: str | None = None,
        collapse_fragility: dict | None = None,
        irreparable_damage: dict | None = None,
        damage_process_approach: str | None = None,
        damage_process_file_path: str | None = None,
        custom_model_dir: str | None = None,
    ) -> None:
        """
        Calculates damage.

        Parameters
        ----------
        length_unit : str
            Unit of length to be used to add units to the demand data
            if needed.
        component_database : str
            Name of the component database.
        component_database_path : str or None
            Optional path to a component database file.
        collapse_fragility : dict or None
            Collapse fragility information.
        irreparable_damage : dict or None
            Information for irreparable damage.
        damage_process_approach : str or None
            Approach for the damage process.
        damage_process_file_path : str or None
            Optional path to a damage process file.
        custom_model_dir : str or None
            Optional directory for custom models.

        Raises
        ------
        ValueError
            With invalid combinations of arguments.

        """

        component_db = self._get_component_db(
            component_database, component_database_path, custom_model_dir
        )

        # prepare additional fragility data
        additional_fragility_df = self._additional_fragility_data(
            collapse_fragility, length_unit, irreparable_damage
        )

        self.damage.load_model_parameters(
            component_db + [additional_fragility_df],
            set(self.asset.list_unique_component_ids()),
        )

        # load the damage process if needed
        if damage_process_approach is not None:
            dmg_process = self._get_dmg_process(
                damage_process_approach, damage_process_file_path
            )
        else:
            dmg_process = None

        # calculate damages
        self.damage.calculate(dmg_process=dmg_process)

    def _get_component_db(
        self,
        component_database: str,
        component_database_path: str | None,
        custom_model_dir: str | None,
    ) -> list[str]:
        # load the fragility information
        if component_database in default_DBs['fragility']:
            component_db = [
                'PelicunDefault/' + default_DBs['fragility'][component_database],
            ]
        else:
            component_db = []

        if component_database_path is not None:

            if custom_model_dir is None:
                raise ValueError(
                    '`custom_model_dir` needs to be specified '
                    'when `component_database_path` is not None.'
                )

            if 'CustomDLDataFolder' in component_database_path:
                component_database_path = component_database_path.replace(
                    'CustomDLDataFolder', custom_model_dir
                )

            component_db += [component_database_path]

        component_db = component_db[::-1]
        return component_db

    def _get_dmg_process(
        self, damage_process_approach: str, damage_process_file_path: str | None
    ) -> dict | None:
        if damage_process_approach in damage_process_approaches:
            dmg_process = damage_process_approaches[damage_process_approach]

            # For Hazus Earthquake, we need to specify the component ids
            if damage_process_approach == 'Hazus Earthquake':

                cmp_sample = self.asset.save_cmp_sample()
                assert isinstance(cmp_sample, pd.DataFrame)

                cmp_list = cmp_sample.columns.unique(level=0)

                cmp_map = {'STR': '', 'LF': '', 'NSA': ''}

                for cmp in cmp_list:
                    for cmp_type in cmp_map:
                        if cmp_type + '.' in cmp:
                            cmp_map[cmp_type] = cmp

                new_dmg_process = dmg_process.copy()
                for source_cmp, action in dmg_process.items():
                    # first, look at the source component id
                    new_source = None
                    for cmp_type, cmp_id in cmp_map.items():
                        if (cmp_type in source_cmp) and (cmp_id != ''):
                            new_source = source_cmp.replace(cmp_type, cmp_id)
                            break

                    if new_source is not None:
                        new_dmg_process[new_source] = action
                        del new_dmg_process[source_cmp]
                    else:
                        new_source = source_cmp

                    # then, look at the target component ids
                    for ds_i, target_vals in action.items():
                        if isinstance(target_vals, str):
                            for cmp_type, cmp_id in cmp_map.items():
                                if (cmp_type in target_vals) and (cmp_id != ''):
                                    target_vals = target_vals.replace(
                                        cmp_type, cmp_id
                                    )

                            new_target_vals = target_vals

                        else:
                            # we assume that target_vals is a list of str
                            new_target_vals = []

                            for target_val in target_vals:
                                for cmp_type, cmp_id in cmp_map.items():
                                    if (cmp_type in target_val) and (cmp_id != ''):
                                        target_val = target_val.replace(
                                            cmp_type, cmp_id
                                        )

                                new_target_vals.append(target_val)

                        new_dmg_process[new_source][ds_i] = new_target_vals

                dmg_process = new_dmg_process

        elif damage_process_approach == "User Defined":

            if damage_process_file_path is None:
                raise ValueError(
                    'When `damage_process_approach` is set to '
                    '`User Defined`, a `damage_process_file_path` '
                    'needs to be provided.'
                )

            # load the damage process from a file
            with open(damage_process_file_path, 'r', encoding='utf-8') as f:
                dmg_process = json.load(f)

        elif damage_process_approach == "None":
            # no damage process applied for the calculation
            dmg_process = None

        else:
            raise ValueError(
                f"Unknown damage process approach: " f"`{damage_process_approach}`."
            )
        return dmg_process

    def _additional_fragility_data(
        self,
        collapse_fragility: dict | None,
        length_unit: str,
        irreparable_damage: dict | None,
    ) -> pd.DataFrame:
        columns = pd.MultiIndex.from_tuples(
            [
                ('Demand', 'Directional'),
                ('Demand', 'Offset'),
                ('Demand', 'Type'),
                ('Demand', 'Unit'),
                ('Incomplete', ''),
                ('LS1', 'DamageStateWeights'),
                ('LS1', 'Family'),
                ('LS1', 'Theta_0'),
                ('LS1', 'Theta_1'),
                ('LS2', 'DamageStateWeights'),
                ('LS2', 'Family'),
                ('LS2', 'Theta_0'),
                ('LS2', 'Theta_1'),
                ('LS3', 'DamageStateWeights'),
                ('LS3', 'Family'),
                ('LS3', 'Theta_0'),
                ('LS3', 'Theta_1'),
                ('LS4', 'DamageStateWeights'),
                ('LS4', 'Family'),
                ('LS4', 'Theta_0'),
                ('LS4', 'Theta_1'),
            ],
        )
        adf = pd.DataFrame(columns=columns)

        if collapse_fragility:

            assert self.asset.cmp_marginal_params is not None

            if (
                'excessive.coll.DEM'
                in self.asset.cmp_marginal_params.index.get_level_values('cmp')
            ):
                # if there is story-specific evaluation
                coll_CMP_name = 'excessive.coll.DEM'
            else:
                # otherwise, for global collapse evaluation
                coll_CMP_name = 'collapse'

            adf.loc[coll_CMP_name, ('Demand', 'Directional')] = 1
            adf.loc[coll_CMP_name, ('Demand', 'Offset')] = 0

            coll_DEM = collapse_fragility['DemandType']

            if '_' in coll_DEM:
                coll_DEM, coll_DEM_spec = coll_DEM.split('_')
            else:
                coll_DEM_spec = None

            coll_DEM_name = None
            for demand_name, demand_short in EDP_to_demand_type.items():
                if demand_short == coll_DEM:
                    coll_DEM_name = demand_name
                    break

            if coll_DEM_name is None:
                raise ValueError('`coll_DEM_name` cannot be None.')

            if coll_DEM_spec is None:
                adf.loc[coll_CMP_name, ('Demand', 'Type')] = coll_DEM_name

            else:
                adf.loc[coll_CMP_name, ('Demand', 'Type')] = (
                    f'{coll_DEM_name}|{coll_DEM_spec}'
                )

            coll_DEM_unit = _add_units(
                pd.DataFrame(
                    columns=[
                        f'{coll_DEM}-1-1',
                    ]
                ),
                length_unit,
            ).iloc[0, 0]

            adf.loc[coll_CMP_name, ('Demand', 'Unit')] = coll_DEM_unit
            adf.loc[coll_CMP_name, ('LS1', 'Family')] = collapse_fragility[
                'CapacityDistribution'
            ]
            adf.loc[coll_CMP_name, ('LS1', 'Theta_0')] = collapse_fragility[
                'CapacityMedian'
            ]
            adf.loc[coll_CMP_name, ('LS1', 'Theta_1')] = collapse_fragility[
                'Theta_1'
            ]
            adf.loc[coll_CMP_name, 'Incomplete'] = 0

            if coll_CMP_name != 'collapse':
                # for story-specific evaluation, we need to add a placeholder
                # fragility that will never trigger, but helps us aggregate
                # results in the end
                adf.loc['collapse', ('Demand', 'Directional')] = 1
                adf.loc['collapse', ('Demand', 'Offset')] = 0
                adf.loc['collapse', ('Demand', 'Type')] = 'One'
                adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
                adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
                adf.loc['collapse', 'Incomplete'] = 0

        else:

            # add a placeholder collapse fragility that will never trigger
            # collapse, but allow damage processes to work with collapse

            adf.loc['collapse', ('Demand', 'Directional')] = 1
            adf.loc['collapse', ('Demand', 'Offset')] = 0
            adf.loc['collapse', ('Demand', 'Type')] = 'One'
            adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
            adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
            adf.loc['collapse', 'Incomplete'] = 0

        if irreparable_damage:

            # add excessive RID fragility according to settings provided in the
            # input file
            adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
            adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
            adf.loc['excessiveRID', ('Demand', 'Type')] = (
                'Residual Interstory Drift Ratio'
            )

            adf.loc['excessiveRID', ('Demand', 'Unit')] = 'unitless'
            adf.loc['excessiveRID', ('LS1', 'Theta_0')] = irreparable_damage[
                'DriftCapacityMedian'
            ]
            adf.loc['excessiveRID', ('LS1', 'Family')] = "lognormal"
            adf.loc['excessiveRID', ('LS1', 'Theta_1')] = irreparable_damage[
                'DriftCapacityLogStd'
            ]

            adf.loc['excessiveRID', 'Incomplete'] = 0

            # add a placeholder irreparable fragility that will never trigger
            # damage, but allow damage processes to aggregate excessiveRID here
            adf.loc['irreparable', ('Demand', 'Directional')] = 1
            adf.loc['irreparable', ('Demand', 'Offset')] = 0
            adf.loc['irreparable', ('Demand', 'Type')] = 'One'
            adf.loc['irreparable', ('Demand', 'Unit')] = 'unitless'
            adf.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
            adf.loc['irreparable', 'Incomplete'] = 0

        return adf

    def calculate_loss(
        self,
        loss_map_approach: str,
        occupancy_type: str,
        consequence_database: str,
        consequence_database_path: str | None = None,
        custom_model_dir: str | None = None,
        damage_process_approach: str = 'User Defined',
        replacement_cost_parameters: dict[str, float | str] | None = None,
        replacement_time_parameters: dict[str, float | str] | None = None,
        replacement_carbon_parameters: dict[str, float | str] | None = None,
        replacement_energy_parameters: dict[str, float | str] | None = None,
        loss_map_path: str | None = None,
        decision_variables: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates losses.

        Parameters
        ----------
        loss_map_approach : str
            Approach for the loss map generation. Can be either
            `User Defined` or `Automatic`.
        occupancy_type : str
            Occupancy type.
        consequence_database : str
            Name of the consequence database.
        consequence_database_path : str or None
            Optional path to a consequence database file.
        custom_model_dir : str or None
            Optional directory for custom models.
        damage_process_approach : str
            Damage process approach. Defaults to `User Defined`.
        replacement_cost_parameters : dict or None
            Parameters for replacement cost.
        replacement_time_parameters : dict or None
            Parameters for replacement time.
        replacement_carbon_parameters : dict or None
            Parameters for replacement carbon.
        replacement_energy_parameters : dict or None
            Parameters for replacement energy.
        loss_map_path : str or None
            Optional path to a loss map file.
        decision_variables : list[str] or None
            Optional decision variables for the assessment.

        Returns
        -------
        tuple
            Dataframe with the aggregated loss of each realization,
            and another boolean dataframe with information on which DV
            thresholds were exceeded in each realization, triggering
            replacement. If no thresholds are specified it only
            contains False values.

        Raises
        ------
        ValueError
            When an invalid loss map approach is specified.

        """

        conseq_df, consequence_db = load_consequence_info(
            self,
            consequence_database,
            consequence_database_path,
            custom_model_dir,
        )

        # remove duplicates from conseq_df
        conseq_df = conseq_df.loc[conseq_df.index.unique(), :]

        # add the replacement consequence to the data
        adf = pd.DataFrame(
            columns=conseq_df.columns,
            index=pd.MultiIndex.from_tuples(
                [
                    ('replacement', 'Cost'),
                    ('replacement', 'Time'),
                    ('replacement', 'Carbon'),
                    ('replacement', 'Energy'),
                ]
            ),
        )

        _loss__add_replacement_cost(
            adf,
            damage_process_approach,
            unit=get(replacement_cost_parameters, 'Unit'),
            median=get(replacement_cost_parameters, 'Median'),
            distribution=get(replacement_cost_parameters, 'Distribution'),
            theta_1=get(replacement_cost_parameters, 'Theta_1'),
        )

        _loss__add_replacement_time(
            adf,
            damage_process_approach,
            conseq_df,
            occupancy_type=occupancy_type,
            unit=get(replacement_time_parameters, 'Unit'),
            median=get(replacement_time_parameters, 'Median'),
            distribution=get(replacement_time_parameters, 'Distribution'),
            theta_1=get(replacement_time_parameters, 'Theta_1'),
        )

        _loss__add_replacement_carbon(
            adf,
            damage_process_approach,
            unit=get(replacement_carbon_parameters, 'Unit'),
            median=get(replacement_carbon_parameters, 'Median'),
            distribution=get(replacement_carbon_parameters, 'Distribution'),
            theta_1=get(replacement_carbon_parameters, 'Theta_1'),
        )

        _loss__add_replacement_energy(
            adf,
            damage_process_approach,
            unit=get(replacement_energy_parameters, 'Unit'),
            median=get(replacement_energy_parameters, 'Median'),
            distribution=get(replacement_energy_parameters, 'Distribution'),
            theta_1=get(replacement_energy_parameters, 'Theta_1'),
        )

        # prepare the loss map
        loss_map = None
        if loss_map_approach == "Automatic":
            # get the damage sample
            loss_map = _loss__map_auto(
                self, conseq_df, damage_process_approach, occupancy_type
            )

        elif loss_map_approach == "User Defined":
            loss_map = _loss__map_user(custom_model_dir, loss_map_path)

        else:
            raise ValueError(f'Invalid MapApproach value: `{loss_map_approach}`.')

        # prepare additional loss map entries, if needed
        if 'DMG-collapse' not in loss_map.index:
            loss_map.loc['DMG-collapse', 'Repair'] = 'replacement'
            loss_map.loc['DMG-irreparable', 'Repair'] = 'replacement'

        if decision_variables:
            self.loss.decision_variables = decision_variables

        self.loss.add_loss_map(loss_map, loss_map_policy=None)
        self.loss.load_model_parameters(consequence_db + [adf])

        self.loss.calculate()

        df_agg, exceedance_bool_df = self.loss.aggregate_losses(future=True)
        assert isinstance(df_agg, pd.DataFrame)
        assert isinstance(exceedance_bool_df, pd.DataFrame)
        return df_agg, exceedance_bool_df


class WaterNetworkAssessment(GeneralAssessment):
    """
    Water network assessment class.

    """

    __slots__: list[str] = []

    def _additional_fragility_data(
        self,
        collapse_fragility: dict | None,
        length_unit: str,
        irreparable_damage: dict | None,
    ) -> pd.DataFrame:
        columns = pd.MultiIndex.from_tuples(
            [
                ('Demand', 'Directional'),
                ('Demand', 'Offset'),
                ('Demand', 'Type'),
                ('Demand', 'Unit'),
                ('Incomplete', ''),
                ('LS1', 'DamageStateWeights'),
                ('LS1', 'Family'),
                ('LS1', 'Theta_0'),
                ('LS1', 'Theta_1'),
                ('LS2', 'DamageStateWeights'),
                ('LS2', 'Family'),
                ('LS2', 'Theta_0'),
                ('LS2', 'Theta_1'),
                ('LS3', 'DamageStateWeights'),
                ('LS3', 'Family'),
                ('LS3', 'Theta_0'),
                ('LS3', 'Theta_1'),
                ('LS4', 'DamageStateWeights'),
                ('LS4', 'Family'),
                ('LS4', 'Theta_0'),
                ('LS4', 'Theta_1'),
            ],
        )
        adf = pd.DataFrame(columns=columns)

        if collapse_fragility:

            assert self.asset.cmp_marginal_params is not None

            if (
                'excessive.coll.DEM'
                in self.asset.cmp_marginal_params.index.get_level_values('cmp')
            ):
                # if there is story-specific evaluation
                coll_CMP_name = 'excessive.coll.DEM'
            else:
                # otherwise, for global collapse evaluation
                coll_CMP_name = 'collapse'

            adf.loc[coll_CMP_name, ('Demand', 'Directional')] = 1
            adf.loc[coll_CMP_name, ('Demand', 'Offset')] = 0

            coll_DEM = collapse_fragility['DemandType']

            if '_' in coll_DEM:
                coll_DEM, coll_DEM_spec = coll_DEM.split('_')
            else:
                coll_DEM_spec = None

            coll_DEM_name = None
            for demand_name, demand_short in EDP_to_demand_type.items():
                if demand_short == coll_DEM:
                    coll_DEM_name = demand_name
                    break

            if coll_DEM_name is None:
                raise ValueError('`coll_DEM_name` cannot be None.')

            if coll_DEM_spec is None:
                adf.loc[coll_CMP_name, ('Demand', 'Type')] = coll_DEM_name

            else:
                adf.loc[coll_CMP_name, ('Demand', 'Type')] = (
                    f'{coll_DEM_name}|{coll_DEM_spec}'
                )

            coll_DEM_unit = _add_units(
                pd.DataFrame(
                    columns=[
                        f'{coll_DEM}-1-1',
                    ]
                ),
                length_unit,
            ).iloc[0, 0]

            adf.loc[coll_CMP_name, ('Demand', 'Unit')] = coll_DEM_unit
            adf.loc[coll_CMP_name, ('LS1', 'Family')] = collapse_fragility[
                'CapacityDistribution'
            ]
            adf.loc[coll_CMP_name, ('LS1', 'Theta_0')] = collapse_fragility[
                'CapacityMedian'
            ]
            adf.loc[coll_CMP_name, ('LS1', 'Theta_1')] = collapse_fragility[
                'Theta_1'
            ]
            adf.loc[coll_CMP_name, 'Incomplete'] = 0

            if coll_CMP_name != 'collapse':
                # for story-specific evaluation, we need to add a placeholder
                # fragility that will never trigger, but helps us aggregate
                # results in the end
                adf.loc['collapse', ('Demand', 'Directional')] = 1
                adf.loc['collapse', ('Demand', 'Offset')] = 0
                adf.loc['collapse', ('Demand', 'Type')] = 'One'
                adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
                adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
                adf.loc['collapse', 'Incomplete'] = 0

        if irreparable_damage:

            # add excessive RID fragility according to settings provided in the
            # input file
            adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
            adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
            adf.loc['excessiveRID', ('Demand', 'Type')] = (
                'Residual Interstory Drift Ratio'
            )

            adf.loc['excessiveRID', ('Demand', 'Unit')] = 'unitless'
            adf.loc['excessiveRID', ('LS1', 'Theta_0')] = irreparable_damage[
                'DriftCapacityMedian'
            ]
            adf.loc['excessiveRID', ('LS1', 'Family')] = "lognormal"
            adf.loc['excessiveRID', ('LS1', 'Theta_1')] = irreparable_damage[
                'DriftCapacityLogStd'
            ]

            adf.loc['excessiveRID', 'Incomplete'] = 0

            # add a placeholder irreparable fragility that will never trigger
            # damage, but allow damage processes to aggregate excessiveRID here
            adf.loc['irreparable', ('Demand', 'Directional')] = 1
            adf.loc['irreparable', ('Demand', 'Offset')] = 0
            adf.loc['irreparable', ('Demand', 'Type')] = 'One'
            adf.loc['irreparable', ('Demand', 'Unit')] = 'unitless'
            adf.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
            adf.loc['irreparable', 'Incomplete'] = 0

        # add a placeholder aggregate fragility that will never trigger
        # damage, but allow damage processes to aggregate the
        # various pipeline damages
        adf.loc['aggregate', ('Demand', 'Directional')] = 1
        adf.loc['aggregate', ('Demand', 'Offset')] = 0
        adf.loc['aggregate', ('Demand', 'Type')] = 'Peak Ground Velocity'
        adf.loc['aggregate', ('Demand', 'Unit')] = 'mps'
        adf.loc['aggregate', ('LS1', 'Theta_0')] = 1e10
        adf.loc['aggregate', ('LS2', 'Theta_0')] = 1e10
        adf.loc['aggregate', 'Incomplete'] = 0
        return adf


class TimeBasedAssessment:
    """
    Implements a time-based assessment.

    A time-based assessment evaluates the performance of an asset
    considering all potential instances of a type of hazard that could
    occur over a specified period of time, producing annualized loss
    performance metrics.

    """

    def __init__(self, assessments: list[Assessment], scenario_info: pd.DataFrame):
        """
        Initializes a time-based assessment object.

        Parameters
        ----------


        """


def load_consequence_info(
    self,
    consequence_database: str,
    consequence_database_path: str | None = None,
    custom_model_dir: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load consequence information for the assessment.

    Parameters
    ----------
    consequence_database : str
        Name of the consequence database.
    consequence_database_path : str or None
        Optional path to a consequence database file.
    custom_model_dir : str or None
        Optional directory for custom models.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A tuple containing:
        - A DataFrame with the consequence data.
        - A list of paths to the consequence databases used.

    Raises
    ------
    ValueError
        With invalid combinations of arguments.

    """
    if consequence_database in default_DBs['repair']:
        consequence_db = [
            'PelicunDefault/' + default_DBs['repair'][consequence_database],
        ]

        conseq_df = self.get_default_data(
            default_DBs['repair'][consequence_database][:-4]
        )
    else:
        consequence_db = []

        conseq_df = pd.DataFrame()

    if consequence_database_path is not None:

        if custom_model_dir is None:
            raise ValueError(
                'When `consequence_database_path` is specified, '
                '`custom_model_dir` needs to be specified as well.'
            )

        if 'CustomDLDataFolder' in consequence_database_path:
            consequence_database_path = consequence_database_path.replace(
                'CustomDLDataFolder', custom_model_dir
            )

        consequence_db += [consequence_database_path]

        extra_conseq_df = file_io.load_data(
            consequence_database_path,
            unit_conversion_factors=None,
            orientation=1,
            reindex=False,
        )
        assert isinstance(extra_conseq_df, pd.DataFrame)

        if isinstance(conseq_df, pd.DataFrame):
            conseq_df = pd.concat([conseq_df, extra_conseq_df])
        else:
            conseq_df = extra_conseq_df

    consequence_db = consequence_db[::-1]

    return conseq_df, consequence_db


def _add_units(raw_demands: pd.DataFrame, length_unit: str) -> pd.DataFrame:
    """
    Add units to demand columns in a DataFrame.

    Parameters
    ----------
    raw_demands : pd.DataFrame
        The raw demand data to which units will be added.
    length_unit : str
        The unit of length to be used (e.g., 'in' for inches).

    Returns
    -------
    pd.DataFrame
        The DataFrame with units added to the appropriate demand columns.

    """
    demands = raw_demands.T

    demands.insert(0, "Units", np.nan)

    if length_unit == 'in':
        length_unit = 'inch'

    demands_mi = base.convert_to_MultiIndex(demands, axis=0).sort_index(axis=0).T
    assert isinstance(demands_mi, pd.DataFrame)
    demands = demands_mi

    if demands.columns.nlevels == 4:
        DEM_level = 1
    else:
        DEM_level = 0

    # drop demands with no EDP type identified
    demands.drop(
        demands.columns[demands.columns.get_level_values(DEM_level) == ''],
        axis=1,
        inplace=True,
    )

    # assign units
    demand_cols = demands.columns.get_level_values(DEM_level)

    # remove additional info from demand names
    demand_cols = pd.Index([d.split('_')[0] for d in demand_cols])

    # acceleration
    acc_EDPs = ['PFA', 'PGA', 'SA']
    EDP_mask = np.isin(demand_cols, acc_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = length_unit + 'ps2'  # type: ignore

    # speed
    speed_EDPs = ['PFV', 'PWS', 'PGV', 'SV']
    EDP_mask = np.isin(demand_cols, speed_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = length_unit + 'ps'  # type: ignore

    # displacement
    disp_EDPs = ['PFD', 'PIH', 'SD', 'PGD']
    EDP_mask = np.isin(demand_cols, disp_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = length_unit  # type: ignore

    # drift ratio
    rot_EDPs = ['PID', 'PRD', 'DWD', 'RDR', 'PMD', 'RID']
    EDP_mask = np.isin(demand_cols, rot_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = 'unitless'  # type: ignore

    # convert back to simple header and return the DF
    return base.convert_to_SimpleIndex(demands, axis=1)


def _loss__add_replacement_energy(
    adf: pd.DataFrame,
    damage_process_approach: str,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
) -> None:
    ren = ('replacement', 'Energy')
    if median is not None:

        # TODO: in this case we need unit (add config parser check)

        adf.loc[ren, ('Quantity', 'Unit')] = "1 EA"
        adf.loc[ren, ('DV', 'Unit')] = unit
        adf.loc[ren, ('DS1', 'Theta_0')] = median

        if distribution is not None:

            # TODO: in this case we need theta_1 (add config parser check)

            adf.loc[ren, ('DS1', 'Family')] = distribution
            adf.loc[ren, ('DS1', 'Theta_1')] = theta_1
    else:
        # add a default replacement energy value as a placeholder
        # the default value depends on the consequence database

        # for FEMA P-58, use 0 kg
        if damage_process_approach == 'FEMA P-58':
            adf.loc[ren, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[ren, ('DV', 'Unit')] = 'MJ'
            adf.loc[ren, ('DS1', 'Theta_0')] = 0

        else:
            # for everything else, remove this consequence
            adf.drop(ren, inplace=True)


def _loss__add_replacement_carbon(
    adf: pd.DataFrame,
    damage_process_approach: str,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
) -> None:
    rcarb = ('replacement', 'Carbon')
    if median is not None:

        # TODO: in this case we need unit (add config parser check)

        adf.loc[rcarb, ('Quantity', 'Unit')] = "1 EA"
        adf.loc[rcarb, ('DV', 'Unit')] = unit
        adf.loc[rcarb, ('DS1', 'Theta_0')] = median

        if distribution is not None:

            # TODO: in this case we need theta_1 (add config parser check)

            adf.loc[rcarb, ('DS1', 'Family')] = distribution
            adf.loc[rcarb, ('DS1', 'Theta_1')] = theta_1
    else:

        # add a default replacement carbon value as a placeholder
        # the default value depends on the consequence database

        # for FEMA P-58, use 0 kg
        if damage_process_approach == 'FEMA P-58':
            adf.loc[rcarb, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rcarb, ('DV', 'Unit')] = 'kg'
            adf.loc[rcarb, ('DS1', 'Theta_0')] = 0

        else:
            # for everything else, remove this consequence
            adf.drop(rcarb, inplace=True)


def _loss__add_replacement_time(
    adf: pd.DataFrame,
    damage_process_approach: str,
    conseq_df: pd.DataFrame,
    occupancy_type: str | None = None,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
):
    rt = ('replacement', 'Time')
    if median is not None:

        # TODO: in this case we need unit (add config parser check)

        adf.loc[rt, ('Quantity', 'Unit')] = "1 EA"
        adf.loc[rt, ('DV', 'Unit')] = unit
        adf.loc[rt, ('DS1', 'Theta_0')] = median

        if distribution is not None:

            # TODO: in this case we need theta_1 (add config parser check)

            adf.loc[rt, ('DS1', 'Family')] = distribution
            adf.loc[rt, ('DS1', 'Theta_1')] = theta_1
    else:

        # add a default replacement time value as a placeholder
        # the default value depends on the consequence database

        # for FEMA P-58, use 0 worker_days
        if damage_process_approach == 'FEMA P-58':
            adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rt, ('DV', 'Unit')] = 'worker_day'
            adf.loc[rt, ('DS1', 'Theta_0')] = 0

        # for Hazus EQ, use 1.0 as a loss_ratio
        elif damage_process_approach == 'Hazus Earthquake - Buildings':
            adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rt, ('DV', 'Unit')] = 'day'

            # load the replacement time that corresponds to total loss
            adf.loc[rt, ('DS1', 'Theta_0')] = conseq_df.loc[
                (f"STR.{occupancy_type}", 'Time'), ('DS5', 'Theta_0')
            ]

        # otherwise, use 1 (and expect to have it defined by the user)
        else:
            adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rt, ('DV', 'Unit')] = 'loss_ratio'
            adf.loc[rt, ('DS1', 'Theta_0')] = 1


def _loss__add_replacement_cost(
    adf: pd.DataFrame,
    damage_process_approach: str,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
):

    rc = ('replacement', 'Cost')
    if median is not None:

        # TODO: in this case we need unit (add config parser check)

        adf.loc[rc, ('Quantity', 'Unit')] = "1 EA"
        adf.loc[rc, ('DV', 'Unit')] = median
        adf.loc[rc, ('DS1', 'Theta_0')] = unit

        if distribution is not None:

            # TODO: in this case we need theta_1 (add config parser check)

            adf.loc[rc, ('DS1', 'Family')] = distribution
            adf.loc[rc, ('DS1', 'Theta_1')] = theta_1

    else:

        # add a default replacement cost value as a placeholder
        # the default value depends on the consequence database

        # for FEMA P-58, use 0 USD
        if damage_process_approach == 'FEMA P-58':
            adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rc, ('DV', 'Unit')] = 'USD_2011'
            adf.loc[rc, ('DS1', 'Theta_0')] = 0

        # for Hazus EQ and HU, use 1.0 as a loss_ratio
        elif damage_process_approach in {'Hazus Earthquake', 'Hazus Hurricane'}:
            adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'

            # store the replacement cost that corresponds to total loss
            adf.loc[rc, ('DS1', 'Theta_0')] = 1.00

        # otherwise, use 1 (and expect to have it defined by the user)
        else:
            adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
            adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'
            adf.loc[rc, ('DS1', 'Theta_0')] = 1


def _loss__map_user(
    custom_model_dir: str | None, loss_map_path: str | None = None
) -> pd.DataFrame:
    if loss_map_path is not None:

        assert custom_model_dir is not None
        loss_map_path = loss_map_path.replace('CustomDLDataFolder', custom_model_dir)

    else:
        raise ValueError('Missing loss map path.')

    loss_map = pd.read_csv(loss_map_path, index_col=0)

    return loss_map


def _loss__map_auto(
    assessment: Assessment,
    conseq_df: pd.DataFrame,
    damage_process_approach: str,
    occupancy_type: str | None = None,
) -> pd.DataFrame:
    # get the damage sample
    dmg_sample = assessment.damage.save_sample()

    # create a mapping for all components that are also in
    # the prescribed consequence database
    dmg_cmps = dmg_sample.columns.unique(level='cmp')  # type: ignore
    loss_cmps = conseq_df.index.unique(level=0)

    drivers = []
    loss_models = []

    if damage_process_approach in {'FEMA P-58', 'Hazus Hurricane'}:
        # with these methods, we assume fragility and consequence data
        # have the same IDs

        for dmg_cmp in dmg_cmps:
            if dmg_cmp == 'collapse':
                continue

            if dmg_cmp in loss_cmps:
                drivers.append(f'DMG-{dmg_cmp}')
                loss_models.append(dmg_cmp)

    elif damage_process_approach in {
        'Hazus Earthquake',
        'Hazus Earthquake Transportation',
    }:
        # with Hazus Earthquake we assume that consequence
        # archetypes are only differentiated by occupancy type
        for dmg_cmp in dmg_cmps:
            if dmg_cmp == 'collapse':
                continue

            cmp_class = dmg_cmp.split('.')[0]
            if occupancy_type is not None:
                loss_cmp = f'{cmp_class}.{occupancy_type}'
            else:
                loss_cmp = cmp_class

            if loss_cmp in loss_cmps:
                drivers.append(f'DMG-{dmg_cmp}')
                loss_models.append(loss_cmp)

    loss_map = pd.DataFrame(loss_models, columns=['Repair'], index=drivers)

    return loss_map
