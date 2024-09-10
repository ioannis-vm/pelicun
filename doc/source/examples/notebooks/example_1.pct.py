# %% [markdown]
# # Example 1

# %% [markdown]
"""
## Introduction

This example focuses on a simple HAZUS Earthquake application
involving a single building.

"""

# %%
# Imports
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from pelicun.assessment import Assessment, DLCalculationAssessment
from pelicun.base import convert_to_MultiIndex

idx = pd.IndexSlice
pd.options.display.max_rows = 30

# %% [markdown]
"""
## Initialize Assessment

When creating a Pelicun Assessment, you can provide a number of
settings to control the the analysis. The following options are
currently available:

- **Verbose** If True, provides more detailed messages about the
  calculations. Default: False.

- **Seed** Providing a seed makes probabilistic calculations
  reproducible. Default: No seed.

- **PrintLog** If True, prints the messages on the screen as well as
  in the log file. Default: False.

- **LogFile** Allows printing the log to a specific file under a path
  provided here as a string. By default, the log is printed to the
  pelicun_log.txt file.

- **LogShowMS** If True, Pelicun provides more detailed time
  information by showing times up to microsecond
  precision. Default: False, meaning times are provided with
  second precision.

- **SamplingMethod** Three methods are available: {'MonteCarlo',
  'LHS', 'LHS_midpoint'}; Default: LHS_midpoint
    * 'MonteCarlo' stands for conventional random sampling;
    * 'LHS' is Latin HyperCube Sampling with random sample location
       within each chosen bin of the hypercube;
    * 'LHS_midpoint' is like LHS, but the samples are assigned to
       the midpoints of the hypercube bins.

- **DemandOffset** Expects a dictionary with
  {demand_type:offset_value} key-value pairs. demand_type could be
  'PFA' or 'PIH' for example. The offset values are applied to the
  location values when Performance Group locations are parsed to
  demands that control the damage or losses. Default: {'PFA':-1,
  'PFV':-1}, meaning floor accelerations and velocities are pulled
  from the bottom slab associated with the given floor. For
  example, floor 2 would get accelerations from location 1, which
  is the first slab above ground.

- **NonDirectionalMultipliers** Expects a dictionary with
  {demand_type:scale_factor} key-value pairs. demand_type could be
  'PFA' or 'PIH' for example; use 'ALL' to define a scale factor
  for all demands at once. The scale factor considers that for
  components with non-directional behavior the maximum of demands
  is typically larger than the ones available in two orthogonal
  directions. Default: {'ALL': 1.2}, based on FEMA P-58.

- **RepairCostAndTimeCorrelation** Specifies the correlation
  coefficient between the repair cost and repair time of
  individual component blocks. Default: 0.0, meaning uncorrelated
  behavior. Use 1.0 to get perfect correlation or anything between
  0-1 to get partial correlation. Values in the -1 - 0 range are
  also valid to consider negative correlation between cost and
  time.

- **EconomiesOfScale** Controls how the damages are aggregated when
  the economies of scale are calculated. Expects the following
  dictionary: {'AcrossFloors': bool, 'AcrossDamageStates': bool}
  where bool is either True or False. Default: {'AcrossFloors':
  True, 'AcrossDamageStates': False}

  * 'AcrossFloors' if True, aggregates damages across floors to get
    the quantity of damage. If False, it uses damaged quantities and
    evaluates economies of scale independently for each floor.

  * 'AcrossDamageStates' if True, aggregates damages across damage
    states to get the quantity of damage. If False, it uses damaged
    quantities and evaluates economies of scale independently for each
    damage state.

We use the default values for this analysis and only ask for a seed
to make the results repeatable and ask to print the log file to show
outputs within this Jupyter notebook.
"""

# %%
# initialize a pelicun Assessment
assessment = Assessment({'PrintLog': True, 'Seed': 415})

# %% [markdown]
"""
## Demands

### Load demand distribution data

Demand distribution data was extracted from the FEMA P-58 background
documentation referenced in the Introduction. The nonlinear analysis
results from Figures 1-14 &ndash; 1-21 provide the 10th percentile,
median, and 90th percentile of EDPs in two directions on each floor
at each intensity level. We fit a lognormal distributions to those
data and collected the parameters of those distribution in the
demand_data.csv file.

Note that these results do not match the (non-directional) EDP
parameters in Table 1-35 &ndash; 1-42 in the report, so those must
have been processed in another way. The corresponding methodology is
not provided in the report; we are not using the results from those
tables in this example.
"""

# %%
raw_demands = pd.read_csv('example_1/demand_data.csv', index_col=0)
raw_demands

# %% [markdown]
"""
**Pelicun uses SimCenter's naming convention for demands:**

- The first number represents the event_ID. This can be used to
  differentiate between multiple stripes of an analysis, or multiple
  consecutive events in a main-shock - aftershock sequence, for
  example. Currently, Pelicun does not use the first number
  internally, but we plan to utilize it in the future.

- The type of the demand identifies the EDP or IM. The following
  options are available:
  * 'Story Drift Ratio' :              'PID',
  * 'Peak Interstory Drift Ratio':     'PID',
  * 'Roof Drift Ratio' :               'PRD',
  * 'Peak Roof Drift Ratio' :          'PRD',
  * 'Damageable Wall Drift' :          'DWD',
  * 'Racking Drift Ratio' :            'RDR',
  * 'Peak Floor Acceleration' :        'PFA',
  * 'Peak Floor Velocity' :            'PFV',
  * 'Peak Gust Wind Speed' :           'PWS',
  * 'Peak Inundation Height' :         'PIH',
  * 'Peak Ground Acceleration' :       'PGA',
  * 'Peak Ground Velocity' :           'PGV',
  * 'Spectral Acceleration' :          'SA',
  * 'Spectral Velocity' :              'SV',
  * 'Spectral Displacement' :          'SD',
  * 'Peak Spectral Acceleration' :     'SA',
  * 'Peak Spectral Velocity' :         'SV',
  * 'Peak Spectral Displacement' :     'SD',
  * 'Permanent Ground Deformation' :   'PGD',
  * 'Mega Drift Ratio' :               'PMD',
  * 'Residual Drift Ratio' :           'RID',
  * 'Residual Interstory Drift Ratio': 'RID'

- The third part is an integer the defines the location where the
  demand was recorded. In buildings, locations are typically floors,
  but in other assets, locations could reference any other part of the
  structure. Other pelicun examples show how location can also
  identify individual buildings in a regional analysis.

- The last part is an integer the defines the direction of the
  demand. Typically 1 stands for horizontal X and 2 for horizontal Y,
  but any other numbering convention can be used. Direction does not
  have to be used strictly to identify directions. It can be
  considered a generic second-level location identifier that
  differentiates demands and Performance Groups within a location.

The location and direction numbers need to be in line with the
component definitions presented later.

**MultiIndex and SimpleIndex in Pelicun**:

Pelicun uses a hierarchical indexing for rows and columns to organize
data efficiently internally. It provides methods to convert simple
indexes to hierarchical ones (so-called MultiIndex in Python's pandas
package). These methods require simple indexes follow some basic
formatting conventions:

- information at different levels is separated by a dash character: '-'

- no dash character is used in the labels themselves

- spaces are allowed, but are not preserved

The index of the DataFrame above shows how the simple index labels
look like and the DataFrame below shows how they are converted to a
hierarchical MultiIndex.
"""

# %%
# convert index to MultiIndex to make it easier to slice the data
raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
raw_demands.index.names = ['stripe', 'type', 'loc', 'dir']
raw_demands.tail(30)

# %% [markdown]
"""
### Prepare demand input for pelicun

Pelicun offers several options to obtain a desired demand sample:
1. provide the sample directly;
2. provide a distribution (i.e., marginals, and optional correlation
   matrix) and sample it;
3. provide a small set of demand values, fit a distribution, and
   sample that distribution to a large enough sample for performance
   assessment.

In this example, we are going to use the demand information from the
FEMA P-58 background documentation to provide a marginal of each
demand type and sample it (i.e., option 2 from the list). Then, we
will extract the sample from pelicun, extend it with additional demand
types and load it back into Pelicun (i.e., option 1 from the list)

**Scenarios**

Currently, Pelicun performs calculations for one scenario at a
time. Hence, we need to select the stripe we wish to investigate from
the eight available stripes that were used in the multi-stripe
analysis.

**Units**

Pelicun allows users to choose from various units for the all inputs,
including demands. Internally, Pelicun uses Standard International
units, but we support typical units used in the United States as
well. Let us know if a unit you desire to use is not supported - you
will see an error message in this case - we are happy to extend the
list of supported units.
"""

# %%
# we'll use stripe 3 for this example
stripe = '3'
stripe_demands = raw_demands.loc[stripe, :]

# units - - - - - - - - - - - - - - - - - - - - - - - -
stripe_demands.insert(0, 'Units', '')

# PFA is in "g" in this example, while PID is "rad"
stripe_demands.loc['PFA', 'Units'] = 'g'
stripe_demands.loc['PID', 'Units'] = 'rad'

# distribution family  - - - - - - - - - - - - - - - - -
stripe_demands.insert(1, 'Family', '')

# we assume lognormal distribution for all demand marginals
stripe_demands['Family'] = 'lognormal'

# distribution parameters  - - - - - - - - - - - - - - -
# pelicun uses generic parameter names to handle various distributions within the same data structure
# we need to rename the parameter columns as follows:
# median -> theta_0
# log_std -> theta_1
stripe_demands = stripe_demands.rename(
    columns={'median': 'Theta_0', 'log_std': 'Theta_1'}
)

stripe_demands

# %% [markdown]
# Let's plot the demand data to perform a sanity check before the
# analysis

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        '<b>Peak Interstory Drift ratio</b><br> ',
        '<b>Peak Floor Acceleration</b><br> ',
    ),
    shared_yaxes=True,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
)

for demand_i, demand_type in enumerate(['PID', 'PFA']):
    if demand_type == 'PID':
        offset = -0.5
    else:
        offset = 0.0

    for d_i, (dir_, d_color) in enumerate(zip([1, 2], ['blue', 'red'])):
        result_name = f'{demand_type} dir {dir_}'

        params = stripe_demands.loc[
            idx[demand_type, :, str(dir_)], ['Theta_0', 'Theta_1']
        ]
        params.index = params.index.get_level_values(1).astype(float)

        # plot +- 2 log std
        for mul, m_dash in zip([1, 2], ['dash', 'dot']):
            if mul == 1:
                continue

            for sign in [-1, 1]:
                fig.add_trace(
                    go.Scatter(
                        x=np.exp(
                            np.log(params['Theta_0'].values)
                            + params['Theta_1'].to_numpy() * sign * mul
                        ),
                        y=params.index + offset,
                        hovertext=result_name + ' median +/- 2logstd',
                        name=result_name + ' median +/- 2logstd',
                        mode='lines+markers',
                        line={'color': d_color, 'dash': m_dash, 'width': 0.5},
                        marker={'size': 4 / mul},
                        showlegend=False,
                    ),
                    row=1,
                    col=demand_i + 1,
                )

        # plot the medians
        fig.add_trace(
            go.Scatter(
                x=params['Theta_0'].values,
                y=params.index + offset,
                hovertext=result_name + ' median',
                name=result_name + ' median',
                mode='lines+markers',
                line={'color': d_color, 'width': 1.0},
                marker={'size': 8},
                showlegend=False,
            ),
            row=1,
            col=demand_i + 1,
        )

        if d_i == 0:
            shared_ax_props = {
                'showgrid': True,
                'linecolor': 'black',
                'gridwidth': 0.05,
                'gridcolor': 'rgb(192,192,192)',
            }

            if demand_type == 'PID':
                fig.update_xaxes(
                    title_text='drift ratio',
                    range=[0, 0.05],
                    row=1,
                    col=demand_i + 1,
                    **shared_ax_props,
                )

            elif demand_type == 'PFA':
                fig.update_xaxes(
                    title_text='acceleration [g]',
                    range=[0, 1.0],
                    row=1,
                    col=demand_i + 1,
                    **shared_ax_props,
                )

            if demand_i == 0:
                fig.update_yaxes(
                    title_text='story',
                    range=[0, 4],
                    row=1,
                    col=demand_i + 1,
                    **shared_ax_props,
                )
            else:
                fig.update_yaxes(
                    range=[0, 4], row=1, col=demand_i + 1, **shared_ax_props
                )

fig.update_layout(
    title=f'intensity level {stripe} ~ 475 yr return period',
    height=500,
    width=900,
    plot_bgcolor='white',
)

fig.show()

# %% [markdown]
"""
### Sample the demand distribution

The scripts below load the demand marginal information to Pelicun and
ask it to generate a sample with the provided number of
realizations. We do not have correlation information from the
background documentation, but it is generally better (i.e.,
conservative from a damage, loss, and risk point of view) to assume
perfect correlation in such cases than to assume independence. Hence,
we prepare a correlation matrix that represents perfect correlation
and feed it to Pelicun with the marginal parameters.

After generating the sample, we extract it and print the first few
realizations below.
"""

# %%
# prepare a correlation matrix that represents perfect correlation
ndims = stripe_demands.shape[0]
demand_types = stripe_demands.index

perfect_corr = pd.DataFrame(
    np.ones((ndims, ndims)), columns=demand_types, index=demand_types
)

# load the demand model
assessment.demand.load_model(
    {'marginals': stripe_demands, 'correlation': perfect_corr}
)

# %%
# choose a sample size for the analysis
sample_size = 10000

# generate demand sample
assessment.demand.generate_sample({'SampleSize': sample_size})

# extract the generated sample

# Note that calling the save_sample() method is better than directly
# pulling the sample attribute from the demand object because the
# save_sample method converts demand units back to the ones you
# specified when loading in the demands.

demand_sample = assessment.demand.save_sample()

demand_sample.head()

# %% [markdown]
r"""
### Extend the sample

The damage and loss models we use later in this example need residual
drift and spectral acceleration [Sa(T=1.13s)] information for each
realizations. The residual drifts are used to consider irreparable
damage to the building; the spectral accelerations are used to
evaluate the likelihood of collapse.

**Residual drifts**

Residual drifts could come from nonlinear analysis, but they are often
not available or not robust enough. Pelicun provides a convenience
method to convert PID to RID and we use that function in this
example. Currently, the method implements the procedure recommended in
FEMA P-58, but it is designed to support multiple approaches for
inferring RID from available demand information.

The FEMA P-58 RID calculation is based on the yield drift ratio. There
are conflicting data in FEMA P-58 on the yield drift ratio that should
be applied for this building:

* According to Vol 2 4.7.3, $\Delta_y = 0.0035$ , but this value leads
  to excessive irreparable drift likelihood that does not match the
  results in the background documentation.

* According to Vol 1 Table C-2, $\Delta_y = 0.0075$ , which leads to
  results that are more in line with those in the background
  documentation.

We use the second option below. Note that we get a different set of
residual drift estimates for every floor of the building.

**Spectral acceleration**

The Sa(T) can be easily added as a new column to the demand
sample. Note that Sa values are identical across all realizations
because we are running the analysis for one stripe that has a
particular Sa(T) assigned to it. We assign the Sa values to direction
1 and we will make sure to have the collapse fragility defined as a
directional component (see Damage/Fragility data) to avoid scaling
these spectral accelerations with the nondirectional scale factor.

The list below provides Sa values for each stripe from the analysis -
the values are from the background documentation referenced in the
Introduction.
"""

# %%
# get residual drift estimates
delta_y = 0.0075
PID = demand_sample['PID']

RID = assessment.demand.estimate_RID(PID, {'yield_drift': delta_y})

# and join them with the demand_sample
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

# add spectral acceleration
Sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756]
demand_sample_ext['SA_1.13', 0, 1] = Sa_vals[int(stripe) - 1]

demand_sample_ext.describe().T

# %% [markdown]
"""
The plot below illustrates that the relationship between a PID and RID
variable is not multivariate lognormal. This underlines the importance
of generating the sample for such additional demand types
realization-by-realization rather than adding a marginal RID to the
initial set and asking Pelicun to sample RIDs from a multivariate
lognormal distribution.

You can use the plot below to display the joint distribution of any
two demand variables
"""

# %%
# plot two demands from the sample

demands = ['PID-1-1', 'RID-1-1']

fig = go.Figure()

demand_file = 'response.csv'
output_path = 'doc/source/examples/notebooks/example_1/output'
coupled_edp = True
realizations = '100'
auto_script_path = 'PelicunDefault/Hazus_Earthquake_IM.py'
detailed_results = False
output_format = None
custom_model_dir = None
color_warnings = False

shared_ax_props = {
    'showgrid': True,
    'linecolor': 'black',
    'gridwidth': 0.05,
    'gridcolor': 'rgb(192,192,192)',
    'type': 'log',
}

config_options = {
    'NonDirectionalMultipliers': {'ALL': 1.0},
    'Sampling': {'SampleSize': 1000},
    'LogFile': None,
    'Verbose': True,
    'ListAllDamageStates': True,
}

assessment = DLCalculationAssessment(config_options=config_options)

assessment.calculate_demand(
    demand_path=Path(
        'doc/source/examples/notebooks/' 'example_1/response.csv'
    ).resolve(),
    collapse_limits=None,
    length_unit='ft',
    demand_calibration=None,
    sample_size=1000,
    coupled_demands=True,
    demand_cloning=None,
    residual_drift_inference=None,
)
demand_sample, demand_units = assessment.demand.save_sample(None, save_units=True)

assessment.calculate_asset(
    num_stories=1,
    component_assignment_file=(
        'doc/source/examples/notebooks/example_1/output/CMP_QNT.csv'
    ),
    collapse_fragility_demand_type=None,
    add_irreparable_damage_columns=False,
    component_sample_file=None,
)
assessment.asset.cmp_sample

assessment.calculate_damage(
    length_unit='ft',
    component_database='Hazus Earthquake - Buildings',
    component_database_path=None,
    collapse_fragility=None,
    is_for_water_network_assessment=False,
    irreparable_damage=None,
    damage_process_approach='Hazus Earthquake',
    damage_process_file_path=None,
    custom_model_dir=None,
)

assessment.damage.ds_model.sample.mean()
