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

"""
These are unit and integration tests on the uq module of pelicun.
"""

import pickle
import itertools
import os
import re
import inspect
import pytest
import numpy as np
from scipy.stats import norm
from pelicun import uq

RNG = np.random.default_rng(40)

# for tests, we sometimes create things or call them just to see if
# things would work, so the following are irrelevant:

# pylint: disable=unused-variable
# pylint: disable=pointless-statement


# The tests maintain the order of definitions of the `uq.py` file.


def export_pickle(filepath, obj, makedirs=True):
    """
    Auxiliary function to export a pickle object.
    Parameters
    ----------
    filepath: str
      The path of the file to be exported,
      including any subdirectories.
    obj: object
      The object to be pickled
    makedirs: bool
      If True, then the directories preceding the filename
      will be created if they do not exist.
    """
    # extract the directory name
    dirname = os.path.dirname(filepath)
    # if making directories is requested,
    if makedirs:
        # and the path does not exist
        if not os.path.exists(dirname):
            # create the directory
            os.makedirs(dirname)
    # open the file with the given filepath
    with open(filepath, 'wb') as f:
        # and store the object in the file
        pickle.dump(obj, f)


def import_pickle(filepath):
    """
    Auxiliary function to import a pickle object.
    Parameters
    ----------
    filepath: str
      The path of the file to be imported.

    Returns
    -------
    The pickled object.

    """
    # open the file with the given filepath
    with open(filepath, 'rb') as f:
        # and retrieve the pickled object
        return pickle.load(f)


def reset_all_test_data(restore=True, purge=False):
    """
    Update the expected result pickle files with new results, accepting
    the values obtained by executing the code as correct from now on.

    Warning: This function should never be used if tests are
    failing. Its only purpose is to aid the development of more tests
    and keeping things tidy. If tests are failing, the specific tests
    need to be investigated, and after rectifying the cause, new
    expected test result values should be created at an individual
    basis.

    Note: This function assumes that the interpreter's current
    directory is the package root directory (`pelicun`). The code
    assumes that the test data directory exists.
    Data deletion only involves `.pcl` files that begin with `test_` and
    reside in /tests/data/uq.

    Parameters
    ----------
    restore: bool
      Whether to re-generate the test result data
    purge: bool
      Whether to remove the test result data before re-generating the
      new values.

    Raises
    ------

    ValueError
      If the test directory is not found.

    """

    # where the uq test result data are stored
    testdir = os.path.join(*('tests', 'data', 'uq'))
    if not os.path.exists(testdir):
        raise ValueError('tests/data/uq directory not found.')

    # clean up existing test result data
    # only remove .pcl files that start with `test_`
    pattern = re.compile(r'^test_.\.pcl')
    if purge:
        for root, _, files in os.walk('.'):
            for filename in files:
                if pattern.match(filename):
                    print(f'removing: {filename}')
                    file_path = os.path.join(root, filename)
                    os.remove(file_path)

    # generate new data
    if restore:
        # generate a list of functions defined in the current file.
        functions = [obj for obj in globals().values() if inspect.isfunction(obj)]
        # filter functions that have `reset` as one of their arguments
        reset_functions = [
            f for f in functions if 'reset' in f.__code__.co_varnames]
        # and their name begins with `test_`
        test_functions = [
            f for f in reset_functions if f.__name__.startswith('test_')]
        # execute them
        for f in test_functions:
            f(reset=True)

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.


def test_scale_distribution(reset=False):
    """
    Tests the functionality of the scale_distribution function.
    """

    # test data location
    data_dir = 'tests/data/uq/test_scale_distribution'

    # generate combinations of arguments
    args_iter = itertools.product(
        (2.00,),
        ('normal', 'lognormal', 'uniform'),
        (np.array((-1.00, 1.00)),),
        (np.array((-2.00, 2.00)),)
    )
    args_list = list(args_iter)
    args_list.append(
        (2.00, 'uniform', np.array((-1.00, 1.00)), np.array((-2.00, 2.00)))
    )

    # verify that each set works as intended
    for file_incr, arg in enumerate(args_list):
        # retrieve arguments
        factor, distr, theta, trunc = arg
        # run the function
        res = uq.scale_distribution(factor, distr, theta, trunc)
        # construct a filepath for the results
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res[0], compare[0])
        assert np.allclose(res[1], compare[1])


def test_mvn_orthotope_density(reset=False):
    """
    Tests the functionality of the mvn_orthotope_density function.
    """

    # test data location
    data_dir = 'tests/data/uq/test_mvn_orthotope_density'

    # generate combinations of arguments
    mu_vals = (
        0.00,
        0.00,
        0.00,
        np.array((0.00, 0.00)),
        np.array((0.00, 0.00)),
    )
    cov_vals = (
        1.00,
        1.00,
        1.00,
        np.array(
            ((1.00, 0.00),
             (0.00, 1.00))
        ),
        np.array(
            ((1.00, 0.50),
             (0.50, 1.00))
        ),
    )
    lower_vals = (
        -1.00,
        np.nan,
        +0.00,
        np.array((0.00, 0.00)),
        np.array((0.00, 0.00))
    )
    upper_vals = (
        -1.00,
        +0.00,
        np.nan,
        np.array((np.nan, np.nan)),
        np.array((np.nan, np.nan))
    )

    # verify that each set works as intended
    file_incr = 0
    for args in zip(
            mu_vals, cov_vals, lower_vals, upper_vals):
        file_incr += 1
        # run the function
        res = uq.mvn_orthotope_density(*args)
        # construct a filepath for the results
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res[0], compare[0])
        assert np.allclose(res[1], compare[1])


def test__get_theta(reset=False):
    """
    Tests the functionality of the _get_theta utility function.
    """

    # test data location
    data_dir = 'tests/data/uq/test__get_theta'

    # generate combinations of arguments
    res = uq._get_theta(
        np.array(
            (
                (1.00, 1.00),
                (1.00, 0.5)
            )
        ),
        np.array(
            (
                (0.00, 1.00),
                (1.00, 0.5)
            )
        ),
        ['normal', 'lognormal']
    )
    # construct a filepath for the results
    filename = f'{data_dir}/test_1.pcl'
    # overwrite results if needed
    if reset: export_pickle(filename, res)
    # retrieve expected results
    compare = import_pickle(filename)
    # verify equality
    assert np.allclose(res, compare)
    # verify failure
    with pytest.raises(ValueError):
        uq._get_theta(
            np.array((1.00,)), np.array((1.00,)),
            'not_a_distribution')


def test__get_limit_probs(reset=False):
    """
    Tests the functionality of the _get_limit_probs function.
    """

    # test data location
    data_dir = 'tests/data/uq/test__get_limit_probs'

    # generate combinations of arguments
    args_iter = itertools.product(
        (
            np.array((0.10, 0.20)),
            np.array((np.nan, 0.20)),
            np.array((0.10, np.nan)),
            np.array((np.nan, np.nan))
        ),
        ('normal', 'lognormal'),
        (np.array((0.15, 1.0)),)
    )

    # verify that each set works as intended
    for file_incr, args in enumerate(args_iter):
        # run the function
        res = uq._get_limit_probs(*args)
        # construct a filepath for the results
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res[0], compare[0])
        assert np.allclose(res[1], compare[1])
    # verify failure
    with pytest.raises(ValueError):
        uq._get_limit_probs(
            np.array((1.00,)),
            'not_a_distribution',
            np.array((1.00,)),
        )


def test__get_std_samples(reset=False):
    """
    Tests the functionality of the _get_std_samples utility function.
    """

    # test data location
    data_dir = 'tests/data/uq/test__get_std_samples'

    # generate combinations of arguments
    samples_list = [
        np.array((
            (1.00, 2.00, 3.00),
        )),
        np.array((
            (0.657965, 1.128253, 1.044239, 1.599209),
            (1.396495, 1.435923, 2.055659, 1.416298),
            (1.948161, 1.576571, 1.469571, 1.190853)
        )),
    ]
    theta_list = [
        np.array((
            (0.00, 1.0),
        )),
        np.array((
            (1.00, 0.20),
            (1.50, 0.6),
            (1.30, 2.0),
        )),
    ]
    tr_limits_list = [
        np.array((
            (np.nan, np.nan),
        )),
        np.array((
            (np.nan, np.nan),
            (1.10, np.nan),
            (np.nan, 2.80),
        ))
    ]
    dist_list_list = [
        np.array(('normal',)),
        np.array(('normal', 'lognormal', 'normal')),
    ]

    # verify that each set works as intended
    for file_incr, args in enumerate(zip(
            samples_list, theta_list, tr_limits_list, dist_list_list
    )):
        # run the function
        res = uq._get_std_samples(*args)
        # construct a filepath for the results
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res, compare)
    # verify failure
    with pytest.raises(ValueError):
        uq._get_std_samples(
            np.array((
                (1.00, 2.00, 3.00),
            )),
            np.array((
                (0.00, 1.0),
            )),
            np.array((
                (np.nan, np.nan),
            )),
            np.array(('some_unsupported_distribution',)),
        )


def test__get_std_corr_matrix(reset=False):
    """
    Tests the functionality of the _get_std_corr_matrix utility
    function.
    """

    # test data location
    data_dir = 'tests/data/uq/test__get_std_corr_matrix'

    # generate combinations of arguments
    std_samples_list = [
        np.array((
            (1.00,),
        )),
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        )),
        np.array((
            (1.00, 0.00),
            (0.00, -1.00)
        )),
        np.array((
            (1.00, 1.00),
            (1.00, 1.00)
        )),
        np.array((
            (1.00, 1e50),
            (-1.00, -1.00)
        )),
    ]

    # verify that each set works as intended
    for file_incr, std_samples in enumerate(std_samples_list):
        # run the function
        res = uq._get_std_corr_matrix(std_samples)
        # construct a filepath for the results
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res, compare)
    # verify failure
    for bad_item in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            x = np.array((
                (1.00, bad_item),
                (-1.00, -1.00)
            ))
            uq._get_std_corr_matrix(x)


def test__mvn_scale(reset=False):
    """
    Tests the functionality of the _mvn_scale utility function.
    """

    # test data location
    data_dir = 'tests/data/uq/test__mvn_scale'

    # generate combinations of arguments
    np.random.seed(40)
    sample_list = [
        np.random.normal(0.00, 1.00, size=(2, 5)).T,
        np.random.normal(1.0e10, 1.00, size=(2, 5)).T
    ]
    rho_list = [
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        )),
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        ))
    ]

    # verify that each set works as intended
    for file_incr, args in enumerate(zip(sample_list, rho_list)):
        # run the function
        res = uq._mvn_scale(*args)
        # construct a filepath for the results
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res, compare)


def test_fit_distribution_to_sample_univariate(reset=False):
    """
    Tests the functionality of the
    fit_distribution_to_sample_univariate function, only considering
    univariate input cases.
    """

    # test data location
    data_dir = 'tests/data/uq/test_fit_distribution_to_sample_univariate'

    file_incr = 0

    # baseline case
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    res = uq.fit_distribution_to_sample(
        sample_vec,
        'normal'
    )
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], np.inf)  # we know...
    assert np.isclose(res[1][0, 0], 1.00)

    # baseline case where the cov=mu/sigma is defined
    sample_vec += 10.00
    res = uq.fit_distribution_to_sample(
        sample_vec,
        'normal'
    )
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(
        res[0][0, 1],
        np.std(sample_vec) / np.mean(sample_vec))
    assert np.isclose(res[1][0, 0], 1.00)

    # lognormal
    log_sample_vec = np.log(sample_vec)
    res = uq.fit_distribution_to_sample(
        log_sample_vec,
        'lognormal'
    )
    assert np.isclose(np.log(res[0][0, 0]), np.mean(log_sample_vec))
    assert np.isclose(res[0][0, 1], np.std(log_sample_vec))
    assert np.isclose(res[1][0, 0], 1.00)

    # censored data, lower and upper
    np.random.seed(40)
    c_lower = 1.00 - 2.00 * 0.20
    c_upper = 1.00 + 2.00 * 0.20
    sample_vec = np.array(
        (1.19001858, 0.94546098, 1.17789766, 1.20168158, 0.91329968,
         0.92214045, 0.83480078, 0.75774220, 1.12245935, 1.11947970,
         0.84877398, 0.98338148, 0.68880282, 1.20237202, 0.94543761,
         1.26858046, 1.14934510, 1.21250879, 0.89558603, 0.90804330))
    usable_sample_idx = np.all(
        [sample_vec > c_lower, sample_vec < c_upper], axis=0)
    usable_sample = sample_vec[usable_sample_idx].reshape((1, -1))
    c_count = len(sample_vec) - len(usable_sample)
    usable_sample = usable_sample.reshape((1, -1))
    res_a = uq.fit_distribution_to_sample(
        usable_sample, 'normal',
        censored_count=c_count,
        detection_limits=[c_lower, c_upper])
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res_a)
    compare = import_pickle(filename)
    assert np.allclose(res_a[0], compare[0])
    assert np.allclose(res_a[1], compare[1])

    # censored data, only lower
    np.random.seed(40)
    c_lower = -1.50
    c_upper = np.inf
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00))
    usable_sample_idx = np.all([sample_vec > c_lower, sample_vec < c_upper], axis=0)
    usable_sample = sample_vec[usable_sample_idx].reshape((1, -1))
    c_count = len(sample_vec) - len(usable_sample)
    usable_sample = usable_sample.reshape((1, -1))
    res_b = uq.fit_distribution_to_sample(
        usable_sample, 'normal',
        censored_count=c_count,
        detection_limits=[c_lower, c_upper])
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res_b)
    compare = import_pickle(filename)
    assert np.allclose(res_b[0], compare[0])
    assert np.allclose(res_b[1], compare[1])

    # censored data, only upper
    np.random.seed(40)
    c_lower = -np.inf
    c_upper = 1.50
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00))
    usable_sample_idx = np.all([sample_vec > c_lower, sample_vec < c_upper], axis=0)
    usable_sample = sample_vec[usable_sample_idx].reshape((1, -1))
    c_count = len(sample_vec) - len(usable_sample)
    usable_sample = usable_sample.reshape((1, -1))
    res_c = uq.fit_distribution_to_sample(
        usable_sample, 'normal',
        censored_count=c_count,
        detection_limits=[c_lower, c_upper])
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res_c)
    compare = import_pickle(filename)
    assert np.allclose(res_c[0], compare[0])
    assert np.allclose(res_c[1], compare[1])

    # symmetry check
    assert np.isclose(res_b[0][0, 0], -res_c[0][0, 0])
    assert np.isclose(res_b[0][0, 1], res_c[0][0, 1])

    # truncated data, lower and upper, expect failure
    t_lower = -1.50
    t_upper = 1.50
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    with pytest.raises(ValueError):
        res = uq.fit_distribution_to_sample(
            sample_vec, 'normal',
            truncation_limits=[t_lower, t_upper])

    # truncated data, only lower, expect failure
    t_lower = -1.50
    t_upper = np.inf
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    with pytest.raises(ValueError):
        res = uq.fit_distribution_to_sample(
            sample_vec, 'normal',
            truncation_limits=[t_lower, t_upper])

    # truncated data, only upper, expect failure
    t_lower = -np.inf
    t_upper = 1.50
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    with pytest.raises(ValueError):
        res = uq.fit_distribution_to_sample(
            sample_vec, 'normal',
            truncation_limits=[t_lower, t_upper])

    # truncated data, lower and upper
    np.random.seed(40)
    t_lower = -4.50
    t_upper = 4.50
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    res_a = uq.fit_distribution_to_sample(
        sample_vec, 'normal',
        truncation_limits=[t_lower, t_upper])
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res_a)
    compare = import_pickle(filename)
    assert np.allclose(res_a[0], compare[0])
    assert np.allclose(res_a[1], compare[1])

    # truncated data, only lower
    np.random.seed(40)
    t_lower = -4.50
    t_upper = np.inf
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    res_b = uq.fit_distribution_to_sample(
        sample_vec, 'normal',
        truncation_limits=[t_lower, t_upper])
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res_b)
    compare = import_pickle(filename)
    assert np.allclose(res_b[0], compare[0])
    assert np.allclose(res_b[1], compare[1])

    # truncated data, only upper
    np.random.seed(40)
    t_lower = -np.inf
    t_upper = 4.50
    sample_vec = np.array(
        (-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape((1, -1))
    res_c = uq.fit_distribution_to_sample(
        sample_vec, 'normal',
        truncation_limits=[t_lower, t_upper])
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res_c)
    compare = import_pickle(filename)
    assert np.allclose(res_c[0], compare[0])
    assert np.allclose(res_c[1], compare[1])

    # symmetry check
    assert np.isclose(res_b[0][0, 0], -res_c[0][0, 0])
    assert np.isclose(res_b[0][0, 1], res_c[0][0, 1])


def test_fit_distribution_to_sample_multivariate(reset=False):
    """
    Tests the functionality of the
    fit_distribution_to_sample_univariate function, only considering
    multivariate input cases.
    """

    # test data location
    data_dir = 'tests/data/uq/test_fit_distribution_to_sample_multivariate'
    file_incr = 0

    # uncorrelated, normal
    np.random.seed(40)
    sample = np.random.multivariate_normal(
        (1.00, 1.00),
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        )),
        size=10000
    ).T
    np.random.seed(40)
    res = uq.fit_distribution_to_sample(
        sample,
        ['normal', 'normal']
    )
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res)
    compare = import_pickle(filename)
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # correlated, normal
    np.random.seed(40)
    sample = np.random.multivariate_normal(
        (1.00, 1.00),
        np.array((
            (1.00, 0.70),
            (0.70, 1.00)
        )),
        size=10000
    ).T
    np.random.seed(40)
    res = uq.fit_distribution_to_sample(
        sample,
        ['normal', 'normal']
    )
    file_incr += 1
    filename = f'{data_dir}/test_{file_incr}.pcl'
    if reset: export_pickle(filename, res)
    compare = import_pickle(filename)
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # more to come!


def test_fit_distribution_to_percentiles():
    """
    Tests the functionality of the fit_distribution_to_percentiles
    function.
    """

    # normal, mean of 20 and standard deviation of 10
    percentiles = np.linspace(0.01, 0.99, num=10000)
    values = norm.ppf(percentiles, loc=20, scale=10)
    # run the function to obtain the best distribution candidate and
    # its parameters
    res = uq.fit_distribution_to_percentiles(
        values, percentiles, ['normal', 'lognormal'])
    assert res[0] == 'normal'
    assert np.allclose(res[1], np.array((20.00, 10.00)))

    # lognormal, mean of 20 and standard deviation of 10
    # median and beta are calculated based on the above.
    ln_mu = 20.00
    ln_std = 10.00
    # calculate mu, std of the underlying normal distribution
    n_mu = np.log(ln_mu) - 0.50 * np.log(1.00 + (ln_std / ln_mu)**2)
    n_std = np.sqrt(np.log(1.00 + (ln_std / ln_mu)**2))
    percentiles = np.linspace(0.01, .99, num=10000)
    n_values = norm.ppf(percentiles, loc=n_mu, scale=n_std)
    # values that correspond to those percentiles for the lognormal distr
    ln_values = np.exp(n_values)
    # fit
    res = uq.fit_distribution_to_percentiles(
        ln_values, percentiles, ['normal', 'lognormal'])
    # theoretical lognormal distr median and beta (for assertions)
    ln_delta = ln_mu**2 / np.sqrt(ln_mu**2 + ln_std**2)
    ln_beta = np.sqrt(2.00 * np.log(np.sqrt(ln_mu**2 + ln_std**2) / ln_mu))
    assert res[0] == 'lognormal'
    assert np.allclose(res[1], np.array((ln_delta, ln_beta)))


#  __  __      _   _               _
# |  \/  | ___| |_| |__   ___   __| |___
# | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
# | |  | |  __/ |_| | | | (_) | (_| \__ \
# |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#
# The following tests verify the methods of the objects of the module.


def test_RandomVariable():
    """
    Tests the instantiation and basic functionality of the
    `RandomVariable` class.
    """

    # instantiate a random variable with default attributes
    rv_1 = uq.RandomVariable(
        'rv_1', 'empirical')
    # verify that the attributes have been assigned as expected
    assert rv_1.name == 'rv_1'
    assert rv_1._distribution == 'empirical'
    assert np.isnan(rv_1._theta[0])

    # instantiate a random variable with default attributes
    rv_2 = uq.RandomVariable(
        'rv_2',
        'coupled_empirical'
    )
    # verify that the attributes have been assigned as expected
    assert rv_2.name == 'rv_2'
    assert rv_2._distribution == 'coupled_empirical'
    assert np.isnan(rv_2._theta[0])

    # verify that other distributions require theta
    distributions = (
        'normal', 'lognormal', 'multinomial', 'custom',
        'uniform', 'deterministic')
    for distribution in distributions:
        with pytest.raises(ValueError):
            rv_err = uq.RandomVariable(  # noqa: F841
                "won't see the light of day",
                distribution)


def test_RandomVariable_cdf():
    """
    Tests the functionality of the `cdf` method of the
    `RandomVariable` class.
    """
    # create a normal random variable
    rv = uq.RandomVariable(
        'test_rv', 'normal', theta=(1.0, 1.0))

    # evaluate CDF at different points
    x = (-1.0, 0.0, 0.5, 1.0, 2.0)
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    assert np.allclose(
        cdf,
        (0.02275013, 0.15865525, 0.30853754,
         0.5, 0.84134475), rtol=1e-5)

    # same for lognormal
    rv = uq.RandomVariable(
        'test_rv', 'lognormal', theta=(1.0, 1.0))

    x = (-1.0, 0.0, 0.5, 1.0, 2.0)
    cdf = rv.cdf(x)

    assert np.allclose(
        cdf,
        (0.0, 0.0, 0.2441086, 0.5, 0.7558914),
        rtol=1e-5)

    # and for uniform
    rv = uq.RandomVariable(
        'test_rv', 'uniform', theta=(0.0, 1.0))

    x = (-1.0, 0.0, 0.5, 1.0, 2.0)
    cdf = rv.cdf(x)

    assert np.allclose(
        cdf,
        (0.0, 0.0, 0.5, 1.0, 1.0),
        rtol=1e-5)


def test_RandomVariable_inverse_transform():
    """
    Tests the functionality of the `inverse_transform` method of the
    `RandomVariable` class.
    """

    # create a uniform random variable
    rv = uq.RandomVariable('test_rv', 'uniform', theta=(0.0, 1.0))

    samples = np.array((0.10, 0.20, 0.30))

    # compute inverse transform of samples
    rv.uni_sample = samples
    rv.inverse_transform_sampling(samples)
    inverse_transform = rv.sample

    # assert that inverse transform values are correct
    assert np.allclose(inverse_transform, samples, rtol=1e-5)

    # create a lognormal random variable
    rv = uq.RandomVariable('test_rv', 'lognormal', theta=(1.0, 0.5))

    # compute inverse transform of samples
    rv.uni_sample = samples
    rv.inverse_transform_sampling(samples)
    inverse_transform = rv.sample

    # assert that inverse transform values are correct
    assert np.allclose(
        inverse_transform,
        np.array((0.52688352, 0.65651442, 0.76935694)),
        rtol=1e-5)

    # create a normal random variable
    rv = uq.RandomVariable('test_rv', 'normal', theta=(1.0, 0.5))

    # compute inverse transform of samples
    rv.uni_sample = samples
    rv.inverse_transform_sampling(samples)
    inverse_transform = rv.sample

    # assert that inverse transform values are correct
    assert np.allclose(
        inverse_transform,
        np.array((0.35922422, 0.57918938, 0.73779974)),
        rtol=1e-5)


def test_RandomVariable_Set():
    """
    Tests the instantiation and basic functionality of the
    `RandomVariable_Set` class.
    """

    rv_1 = uq.RandomVariable('rv1', 'normal', theta=(1.0, 1.0))
    rv_2 = uq.RandomVariable('rv2', 'normal', theta=(1.0, 1.0))
    rv_set = uq.RandomVariableSet(  # noqa: F841
        'test_set',
        (rv_1, rv_2),
        np.array(((1.0, 0.50), (0.50, 1.0)))
    )


def test_RandomVariable_Set_apply_correlation(reset=False):
    """
    Tests the functionality of the `apply_correlation` method of the
    `RandomVariable_Set` class.
    """

    data_dir = 'tests/data/uq/test_random_variable_set_apply_correlation'
    file_incr = 0

    # correlated, uniform
    np.random.seed(40)
    rv_1 = uq.RandomVariable(
        name='rv1',
        distribution='uniform',
        theta=(-5.0, 5.0)
    )
    rv_2 = uq.RandomVariable(
        name='rv2',
        distribution='uniform',
        theta=(-5.0, 5.0)
    )

    rv_1.uni_sample = np.random.random(size=100)
    rv_2.uni_sample = np.random.random(size=100)

    rvs = uq.RandomVariableSet(
        name='test_set',
        RV_list=[rv_1, rv_2],
        Rho=np.array((
            (1.0, 0.5),
            (0.5, 1.0)
        ))
    )
    rvs.apply_correlation()

    for rv in (rv_1, rv_2):
        res = rv.uni_sample
        file_incr += 1
        filename = f'{data_dir}/test_{file_incr}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)


def test_RandomVariable_Set_apply_correlation_special():
    """"
    This function tests the apply_correlation method of the
    RandomVariableSet class when given special input conditions.
    The first test checks that the method works when given a non
    positive semidefinite correlation matrix.
    The second test checks that the method works when given a non full
    rank matrix.
    """
    # inputs that cause `apply_correlation` to use the SVD

    # non positive semidefinite correlation matrix
    rho = np.array(((1.00, 0.50), (0.50, -1.00)))
    rv_1 = uq.RandomVariable('rv1', 'normal', theta=[5.0, 0.1])
    rv_2 = uq.RandomVariable('rv2', 'normal', theta=[5.0, 0.1])
    rv_1.uni_sample = np.random.random(size=100)
    rv_2.uni_sample = np.random.random(size=100)
    rv_set = uq.RandomVariableSet('rv_set', [rv_1, rv_2], rho)
    rv_set.apply_correlation()

    # non full rank matrix
    rho = np.array(((0.00, 0.00), (0.0, 0.0)))
    rv_1 = uq.RandomVariable('rv1', 'normal', theta=[5.0, 0.1])
    rv_2 = uq.RandomVariable('rv2', 'normal', theta=[5.0, 0.1])
    rv_1.uni_sample = np.random.random(size=100)
    rv_2.uni_sample = np.random.random(size=100)
    rv_set = uq.RandomVariableSet('rv_set', [rv_1, rv_2], rho)
    rv_set.apply_correlation()
    np.linalg.svd(rho, )


def test_RandomVariable_Set_orthotope_density(reset=False):
    """
    Tests the functionality of the `orthotope_density` method of the
    `RandomVariable_Set` class.
    """

    data_dir = 'tests/data/uq/test_random_variable_set_orthotope_density'

    # create some random variables
    rv_1 = uq.RandomVariable(
        'rv1', 'normal', theta=[5.0, 0.1],
        truncation_limits=np.array((np.nan, 10.0)))
    rv_2 = uq.RandomVariable('rv2', 'lognormal', theta=[10.0, 0.2])
    rv_3 = uq.RandomVariable('rv3', 'uniform', theta=[13.0, 17.0])
    rv_4 = uq.RandomVariable('rv4', 'uniform', theta=[0.0, 1.0])
    rv_5 = uq.RandomVariable('rv5', 'uniform', theta=[0.0, 1.0])

    # create a random variable set
    rv_set = uq.RandomVariableSet(
        'rv_set', (rv_1, rv_2, rv_3, rv_4, rv_5), np.identity(5))

    # define test cases
    test_cases = (
        # lower bounds, upper bounds, var_subset
        (
            np.array([4.0, 9.0, 14.0, np.nan]),
            np.array([6.0, 11., 16.0, 0.80]),
            ('rv1', 'rv2', 'rv3', 'rv4')
        ),
        (
            np.array([4.0, 9.0, 14.0, np.nan, 0.20]),
            np.array([6.0, 11., 16.0, 0.80, 0.40]),
            None
        ),
        (
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([6.0, 11., 16.0, 0.80, 0.40]),
            None
        ),
        (
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            None
        )
    )

    # loop over test cases
    for i, (lower, upper, var_subset) in enumerate(test_cases):
        # evaluate the density of the orthotope
        res = rv_set.orthotope_density(lower, upper, var_subset=var_subset)
        # check that the density is equal to the expected value
        # construct a filepath for the results
        filename = f'{data_dir}/test_{i+1}.pcl'
        # overwrite results if needed
        if reset: export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res, compare)


def test_RandomVariableRegistry_generate_sample(reset=False):
    """
    Tests the functionality of the `generate_sample` method of the
    `RandomVariableRegistry` class.
    """

    data_dir = 'tests/data/uq/test_RandomVariableRegistry_generate_sample'
    file_incr = 0

    for method in ('LHS_midpoint', 'LHS', 'MonteCarlo'):

        #
        # Random variable registry with a single random variable
        #

        # create the registry
        rng = np.random.default_rng(0)
        rv_registry_single = uq.RandomVariableRegistry(rng)
        # create the random variable and add it to the registry
        RV = uq.RandomVariable('x', distribution='normal', theta=[1.0, 1.0])
        rv_registry_single.add_RV(RV)

        # Generate a sample
        sample_size = 1000
        rv_registry_single.generate_sample(sample_size, method)

        res = rv_registry_single.RV_sample['x']
        assert len(res) == sample_size

        file_incr += 1
        filename = f'{data_dir}/test_{file_incr}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)

        # unfortunately, tests of stochastic outputs like like these fail
        # on rare occasions.
        # assert np.isclose(np.mean(res), 1.0, atol=1e-2)
        # assert np.isclose(np.std(res), 1.0, atol=1e-2)

        #
        # Random variable registry with multiple random variables
        #

        # create a random variable registry and add some random variables to it
        rng = np.random.default_rng(4)
        rv_registry = uq.RandomVariableRegistry(rng)
        rv_1 = uq.RandomVariable('rv1', 'normal', theta=[5.0, 0.1])
        rv_2 = uq.RandomVariable('rv2', 'lognormal', theta=[10.0, 0.2])
        rv_3 = uq.RandomVariable('rv3', 'uniform', theta=[13.0, 17.0])
        rv_registry.add_RV(rv_1)
        rv_registry.add_RV(rv_2)
        rv_registry.add_RV(rv_3)

        # create a random variable set and add it to the registry
        rv_set = uq.RandomVariableSet(
            'rv_set', [rv_1, rv_2, rv_3], np.identity(3) + np.full((3, 3), 0.20))
        rv_registry.add_RV_set(rv_set)

        # add some more random variables that are not part of the set
        rv_4 = uq.RandomVariable('rv4', 'normal', theta=[14.0, 0.30])
        rv_5 = uq.RandomVariable('rv5', 'normal', theta=[15.0, 0.50])
        rv_registry.add_RV(rv_4)
        rv_registry.add_RV(rv_5)

        rv_registry.generate_sample(10, method=method)

        # verify that all samples have been generated as expected
        for rv_name in (f'rv{i+1}' for i in range(5)):
            res = rv_registry.RV_sample[rv_name]
            file_incr += 1
            filename = f'{data_dir}/test_{file_incr}.pcl'
            if reset: export_pickle(filename, res)
            compare = import_pickle(filename)
            assert np.allclose(res, compare)

        # obtain multiple RVs from the registry
        rv_dictionary = rv_registry.RVs(('rv1', 'rv2'))
        assert 'rv1' in rv_dictionary
        assert 'rv2' in rv_dictionary
        with pytest.raises(KeyError):
            rv_dictionary['rv3']


if __name__ == '__main__':
    pass
