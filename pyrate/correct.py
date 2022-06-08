#   This Python module is part of the PyRate software package.
#
#   Copyright 2022 Geoscience Australia
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# coding: utf-8
"""
This Python module runs the main PyRate correction workflow
"""
import shutil
import os
from pathlib import Path
import pickle as cp

import pyrate.constants as C
from pyrate.core import (shared, algorithm, mpiops)
from pyrate.core.aps import spatio_temporal_filter
from pyrate.core.covariance import maxvar_vcm_calc_wrapper
from pyrate.core.mst import mst_calc_wrapper
from pyrate.core.orbital import orb_fit_calc_wrapper
from pyrate.core.dem_error import dem_error_calc_wrapper
from pyrate.core.phase_closure.closure_check import iterative_closure_check, \
    mask_pixels_with_unwrapping_errors, update_ifg_list
from pyrate.core.ref_phs_est import ref_phase_est_wrapper
from pyrate.core.refpixel import ref_pixel_calc_wrapper
from pyrate.core.shared import PrereadIfg, Ifg, get_tiles, mpi_vs_multiprocess_logging, \
    join_dicts, nan_and_mm_convert, save_numpy_phase
from pyrate.core.logger import pyratelogger as log
from pyrate.configuration import Configuration, ConfigException

MAIN_PROCESS = 0


def create_ifg_dict(config: Configuration):
    """
    Save the preread_ifgs dict with information about the ifgs that are
    later used for fast loading of Ifg files in IfgPart class

    :param list dest_tifs: List of destination tifs
    :param dict config: The workflow configuration
    :param list tiles: List of all Tile instances

    :return: preread_ifgs: Dictionary containing information regarding
                interferograms that are used later in workflow
    :rtype: dict
    """
    dest_tifs = list(config.interferogram_files)
    ifgs_dict = {}
    process_tifs = mpiops.array_split(dest_tifs)
    for d in process_tifs:
        ifg = Ifg(d.tmp_sampled_path) # get the writable copy
        ifg.open()
        nan_and_mm_convert(ifg, config)
        ifgs_dict[d.tmp_sampled_path] = PrereadIfg(
            path=d.sampled_path,
            tmp_path=d.tmp_sampled_path,
            nan_fraction=ifg.nan_fraction,
            first=ifg.first,
            second=ifg.second,
            time_span=ifg.time_span,
            nrows=ifg.nrows,
            ncols=ifg.ncols,
            metadata=ifg.meta_data
        )
        ifg.write_modified_phase() # update phase converted to mm
        ifg.close()
    ifgs_dict = join_dicts(mpiops.comm.allgather(ifgs_dict))

    ifgs_dict = mpiops.run_once(
        _save_ifgs_dict_with_headers_and_epochs,
        dest_tifs, ifgs_dict, config, process_tifs
    )

    config.preread_ifgs = ifgs_dict
    return ifgs_dict


def _save_ifgs_dict_with_headers_and_epochs(
    dest_tifs,
    ifgs_dict,
    config: Configuration, process_tifs
):
    if not os.path.exists(config.tmpdir):
        shared.mkdir_p(config.tmpdir)

    preread_ifgs_file = Configuration.preread_ifgs(config)
    nifgs = len(dest_tifs)
    # add some extra information that's also useful later
    gt, md, wkt = shared.get_geotiff_header_info(process_tifs[0].tmp_sampled_path)
    epochlist = algorithm.get_epochs(ifgs_dict)[0]
    log.info(f'Found {len(epochlist.dates)} unique epochs in the {nifgs} interferogram network')
    ifgs_dict['epochlist'] = epochlist
    ifgs_dict['gt'] = gt
    ifgs_dict['md'] = md
    ifgs_dict['wkt'] = wkt
    # dump ifgs_dict file for later use
    with open(preread_ifgs_file, 'wb') as file:
        cp.dump(ifgs_dict, file)

    for k in ['gt', 'epochlist', 'md', 'wkt']:
        ifgs_dict.pop(k)

    return ifgs_dict


def _copy_mlooked(config: Configuration):
    """
    Make a copy of the multi-looked files in the 'tmp_sampled_path'
    for manipulation during correct steps
    """
    log.info("Copying input files into tempdir for manipulation during 'correct' steps")
    process_mpaths = mpiops.array_split(config.interferogram_files)
    for path in process_mpaths:
        shutil.copy(path.sampled_path, path.tmp_sampled_path)
        # assign write permission as prepifg output is readonly
        Path(path.tmp_sampled_path).chmod(0o664)


def main(config: Configuration):
    """
    Top level function to perform PyRate workflow on given interferograms

    :param Configuration config: The workflow configuration parameters

    :return: refpt: tuple of reference pixel x and y position
    :rtype: tuple
    :return: maxvar: array of maximum variance values of interferograms
    :rtype: ndarray
    :return: vcmt: Variance-covariance matrix array
    :rtype: ndarray
    """
    mpi_vs_multiprocess_logging("correct", config)

    _copy_mlooked(config)

    return correct_ifgs(config)


def update_params_with_tiles(config: Configuration) -> None:
    ifg_path = config.interferogram_files[0].tmp_sampled_path

    # FIXME: putting runtime state in 'params' / the config, isn't ideal - "params"
    # has basically turned into a set of global variables...

    # add tiles to params
    config.tiles = mpiops.run_once(get_tiles, ifg_path, config.rows, config.cols)


def phase_closure_wrapper(config: Configuration) -> dict:
    """
    This wrapper will run the iterative phase closure check to return a stable
    list of checked interferograms, and then mask pixels in interferograms that
    exceed the unwrapping error threshold.
    :param config: Configuration class instance.
    :return: params: Updated dictionary of PyRate configuration parameters.
    """

    if not config.phase_closure:
        log.info("Phase closure correction is not required!")
        return None

    rets = iterative_closure_check(config)
    if rets is None:
        log.info("Zero loops are returned from the iterative closure check.")
        log.warning("Abandoning phase closure correction without modifying the interferograms.")
        return None

    ifg_files, ifgs_breach_count, num_occurences_each_ifg = rets

    # update params with closure checked ifg list
    config.interferogram_files = \
        mpiops.run_once(update_ifg_list, ifg_files, config.interferogram_files)

    if mpiops.rank == 0:
        with open(config.phase_closure_filtered_ifgs_list(config), 'w', encoding="utf-8") as f:
            lines = [p.converted_path + '\n' for p in config.interferogram_files]
            f.writelines(lines)

    # mask ifgs with nans where phase unwrap threshold is breached
    if mpiops.rank == 0:
        mask_pixels_with_unwrapping_errors(ifgs_breach_count, num_occurences_each_ifg, config)

    create_ifg_dict(config) # update the preread_ifgs dict

    ifg_paths = [ifg_path.tmp_sampled_path for ifg_path in config.interferogram_files]
    # update/save the phase_data in the tiled numpy files
    save_numpy_phase(ifg_paths, config)

    # FIXME: Why... this was always returning the same value that was input...
    return config


correct_steps = {
    'orbfit': orb_fit_calc_wrapper,
    'refphase': ref_phase_est_wrapper,
    'phase_closure': phase_closure_wrapper,
    'demerror': dem_error_calc_wrapper,
    'mst': mst_calc_wrapper,
    'apscorrect': spatio_temporal_filter,
    'maxvar': maxvar_vcm_calc_wrapper,
}


def correct_ifgs(config: Configuration) -> None:
    """
    Top level function to perform PyRate workflow on given interferograms
    """
    validate_correct_steps(config)

    # work out the tiling and add to params dict
    update_params_with_tiles(config)

    # create the preread_ifgs dict for use with tiled data
    create_ifg_dict(config)

    ifg_paths = [ifg_path.tmp_sampled_path for ifg_path in config.interferogram_files]

    # create initial tiled phase_data numpy files on disc
    save_numpy_phase(ifg_paths, config)

    config.refxfound, config.refyfound = ref_pixel_calc_wrapper(config)

    # run through the correct steps in user specified sequence
    for step in config['correct']:
        correct_steps[step](config)
    log.info("Finished 'correct' step")


def validate_correct_steps(config: Configuration):
    for step in config['correct']:
        if step not in correct_steps:
            raise ConfigException(f"{step} is not a supported 'correct' step. \n"
                                  f"Supported steps are {correct_steps.keys()}")
