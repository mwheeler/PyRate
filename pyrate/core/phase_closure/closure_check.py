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

from collections import defaultdict
from typing import List, Tuple, Any
from nptyping import NDArray, UInt16, Float32
import numpy as np

from pyrate.core import mpiops
from pyrate.core.phase_closure.mst_closure import sort_loops_based_on_weights_and_date, \
    WeightedLoop, Edge
from pyrate.configuration import Configuration, MultiplePaths
from pyrate.core.phase_closure.sum_closure import sum_phase_closures
from pyrate.core.phase_closure.plot_closure import plot_closure
from pyrate.core.shared import Ifg, nan_and_mm_convert
from pyrate.core.logger import pyratelogger as log


def mask_pixels_with_unwrapping_errors(
    ifgs_breach_count: NDArray[(Any, Any, Any), UInt16],
    num_occurrences_each_ifg: NDArray[(Any,), UInt16],
    config: Configuration
):
    """
    Find pixels in the phase data that breach closure_thr, and mask
    (assign NaNs) to those pixels in those ifgs.
    :param ifgs_breach_count: unwrapping issues at pixels in all loops
    :param num_occurrences_each_ifg:  frequency of ifgs appearing in all loops
    :param config: The PyRate configuration parameters
    """
    log.debug("Masking phase data of retained ifgs")

    ifg_files = config.interferogram_files

    for i, m_p in enumerate(ifg_files):
        pix_index = ifgs_breach_count[:, :, i] == num_occurrences_each_ifg[i]
        ifg = Ifg(m_p.tmp_sampled_path)
        ifg.open()
        nan_and_mm_convert(ifg, config)
        ifg.phase_data[pix_index] = np.nan
        ifg.write_modified_phase()

    log.info(f"Masked phase data of {len(ifg_files)} retained ifgs after phase closure")


def __drop_ifgs_if_not_part_of_any_loop(
    ifg_files: List[str],
    loops: List[WeightedLoop],
    config: Configuration
) -> List[str]:
    """
    Check if an ifg is part of any of the loops, otherwise drop it from the list of interferograms
    for further PyRate processing.
    """
    loop_ifgs = set()
    for weighted_loop in loops:
        for edge in weighted_loop.loop:
            loop_ifgs.add(Edge(edge.first, edge.second))

    ifgs = [Ifg(i) for i in ifg_files]
    for i in ifgs:
        i.open()
        i.nodata_value = config.noDataValue
    selected_ifg_files = []
    for i, f in zip(ifgs, ifg_files):
        if Edge(i.first, i.second) in loop_ifgs:
            selected_ifg_files.append(f)
    if len(ifg_files) != len(selected_ifg_files):
        log.info(f'Only {len(selected_ifg_files)} (out of {len(ifg_files)}) ifgs participate in '
                 f'one or more closure loops, and are selected for further PyRate analysis')
    return selected_ifg_files


def __drop_ifgs_exceeding_threshold(
    orig_ifg_files: List[str],
    ifgs_breach_count,
    num_occurences_each_ifg,
    config: Configuration
):
    """
    Function to identify and drop ifgs, based on two thresholds.
    We demand two thresholds to be breached before an ifg is dropped:
    1. min_loops_per_ifg: the basic ifg loop participation count check: does the ifg participate in
                          enough loops to accurately check for unwrapping errors?
    2. The second threshold is an average check of pixels breached taking all loops into account.
       It is evaluated as follows:
        (i) ifgs_breach_count contains the number of loops where this pixel in this ifg had
            a closure exceeding closure_thr.
        (b) sum(ifgs_breach_count[:, :, i]) is the number of pixels in ifg exceeding closure_thr
            over all loops
        (c) divide by loop_count_of_this_ifg and num of cells (nrows x ncols) for a normalised
            measure of threshold.
    """
    orig_ifg_files.sort()
    nrows, ncols, _ = ifgs_breach_count.shape
    selected_ifg_files = []
    for i, ifg_file in enumerate(orig_ifg_files):
        loop_count_of_this_ifg = num_occurences_each_ifg[i]
        if loop_count_of_this_ifg:  # if the ifg participated in at least one loop
            ifg_breached = np.sum(ifgs_breach_count[:, :, i] == loop_count_of_this_ifg)
            ifg_remove_threshold_breached = ifg_breached / (nrows * ncols) > config.ifg_drop_thr

            if not (
                    # min loops count # check 1
                    (num_occurences_each_ifg[i] > config.min_loops_per_ifg)
                    and
                    ifg_remove_threshold_breached  # and breached threshold
            ):
                selected_ifg_files.append(ifg_file)

    return selected_ifg_files


def iterative_closure_check(
    config: Configuration,
    interactive_plot = True
) -> Tuple[List[str], NDArray[(Any, Any, Any), UInt16], NDArray[(Any,), UInt16]]:
    """
    This function iterates the closure check until a stable list of interferogram files is returned.
    :param config: The PyRate configuration parameters
    :param interactive_plot: bool, whether to plot sum closures of loops
    :return: stable list of ifg files, their ifgs_breach_count, and
        number of occurrences of ifgs in loops
    """
    ifg_files = [ifg_path.tmp_sampled_path for ifg_path in config.interferogram_files]
    i = 1  # iteration counter

    while True:  # iterate till ifgs/loops are stable
        log.info(f"Closure check iteration #{i}: working on {len(ifg_files)} ifgs")
        rets = __wrap_closure_check(config)
        if rets is None:
            return None

        new_ifg_files, closure, ifgs_breach_count, num_occurences_each_ifg, loops = rets
        if interactive_plot:
            if mpiops.rank == 0:
                plot_closure(closure=closure, loops=loops, config=config,
                             thr=config.closure_thr, iteration=i)
        if len(ifg_files) == len(new_ifg_files):
            break

        i += 1
        ifg_files = new_ifg_files
        # exit condition could be some other check like number_of_loops

    mpiops.comm.barrier()

    log.info(
        f"Stable list of ifgs achieved after iteration #{i}. {len(ifg_files)} ifgs are retained"
    )
    return ifg_files, ifgs_breach_count, num_occurences_each_ifg


def discard_loops_containing_max_ifg_count(
    loops: List[WeightedLoop],
    config: Configuration
) -> List[WeightedLoop]:
    """
    This function will discard loops when each ifg participating in a loop has met the
    max loop count criteria.

    :param loops: list of loops
    :param config: The PyRate configuration parameters
    :return: selected loops satisfying MAX_LOOP_REDUNDANCY criteria
    """
    selected_loops = []
    ifg_counter = defaultdict(int)
    for loop in loops:
        edge_appearances = np.array([ifg_counter[e] for e in loop.edges])
        if not np.all(edge_appearances > config.max_loop_redundancy):
            selected_loops.append(loop)
            for edge in loop.edges:
                ifg_counter[edge] += 1
        else:
            log.debug(f"Loop {loop.loop} ignored: all constituent ifgs have been in a loop "
                      f"{config.max_loop_redundancy} times or more")
    return selected_loops


def __wrap_closure_check(config: Configuration) -> \
        Tuple[
            List[str],
            NDArray[(Any, Any), Float32],
            NDArray[(Any, Any, Any), UInt16],
            NDArray[(Any,), UInt16],
            List[WeightedLoop]]:
    """
    This wrapper function returns the closure check outputs for a single iteration of closure check.

    :param config: The PyRate configuraiton parameters
    For return variables see docstring in `sum_phase_closures`.
    """
    ifg_files = [ifg_path.tmp_sampled_path for ifg_path in config.interferogram_files]
    ifg_files.sort()
    log.debug(f"The number of ifgs in the list is {len(ifg_files)}")
    sorted_signed_loops = mpiops.run_once(sort_loops_based_on_weights_and_date, config)
    log.info(f"Total number of selected closed loops with up to MAX_LOOP_LENGTH = "
             f"{config.max_loop_length} edges is {len(sorted_signed_loops)}")

    if len(sorted_signed_loops) < 1:
        return None

    retained_loops = mpiops.run_once(
        discard_loops_containing_max_ifg_count,
        sorted_signed_loops, config
    )

    ifgs_with_loops = mpiops.run_once(
        __drop_ifgs_if_not_part_of_any_loop,
        ifg_files, retained_loops, config
    )

    msg = f"After applying MAX_LOOP_REDUNDANCY = {config.max_loop_redundancy} criteria, " \
          f"{len(retained_loops)} loops are retained"
    if len(retained_loops) < 1:
        return None

    log.info(msg)

    closure_sum = sum_phase_closures(ifgs_with_loops, retained_loops, config)
    closure, ifgs_breach_count, num_occurences_each_ifg = closure_sum

    if mpiops.rank == 0:
        closure_ins = config.closure()
        np.save(closure_ins.closure, closure)
        np.save(closure_ins.ifgs_breach_count, ifgs_breach_count)
        np.save(closure_ins.num_occurences_each_ifg, num_occurences_each_ifg)
        np.save(closure_ins.loops, retained_loops, allow_pickle=True)

    selected_ifg_files = mpiops.run_once(
        __drop_ifgs_exceeding_threshold,
        ifgs_with_loops, ifgs_breach_count, num_occurences_each_ifg, config
    )

    # update the ifg list in the parameters dictionary
    config.interferogram_files = \
        mpiops.run_once(update_ifg_list, selected_ifg_files, config.interferogram_files)
    return selected_ifg_files, closure, ifgs_breach_count, num_occurences_each_ifg, retained_loops


def update_ifg_list(ifg_files: List[str], multi_paths: List[MultiplePaths]) -> List[MultiplePaths]:
    """
    Function to extract full paths for a subsetted list of interferograms
    :param ifg_files: list of interferograms to subset
    :param multi_paths: list of full paths for list of original interferograms
    :return: filtered_multi_paths: list of full paths for the subset
    """
    filtered_multi_paths = []
    for m_p in multi_paths:
        if m_p.tmp_sampled_path in ifg_files:
            filtered_multi_paths.append(m_p)
    return filtered_multi_paths
