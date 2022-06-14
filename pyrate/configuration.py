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
"""
This Python module contains utilities to validate user input parameters
parsed in a PyRate configuration file.
"""
from enum import IntEnum
import re
from configparser import ConfigParser
from pathlib import Path, PurePath
from typing import Union, List

import pyrate.constants as C
from pyrate.constants import NO_OF_PARALLEL_PROCESSES, sixteen_digits_pattern, \
    twelve_digits_pattern, ORB_ERROR_DIR, DEM_ERROR_DIR, TEMP_MLOOKED_DIR
from pyrate.core import ifgconstants as ifg
from pyrate.default_parameters import PYRATE_DEFAULT_CONFIGURATION
from pyrate.core.algorithm import factorise_integer
from pyrate.core.shared import Tile, extract_epochs_from_filename, InputTypes, get_tiles


def set_parameter_value(data_type, input_value, default_value, required, input_name):
    """
    Convers a user-provided value into a final value, by applying data types and
    default values on the input.

    This function will raise an error if an input value is not given, but required.
    """
    if input_value is not None and len(input_value) < 1:
        input_value = None
        if required:  # pragma: no cover
            msg = f"A required parameter is missing from the configuration file: {input_name}"
            raise ValueError(msg)

    if input_value is not None:
        if str(data_type) in "path":
            return Path(input_value)
        return data_type(input_value)

    return default_value


def validate_parameter_value(
    input_name: str,
    input_value,
    min_value=None,
    max_value=None,
    possible_values=None
):
    """Validates that a supplied value sits within a range or set of allowed values."""

    if isinstance(input_value, PurePath):
        if not Path.exists(input_value):  # pragma: no cover
            raise ValueError(f"Given path: {input_value} does not exist.")

    if input_value is not None:
        if min_value is not None:
            if input_value < min_value:  # pragma: no cover
                msg = f"Invalid value for {input_name} supplied: {input_value}. " \
                      f"Provide a value greater than or equal to {min_value}."
                raise ValueError(msg)

        if max_value is not None:
            if input_value > max_value:  # pragma: no cover
                msg = f"Invalid value for {input_name} supplied: {input_value}. " \
                      f"Provide a value less than or equal to {max_value}."
                raise ValueError(msg)

    if possible_values is not None:
        if input_value not in possible_values:  # pragma: no cover
            msg = f"Invalid value for {input_name} supplied: {input_value}. " \
                  f"Provide one of these values: {possible_values}."
            raise ValueError(msg)
    return True


def validate_file_list_values(file_list, no_of_epochs):
    """Validates that the provided IFG files have enough epochs"""
    if file_list is None:  # pragma: no cover
        raise ValueError("No value supplied for input file list: " + str(file_list))

    files = parse_namelist(file_list)

    for file in files:
        if not Path(file).exists():  # pragma: no cover
            raise ConfigException(f"{file} does not exist")

        matches = extract_epochs_from_filename(filename_with_epochs=file)
        if len(matches) < no_of_epochs:  # pragma: no cover
            msg = f"the number of epochs in {file} names are less " \
                  f"the required number: {no_of_epochs}"
            raise ConfigException(msg)


class MultiplePaths:
    """A utility class which provides paths related to an interferogram file."""
    def __init__(self, file_name: str, params: dict, input_type: InputTypes = InputTypes.IFG):
        self.input_type = input_type
        out_dir = params[C.OUT_DIR]
        tempdir = params[C.TEMP_MLOOKED_DIR]
        if isinstance(tempdir, str):
            tempdir = Path(tempdir)
        b = Path(file_name)
        if input_type in [InputTypes.IFG, InputTypes.COH]:
            d = re.search(sixteen_digits_pattern, b.stem)
            if d is None:  # could be 6 digit epoch dates
                d = re.search(twelve_digits_pattern, b.stem)
            if d is None:
                msg = f"{input_type.value} filename does not contain two 8 or 6 digit date strings"
                raise ValueError(msg)
            filestr = d.group() + '_'
        else:
            filestr = ''

        dir_exists = input_type.value in InputTypes.DIR_MAP.value.keys()
        anchoring_dir = Path(out_dir).joinpath(InputTypes.DIR_MAP.value[input_type.value]) \
            if dir_exists else Path(out_dir)

        if b.suffix == ".tif":
            self.unwrapped_path = None
            converted_path = b  # original file
            self.sampled_path = anchoring_dir.joinpath(filestr + input_type.value + '.tif')
        else:
            self.unwrapped_path = b.as_posix()
            converted_path = b.stem.split('.')[0] + '_' + b.suffix[1:]
            converted_path = (anchoring_dir / converted_path).with_suffix('.tif')
            self.sampled_path = converted_path.with_name(filestr + input_type.value + '.tif')

        # tmp_sampled_paths are used after prepifg, during correct steps
        self.tmp_sampled_path = tempdir.joinpath(self.sampled_path.name).as_posix()
        self.converted_path = converted_path.as_posix()
        self.sampled_path = self.sampled_path.as_posix()

    @staticmethod
    def orb_error_path(ifg_path: Union[str, Path], params) -> Path:
        """Returns the orbit error path for a specific interferogram file."""
        if isinstance(ifg_path, str):
            ifg_path = Path(ifg_path)
        return Path(params[C.OUT_DIR], C.ORB_ERROR_DIR,
                    ifg_path.stem + '_' +
                    '_'.join([str(params[C.ORBITAL_FIT_METHOD]),
                              str(params[C.ORBITAL_FIT_DEGREE]),
                              str(params[C.ORBITAL_FIT_LOOKS_X]),
                              str(params[C.ORBITAL_FIT_LOOKS_Y])]) +
                    '_orbfit.npy')

    @staticmethod
    def dem_error_path(ifg_path: Union[str, Path], params) -> Path:
        """Returns the DEM error path for a specific interferogram file."""
        if isinstance(ifg_path, str):
            ifg_path = Path(ifg_path)
        return Path(params[C.OUT_DIR], C.DEM_ERROR_DIR,
                    ifg_path.stem + '_' + str(params[C.DE_PTHR]) + '_dem_error.npy')

    @staticmethod
    def aps_error_path(ifg_path: Union[str, Path], params) -> Path:
        """Returns the APS error path for a specific interferogram file."""
        if isinstance(ifg_path, str):
            ifg_path = Path(ifg_path)
        return Path(params[C.OUT_DIR], C.APS_ERROR_DIR,
                    ifg_path.stem + '_' +
                    '_'.join([str(x) for x in [
                        params[C.SLPF_CUTOFF],
                        params[C.SLPF_NANFILL],
                        params[C.SLPF_NANFILL_METHOD],
                        params[C.TLPF_CUTOFF],
                        params[C.TLPF_PTHR]
                    ]
                              ]) + '_aps_error.npy')

    def __str__(self):  # pragma: no cover
        value = ""
        if self.unwrapped_path is not None:
            value += """\nunwrapped_path = """ + self.unwrapped_path
        else:
            value += """\nunwrapped_path = None"""
        value += f"""
            converted_path = {self.converted_path}
            sampled_path = {self.sampled_path}
            tmp_sampled_path = {self.tmp_sampled_path}
            """
        return value


class IfgCropOption(IntEnum):
    """Method for cropping interferograms"""
    INTERSECTION = 1  # 1 = minimum overlapping area (intersection)
    UNION = 2         # 2 = maximum area (union)
    CUSTOM = 3        # 3 = customised area
    NONE = 4          # 4 = all ifgs already same size

class Configuration:
    """
    The main configuration class for PyRate, which allows access to the values of
    configurable properties.

    :param config_file_path: The path to the configuration text file to load

    :ivar outdir: The PyRate output directory
    """
    outdir: str
    refchipsize: int

    # general parameters:

    #: (0/1/2); The interferogram processor used (0==ROIPAC, 1==GAMMA, 2: GEOTIF)
    processor: int # TODO: Enum...

    # FIXME: Ideally PyRate should use `Path` objects exclusively... but this is a bigger task
    ref_pixel_file: Union[str, Path]
    orb_error_dir: Union[str, Path]
    dem_error_dir: Union[str, Path]
    aps_error_dir: Union[str, Path]
    phase_closure_dir: Union[str, Path]
    mst_dir: Union[str, Path]
    temp_mlooked_dir: Union[str, Path]
    coherence_dir: Union[str, Path]
    interferogram_dir: Union[str, Path]
    geometry_dir: Union[str, Path]
    timeseries_dir: Union[str, Path]
    velocity_dir: Union[str, Path]

    # Number of sigma to report velocity error
    velerror_nsig: int


    largetifs: bool
    # option to save numpy arrays
    savenpy: bool
    # optional to save of merged tsincr products (not sure why it's an int)
    savetsincr: int
    # sign convention for phase data
    signal_polarity: int
    # LOS projection
    los_projection: int  # FIXME: this is an enum...
    # Number of sigma to report velocity error
    velerror_nsig: int

    #: FLOAT; The no data value in the interferogram files.
    noDataValue: float
    # FIXME: We need to proxy this w/ a python-friendly name...
    noDataAveragingThreshold: float
    #: BOOL (0/1): Convert no data values to Nan
    nan_conversion: bool

    cohmask: bool

    # reference pixel parameters
    #: INT; Longitude (decimal degrees) of reference pixel, or if left blank a search will be performed
    refx: int
    refxfound: int
    #: INT; Latitude (decimal degrees) of reference pixel, or if left blank a search will be performed
    refy: int
    refyfound: int
    #: INT; Number of reference pixel grid search nodes in x dimension
    refnx: int
    #: INT; Number of reference pixel grid search nodes in y dimension
    refny: int
    #: INT; Dimension of reference pixel search window (in number of pixels)
    refchipsize: int
    #: FLOAT; Minimum fraction of observations required in search window for pixel to be
    # a viable reference pixel
    refminfrac: float
    #: INT (1/2); Reference phase estimation method
    # 1: median of the whole interferogram
    # 2: median within the window surrounding the reference pixel)
    refest: int

    #MAXVAR = 'maxvar'
    vcmt: object # FIXME: np.ndarray typing...
    preread_ifgs: dict
    tiles: List[Tile]
    rows: int
    cols: int

    # phase closure
    phase_closure: bool
    #CLOSURE_THR = 'closure_thr'
    #IFG_DROP_THR = 'ifg_drop_thr'
    #MIN_LOOPS_PER_IFG = 'min_loops_per_ifg'
    #MAX_LOOP_LENGTH = 'max_loop_length'
    #MAX_LOOP_REDUNDANCY = 'max_loop_redundancy'
    #SUBTRACT_MEDIAN = 'subtract_median'

    # FIXME: Some of the above aren't actually config paraeters, but mutable runtime state!!!!!
    # - I'm not sure when this behaviour started, but it's a bit dodgy (akin to passing global variables around)
    #
    # Examples: refxfound, refyfound, tilesm preread_ifgs

    # Time series parameters
    #: INT (1/2); Method for time series inversion (1: Laplacian Smoothing; 2: SVD)
    tsmethod: int  # FIXME: should be an enum
    #: INT; Number of required input observations per pixel for time series inversion
    ts_pthr: int
    #: INT (1/2); Order of Laplacian smoothing operator, first or second order
    smorder: int
    #: FLOAT; Laplacian smoothing factor (values used is 10**smfactor)
    smfactor: float

    # Stacking parameters
    #: FLOAT; Threshold ratio between 'model minus observation' residuals and a-priori observation
    # standard deviations for stacking estimate acceptance
    # (otherwise remove furthest outlier and re-iterate)
    nsig: float
    #: INT; Number of required observations per pixel for stacking to occur
    pthr: int
    #: FLOAT; Maximum allowable standard error for pixels in stacking
    maxsig: float

    # prepifg params: (TODO: Move into PrepIfgConfiguration?)

    ifgcropopt: int  # TODO: Ideally this would be IfgCropOption (requires a bit of refactoring)
    #: INT; Multi look factor for interferogram preparation in x dimension
    ifglksx: int
    #: INT; Multi look factor for interferogram preparation in y dimension
    ifglksy: int
    #: FLOAT; Coherence threshold for masking
    cohthresh: float
    #: FLOAT; Minimum longitude for cropping with method 3
    ifgxfirst: float
    #: FLOAT; Maximum longitude for cropping with method 3
    ifgyfirst: float
    #: FLOAT; Minimum latitude for cropping with method 3
    ifgxlast: float
    #: FLOAT; Maximum latitude for cropping with method 3
    ifgylast: float

    # orbital error correction/parameters
    #: BOOL (1/0); Perform orbital error correction (1: yes, 0: no)
    orbfit: bool
    #: INT (1/2); Method for orbital error correction (1: independent, 2: network)
    orbfitmethod: int  # FIXME: should be an enum...
    #: INT (1/2/3) Polynomial order of orbital error model
    # 1: planar in x and y - 2 parameter model
    # 2: quadratic in x and y - 5 parameter model
    # 3: quadratic in x and cubic in y - part-cubic 6 parameter model
    orbfitdegrees: int
    #: INT; Multi look factor for orbital error calculation in x dimension
    orbfitlksx: int
    #: INT; Multi look factor for orbital error calculation in y dimension
    orbfitlksy: int
    #: BOOL (1/0); Estimate interferogram offsets during orbit correction design matrix (1: yes, 0: no)
    orbfitoffset: bool
    #: FLOAT; Scaling parameter for orbital correction design matrix
    orbfitscale: float
    orbfitintercept: float

    # pylint: disable=invalid-name
    # JUSTIFICATION: these are long-established in the code already, out of scope to fix just yet.

    # TODO: Write documentation for each of the configuration options, at the very least refering
    # to some other place it's all documented (currently I'm not sure they are?)
    #
    # pylint: disable=missing-function-docstring
    # JUSTIFICATION: This is a whole job in and of itself, the configuration options need docs...
    # and in writing those docs, the typing info about them can be established / QA can be improved

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # JUSTIFICATION: cleaning up configuration class is out of scope currently, will probably be
    # tackled when things are restructured (shared.py and configuration.py are cluttered).

    def __init__(self, config_file_path: Union[str, Path]):
        # Promote config to path object
        if not isinstance(config_file_path, Path):
            config_file_path = Path(config_file_path)

        # Setup default values (in case they're not in the config file)
        self.parallel = False
        self.processes = 1
        self.cohfilelist = None
        self.basefilelist = None
        self.demfile = None

        # Load config file
        parser = ConfigParser()
        parser.optionxform = str
        # mimic header to fulfil the requirement for configparser
        parser.read_string("[root]\n" + config_file_path.read_text("utf-8"))

        for key, value in parser["root"].items():
            self.__dict__[key] = value

        # make output path, if not provided will error
        Path(self.outdir).mkdir(exist_ok=True, parents=True)

        # custom correct sequence if 'correct' section is provided in config
        if 'correct' in parser and 'steps' in parser['correct']:
            filtered_correct = filter(None, parser['correct'].get('steps').splitlines())
            self.__dict__['correct'] = list(filtered_correct)
        else:
            self.__dict__['correct'] = [
                'orbfit',
                'refphase',
                'demerror',
                'phase_closure',
                'mst',
                'apscorrect',
                'maxvar',
            ]

        # Validate required parameters exist.

        required = {k for k, v in PYRATE_DEFAULT_CONFIGURATION.items() if v['Required']}

        if not required.issubset(self.__dict__):  # pragma: no cover
            raise ValueError("Required configuration parameters: " + str(
                required.difference(self.__dict__)) + " are missing from input config file.")

        # handle control parameters
        for parameter_name in PYRATE_DEFAULT_CONFIGURATION:
            param_value = self.__dict__[parameter_name] if parameter_name in required or \
                                                           parameter_name in self.__dict__ else ''

            self.__dict__[parameter_name] = set_parameter_value(
                PYRATE_DEFAULT_CONFIGURATION[parameter_name]["DataType"],
                param_value,
                PYRATE_DEFAULT_CONFIGURATION[parameter_name]["DefaultValue"],
                PYRATE_DEFAULT_CONFIGURATION[parameter_name]["Required"],
                parameter_name)
            validate_parameter_value(parameter_name, self.__dict__[parameter_name],
                                     PYRATE_DEFAULT_CONFIGURATION[parameter_name]["MinValue"],
                                     PYRATE_DEFAULT_CONFIGURATION[parameter_name]["MaxValue"],
                                     PYRATE_DEFAULT_CONFIGURATION[parameter_name]["PossibleValues"])

        # bespoke parameter validation
        if self.refchipsize % 2 != 1:  # pragma: no cover
            if self.refchipsize - 1 > 1:
                # Configuration parameters refchipsize must be odd, values too large (>101)
                # will slow down the process without significant gains in results.
                self.refchipsize = self.refchipsize - 1

        # calculate rows and cols if not supplied
        if hasattr(self, 'rows') and hasattr(self, 'cols'):
            self.rows, self.cols = int(self.rows), int(self.cols)
        else:
            if NO_OF_PARALLEL_PROCESSES > 1:  # i.e. mpirun
                self.rows, self.cols = factorise_integer(NO_OF_PARALLEL_PROCESSES)
            else:
                if self.parallel:  # i.e. joblib parallelism
                    self.rows, self.cols = factorise_integer(self.processes)
                else:  # i.e. serial
                    self.rows, self.cols = 1, 1

        # create a temporary directory if not supplied
        if not hasattr(self, 'tmpdir'):
            self.tmpdir = Path(self.outdir).joinpath("tmpdir")
        else:
            self.tmpdir = Path(self.tmpdir)

        self.tmpdir.mkdir(parents=True, exist_ok=True)

        # create orbfit error dir
        self.orb_error_dir = Path(self.outdir).joinpath(ORB_ERROR_DIR)
        self.orb_error_dir.mkdir(parents=True, exist_ok=True)

        self.interferogram_dir = Path(self.outdir).joinpath(C.INTERFEROGRAM_DIR)
        self.interferogram_dir.mkdir(parents=True, exist_ok=True)

        # create DEM error dir
        self.dem_error_dir = Path(self.outdir).joinpath(DEM_ERROR_DIR)
        self.dem_error_dir.mkdir(parents=True, exist_ok=True)

        # create aps error dir
        self.aps_error_dir = Path(self.outdir).joinpath(C.APS_ERROR_DIR)
        self.aps_error_dir.mkdir(parents=True, exist_ok=True)

        # create mst dir
        self.mst_dir = Path(self.outdir).joinpath(C.MST_DIR)
        self.mst_dir.mkdir(parents=True, exist_ok=True)

        self.phase_closure_dir = Path(self.outdir).joinpath(C.PHASE_CLOSURE_DIR)
        self.phase_closure_dir.mkdir(parents=True, exist_ok=True)

        self.coherence_dir = Path(self.outdir).joinpath(C.COHERENCE_DIR)
        self.coherence_dir.mkdir(parents=True, exist_ok=True)

        self.geometry_dir = Path(self.outdir).joinpath(C.GEOMETRY_DIR)
        self.geometry_dir.mkdir(parents=True, exist_ok=True)

        self.timeseries_dir = Path(self.outdir).joinpath(C.TIMESERIES_DIR)
        self.timeseries_dir.mkdir(parents=True, exist_ok=True)

        self.velocity_dir = Path(self.outdir).joinpath(C.VELOCITY_DIR)
        self.velocity_dir.mkdir(parents=True, exist_ok=True)

        # create temp multilooked files dir
        self.temp_mlooked_dir = Path(self.outdir).joinpath(TEMP_MLOOKED_DIR)
        self.temp_mlooked_dir.mkdir(parents=True, exist_ok=True)

        self.ref_pixel_file = self.ref_pixel_path(self.__dict__)

        # var no longer used
        self.APS_ELEVATION_EXT = None
        self.APS_INCIDENCE_EXT = None
        self.apscorrect = 0
        self.apsmethod = 0
        self.elevationmap = None
        self.incidencemap = None

        # define parallel processes that will run
        self.NUMEXPR_MAX_THREADS = str(NO_OF_PARALLEL_PROCESSES)

        # Validate file names supplied in list exist and contain correct epochs in file names
        if self.cohfilelist is not None:
            # if self.processor != 0:  # not roipac
            validate_file_list_values(self.cohfilelist, 1)
            paths = self.__get_files_from_attr('cohfilelist', input_type=InputTypes.COH)
            self.coherence_file_paths = paths

        if self.basefilelist is not None:
            # if self.processor != 0:  # not roipac
            validate_file_list_values(self.basefilelist, 1)
            paths = self.__get_files_from_attr('basefilelist', input_type=InputTypes.BASE)
            self.baseline_file_paths = paths

        paths = self.__get_files_from_attr('hdrfilelist', input_type=InputTypes.HEADER)
        self.header_file_paths = paths

        paths = self.__get_files_from_attr('ifgfilelist')
        self.interferogram_files = paths

        self.dem_file = MultiplePaths(self.demfile, self.__dict__, input_type=InputTypes.DEM)

        # backward compatibility for string paths
        # pylint: disable=consider-using-dict-items
        for key in self.__dict__:
            if isinstance(self.__dict__[key], PurePath):
                self.__dict__[key] = str(self.__dict__[key])

    def __getitem__(self, key):
        """
        A simple item indexer which lets Configuration act like a dict.

        This is a TEMPORARY measure, to aid in weaning the PyRate code base off of passing `params`
        around in untyped `dict` objects, and looking them up at runtime via constant keys.
        """
        return self.__dict__[key]

    def __setitem__(self, key, val):
        """
        A simple item setter which lets Configuration act like a dict.

        This is a TEMPORARY measure, to aid in weaning the PyRate code base off of passing `params`
        around in untyped `dict` objects, and looking them up at runtime via constant keys.
        """
        self.__dict__[key] = val

    def items(self):
        """See __getitem__ / __setitem__ on why we're trying to mimic a dict"""
        return self.__dict__.items();

    @staticmethod
    def ref_pixel_path(params):
        return Path(params[C.OUT_DIR]).joinpath(
            '_'.join(
                [str(x) for x in [
                    'ref_pixel', params[C.REFX], params[C.REFY], params[
                        C.REFNX], params[C.REFNY],
                    params[C.REF_CHIP_SIZE], params[C.REF_MIN_FRAC], '.npy'
                ]
                 ]
            )
        )

    @staticmethod
    def mst_path(params, index) -> Path:
        return Path(params[C.OUT_DIR], C.MST_DIR).joinpath(f'mst_mat_{index}.npy')

    @staticmethod
    def preread_ifgs(params: dict) -> Path:
        return Path(params[C.TMPDIR], 'preread_ifgs.pk')

    @staticmethod
    def vcmt_path(params):
        return Path(params[C.OUT_DIR], C.VCMT).with_suffix('.npy')

    @staticmethod
    def phase_closure_filtered_ifgs_list(params):
        return Path(params[C.TEMP_MLOOKED_DIR]).joinpath('phase_closure_filtered_ifgs_list')

    def refresh_ifg_list(self, params):  # update params dict
        filtered_ifgs_list = self.phase_closure_filtered_ifgs_list(params)
        files = parse_namelist(filtered_ifgs_list.as_posix())
        params[C.INTERFEROGRAM_FILES] = [
            MultiplePaths(p, self.__dict__, input_type=InputTypes.IFG) for p in files
        ]
        return params

    @staticmethod
    def ref_phs_file(params):
        ref_pixel_path = Configuration.ref_pixel_path(params)
        # add ref pixel path as when ref pixel changes - ref phs path should also change
        return Path(params[C.OUT_DIR]).joinpath(
            ref_pixel_path.stem + '_' +
            '_'.join(['ref_phs', str(params[C.REF_EST_METHOD]), '.npy'])
        )

    @staticmethod
    def get_tiles(params):
        ifg_path = params[C.INTERFEROGRAM_FILES][0].sampled_path
        rows, cols = params['rows'], params['cols']
        return get_tiles(ifg_path, rows, cols)

    def __get_files_from_attr(self, attr, input_type=InputTypes.IFG):
        val = self.__getattribute__(attr)
        files = parse_namelist(val)
        return [MultiplePaths(p, self.__dict__, input_type=input_type) for p in files]

    def closure(self):
        closure_d = Path(self.phase_closure_dir)

        class Closure:
            """Just a simple data class"""
            def __init__(self):
                self.closure = closure_d.joinpath('closure.npy')
                self.ifgs_breach_count = closure_d.joinpath('ifgs_breach_count.npy')
                self.num_occurences_each_ifg = closure_d.joinpath('num_occurrences_each_ifg.npy')
                self.loops = closure_d.joinpath('loops.npy')

        return Closure()

    def coherence_stats(self):
        coh_d = Path(self.coherence_dir)
        keys = [ifg.COH_MEDIAN, ifg.COH_MEAN, ifg.COH_STD]
        return { k: coh_d.joinpath(k.lower() + '.tif').as_posix() for k in keys }

    def geometry_files(self):
        geom_dir = Path(self.geometry_dir)
        return {
            k: geom_dir.joinpath(k.lower() + '.tif').as_posix() for k in C.GEOMETRY_OUTPUT_TYPES
        }


def write_config_parser_file(conf: ConfigParser, output_conf_file: Union[str, Path]):
    """
    Writes a config to a config file. Similar to write_config_file but for ConfigParser.

    :param ConfigParser conf: The config parser instance to write to file.
    :param str output_conf_file: output file name
    """
    with open(output_conf_file, 'w', encoding="utf-8") as configfile:
        conf.write(configfile)


def write_config_file(params, output_conf_file):
    """
    Takes a param object and writes the config file. Reverse of get_conf_params.

    :param dict params: parameter dictionary
    :param str output_conf_file: output file name

    :return: config file
    :rtype: list
    """
    with open(output_conf_file, 'w', encoding="utf-8") as f:
        for key, val in params.items():
            if val is not None:
                if key == 'correct':
                    f.write(''.join(['[', key, ']', ':\t', '', '\n']))
                    f.write(''.join(['steps = ', '\n']))
                    for i in val:
                        f.write(''.join(['\t' + str(i), '\n']))
                elif isinstance(val, list):
                    continue
                else:
                    if isinstance(val, MultiplePaths):
                        if val.unwrapped_path is None:
                            path = val.converted_path
                        else:
                            path = val.unwrapped_path
                    else:
                        path = val
                    f.write(''.join([key, ':\t', str(path), '\n']))
            else:
                f.write(''.join([key, ':\t', '', '\n']))


def parse_namelist(nml):
    """
    Parses name list file into array of paths

    :param str nml: interferogram file list

    :return: list of interferogram file names
    :rtype: list
    """
    with open(nml, encoding="utf-8") as f_in:
        lines = [line.rstrip() for line in f_in]
    return filter(None, lines)


class ConfigException(Exception):
    """
    Default exception class for configuration errors.
    """
