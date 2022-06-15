from pathlib import Path
from pyrate.configuration import Configuration
import pytest
from pyrate import constants as C
from tests.phase_closure.common import IfgDummy
from tests.common import MEXICO_CROPA_DIR, TEST_CONF_ROIPAC


@pytest.fixture()
def closure_params(geotiffs):
    ifg_files = [IfgDummy(ifg_path) for ifg_path in geotiffs]
    config = Configuration(TEST_CONF_ROIPAC)
    config.interferogram_files = ifg_files
    config.max_loop_length = 100
    config['geotiffs'] = geotiffs

    return config


@pytest.fixture(scope='module')
def cropa_geotifs():
    tifs = [u.as_posix() for u in Path(MEXICO_CROPA_DIR).glob('*_unw.tif')]
    tifs.sort()
    return tifs
