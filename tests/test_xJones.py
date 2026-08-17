import numpy as np
#import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
#from astropy.io import fits
import matplotlib.pyplot as plt
import pytest
import pathlib
# for benching

from time import time

from data3gc.jones import xJones


def test_xJones_initialisation_DI():
    # initialise xJones object from function calls
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         antennas=np.array(["CS000",
                                            "CS001",
                                            "RS036",
                                            "FR606"]),
                         times=np.arange(10),
                         freqs=np.arange(200)*1e6+1e6,
                         msname="some_msfile.ms",
                         comments="test initialisation"
    )
    print(test_xjones)