import numpy as np
#import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
#from astropy.io import fits
import matplotlib.pyplot as plt
import pathlib
# for benching

from time import time

from data3gc.jones import xJones, validate_axis


def test_xJones_initialisation_DI():
    # initialise xJones object from function calls
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         antennas=np.array(["CS000",
                                            "CS001",
                                            "RS036",
                                            "FR606"]),
                         times=np.arange(10)*u.s,
                         freqs=(np.arange(40)*1e6+120e6)<<u.Hz,
                         params=np.array(["XX","YY"]),
                         msname="some_msfile.ms",
                         comments=np.array(["test initialisation"])
    )

def test_xJones_initialisation_DD():
    # build 3 jones dirs
    coords = SkyCoord([0,30,45.]<<u.deg,[0,30,45.]<<u.deg,frame="fk5")
    c = SkyCoord(coords, frame="fk5", unit=(u.hourangle, u.deg)).to_string('hmsdms')
    # initialise xJones object from function calls
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         directions=c,
                         antennas=np.array(["CS000",
                                            "CS001",
                                            "RS036",
                                            "FR606"]),
                         times=np.arange(10)*u.s,
                         freqs=(np.arange(40)*1e6+120e6)<<u.Hz,
                         params=np.array(["XX","YY"]),
                         msname="some_msfile.ms",
                         comments=np.array(["test initialisation"])
    )

def test_validate_axis():
    ### first test - basic times are provided. recast them.
    sample_times = np.arange(10)
    reformatted_times = validate_axis(input_var=sample_times,
                                      desired_physical_type='time',
                                      output_unit=u.s)
    assert isinstance(reformatted_times,u.Quantity) # check that output is a quantity
    assert reformatted_times.unit.physical_type=='time' # check that physical type was properly assigned
    assert reformatted_times.unit==u.s # check that units are correct
    ### second test - times are provided in units of hours. recast them.
    sample_times = np.arange(10)<<u.h
    reformatted_times = validate_axis(sample_times,'time',u.s)
    assert isinstance(reformatted_times,u.Quantity) # check that output is a quantity
    assert reformatted_times.unit.physical_type=='time' # check that physical type was properly assigned
    assert reformatted_times.unit==u.s # check that units are correct
    ### third test - freqs are provided in units of MHz. recast them.
    sample_freqs = (np.arange(20)+110)<<u.MHz # lofar-style
    reformatted_freqs = validate_axis(sample_freqs,'freq',u.Hz)
    assert isinstance(reformatted_freqs,u.Quantity) # check that output is a quantity
    assert reformatted_freqs.unit.physical_type=='frequency' # check that physical type was properly assigned
    assert reformatted_freqs.unit==u.Hz # check that units are correct



    ...