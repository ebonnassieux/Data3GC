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

import pytest


def test_xJones_initialisation_DI():
    '''Check initialisation with default behaviour, no directions provided.'''
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
                         comments=np.array(["test DI initialisation"])
    )

def test_xJones_initialisation_DD():
    '''Check initialisation when direction axis is provided.'''
    # build 3 jones dirs
    coords = SkyCoord([0,30,45.]<<u.deg,[0,30,45.]<<u.deg,frame="fk5")
    c = np.array(SkyCoord(coords, frame="fk5", unit=(u.hourangle, u.deg)).to_string('hmsdms'))
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
                         comments=np.array(["test DD initialisation"])
    )

def test_validate_axis():
    '''Check axis recaster.'''
    ### first test - basic times are provided. recast them.
    sample_times = np.arange(10)
    reformatted_times = validate_axis(input_var=sample_times,
                                      desired_physical_type='time',
                                      output_unit=u.s)
    # pylance ouin-ouin jte jure
    if isinstance(reformatted_times, u.Quantity) \
        and reformatted_times.unit is not None :
        assert isinstance(reformatted_times,u.Quantity) # check that output is a quantity
        assert reformatted_times.unit.physical_type=='time' # check that physical type was properly assigned
        assert reformatted_times.unit==u.s # check that units are correct
    else:
        assert False
    ### second test - times are provided in units of hours. recast them.
    sample_times = np.arange(10)<<u.h
    reformatted_times = validate_axis(sample_times,'time',u.s)
    # pylance ouin-ouin jte jure
    if isinstance(reformatted_times, u.Quantity) \
        and reformatted_times.unit is not None :
        assert isinstance(reformatted_times,u.Quantity) # check that output is a quantity
        assert reformatted_times.unit.physical_type=='time' # check that physical type was properly assigned
        assert reformatted_times.unit==u.s # check that units are correct
    else:
        assert False
    ### third test - freqs are provided in units of MHz. recast them.
    sample_freqs = (np.arange(20)+110)<<u.MHz # lofar-style
    reformatted_freqs = validate_axis(sample_freqs,'freq',u.Hz)
    # pylance ouin-ouin jte jure
    if isinstance(reformatted_freqs, u.Quantity) \
        and reformatted_freqs.unit is not None :
        assert isinstance(reformatted_freqs,u.Quantity) # check that output is a quantity
        assert reformatted_freqs.unit.physical_type=='frequency' # check that physical type was properly assigned
        assert reformatted_freqs.unit==u.Hz # check that units are correct
    else:
        assert False

def test_add_xjones_coords():
    '''Check function to add requested coords to xjones dataset.'''
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         antennas=np.array(["CS000",
                                            "CS001"]),
                         times=np.arange(10)*u.s,
                         freqs=(np.arange(10)*1e6+120e6)<<u.Hz,
                         params=np.array(["I"]),
                         msname="some_msfile.ms",
                         comments=np.array(["test adding a coordinate axis"])
    )
    test_xjones.add_coords("gains_t0",
                           np.arange(10)<<u.s,
                           physical_type='time',
                           unit=u.s)
    assert "gains_t0" in test_xjones.gains._coord_names

def test_del_xjones_coords():
    '''Check function to remove requested coords from xjones dataset.'''
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         antennas=np.array(["CS000",
                                            "CS001"]),
                         times=np.arange(10)*u.s,
                         freqs=(np.arange(10)*1e6+120e6)<<u.Hz,
                         params=np.array(["I"]),
                         msname="some_msfile.ms",
                         comments=np.array(["test remove a coordinate axis"])
    )
    test_xjones.del_coords("Freqs")
    assert "Freqs" not in test_xjones.gains._coord_names

def test_add_xjones_attr():
    '''Check function to add requested attribute from xjones dataset.'''
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         antennas=np.array(["CS000",
                                            "CS001"]),
                         times=np.arange(10)*u.s,
                         freqs=(np.arange(10)*1e6+120e6)<<u.Hz,
                         params=np.array(["I"]),
                         msname="some_msfile.ms",
                         comments=np.array(["test adding an attribute axis"])
    )
    test_xjones.add_attr("test_attr","OK")
    assert "test_attr" in test_xjones.gains.attrs

def test_del_xjones_attr():
    '''Check function to remove requested attribute from xjones dataset.'''
    test_xjones = xJones(name='test_gain',
                         gaintype='scalar',
                         antennas=np.array(["CS000",
                                            "CS001"]),
                         times=np.arange(10)*u.s,
                         freqs=(np.arange(10)*1e6+120e6)<<u.Hz,
                         params=np.array(["I"]),
                         msname="some_msfile.ms",
                         comments=np.array(["test removing an attribute"])
    )
    test_xjones.del_attr("msname")
    assert "msname" not in test_xjones.gains.attrs
    
def test_from_dicosols():
    '''Check initialisation from dicosols object.'''
    sols = "./tests/Data/killMS.CohJones.sols.npz"
    test_xjones = xJones.from_dicosols(sols)
    
def test_to_dicosols_from_dicosols_with_missing_axes():
    '''Check that serialisation to dicosols object raises the correct errors when necessary coordinates are missing.'''
    sols = "./tests/Data/killMS.CohJones.sols.npz"
    test_xjones = xJones.from_dicosols(sols)
    test_xjones.del_coords("Times")
    from data3gc.jones import MissingDicoSolsKey
    with pytest.raises(MissingDicoSolsKey):
        test_xjones.to_dicosols('./tests/Data/test-sols-serialisation.npz')

def test_to_dicosols_from_dicosols():
    '''Test serialisation to dicosols object.'''
    sols = "./tests/Data/killMS.CohJones.sols.npz"
    orig = dict(np.load(sols,allow_pickle=True))
    test_xjones = xJones.from_dicosols(sols)
    testsols_name = './tests/Data/test-sols-serialisation.npz'
    new = test_xjones.to_dicosols(testsols_name,
                                    return_dico=True,
                                    write_to_file=True)
    # let type-checker know that both outputs are loaded
    assert orig is not None
    assert new is not None
    # Compare keys
    assert set(orig.keys()) == set(new.keys()), "Different keys in npz files"

    # Compare each array
    for key in orig:
        a, b = orig[key], new[key]
        if isinstance(a, np.ndarray):
            if a.dtype.names:  # Structured array
                for field in a.dtype.names:
                    assert a[field].shape == b[field].shape, f"{key}.{field} shape mismatch"
                    if np.issubdtype(a[field].dtype, np.number):
                        np.testing.assert_array_almost_equal(a[field], b[field])
                    else:
                        np.testing.assert_array_equal(a[field], b[field])
            else:
                assert np.array_equal(a,b), f"{key} differs: {a} != {b}"
        else:
            assert a == b, f"{key} differs: {a} != {b}"
    # compare the files
    ### all the below fail. ask about it.
    # import os
    # # SHA-256
    # os.system(f"sha256sum {sols} {testsols_name}")
    # # MD5
    # os.system(f"md5sum {sols} {testsols_name}")
    # # cmp for simple binary comparison
    # os.system(f"cmp {sols} {testsols_name}")



def from_json():
    ...

def test_serialise_json():
    sols = "./tests/Data/killMS.CohJones.sols.npz"
    test_xjones = xJones.from_dicosols(sols)
    test_xjones.to_json("test.json",include_data=True)



def from_h5parm():
    ...

def test_serialise_h5parm():
    ...

