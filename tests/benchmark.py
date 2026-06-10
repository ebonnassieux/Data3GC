import numpy as np
import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import pytest
import pathlib

# to generate random facets
import random

# for benching
import datetime

# to ignore annoying fits warnings
import warnings
from astropy.wcs.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)

from data3gc.sky import Sky as sky_xarray
from data3gc.sky_ndarray import Sky as Sky_ndarray

def bench_test(path=None,
               nfacets=0,
               nfacets_edit=0,
               benchname="",
               sky_type="xarray"):
    if path is None:
        return
    else:
        # randomly edit nfacets_edit out of the nfacets**2
        edit_facets=list(random.sample(range(0,nfacets**2),min(nfacets**2,nfacets_edit)))
        filename=path.absolute().as_posix()
        # start bench
        t0 = datetime.datetime.now()
        if sky_type=="xarray":
            bench_sky = sky_xarray.from_fits(filename,
                                        nfacets=nfacets,
                                        skyname=benchname)
        elif sky_type=="ndarray":
            bench_sky = Sky_ndarray.from_fits(filename,
                                        nfacets=nfacets,
                                        skyname=benchname)
        else:
            return ValueError
            
        if nfacets>0:
            keylist = list(bench_sky.facets.keys())
            
            for i, key in enumerate(keylist):
                if i in edit_facets:
                    this_facet_data = bench_sky.facets[key].data["restored"]
                    # add noise to this facet to show we can edit specific areas
                    noisevals = float(i+1)*np.random.normal(loc=0,
                                                        scale=0.001,
                                                        size=this_facet_data.shape)
                    bench_sky.facets[key].data["restored"] = this_facet_data + noisevals
            bench_sky.update_sky()
        # finish bench
        t1 = datetime.datetime.now()
        dt = (t1-t0).total_seconds()
        print(f'Bench : {sky_type:8s} | {benchname:8s} - nfacets: {nfacets:3d} - total time: {dt}')
              
#              %12s"%(sky_type+" | "+benchname), " - nfacets: %3i"%nfacets," - total time :",dt)
        del(bench_sky)

def bench_facets():
    # build list of fits file paths to iterate over
    fitslist=[pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-1.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-2.fits")]
    bench_name = ["400pix",
                  "800pix",
                  "1600pix"]
    
#    fitslist = [pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits")]
#    bench_name = ["400pix"]
    # build list of facets to iterate over
    nfacetslist=[0,3,5,11,21,31]
#    nfacetslist=[0]
    # launch bench
    for i, path in enumerate(fitslist):
        print()
        print("--------- %24s ---------"%(path.as_posix()))
        for nfacets in nfacetslist:
            bench_test(path,
                    nfacets,
                    nfacets_edit=5,
                    benchname=bench_name[i],
                    sky_type="xarray")
            bench_test(path,
                    nfacets,
                    nfacets_edit=5,
                    benchname=bench_name[i],
                    sky_type="ndarray")
        # print()
        # print("--------- %24s ---------"%(path.as_posix()))
        # for nfacets in nfacetslist:
        #     bench_test(path,
        #             nfacets,
        #             nfacets_edit=5,
        #             benchname=bench_name[i],
        #             sky_type="ndarray")


if __name__=="__main__":
    bench_facets()