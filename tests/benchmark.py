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
from time import time
from astropy.wcs import WCS
from astropy.io import fits

# to ignore annoying fits warnings
import warnings
from astropy.wcs.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)

from data3gc.sky import Sky as Sky_xarray
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


def bench_test_detailed(path=None,
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
        t0 = time()
        filepath = pathlib.Path(filename)
        if filepath.exists()==False:
            raise FileNotFoundError
        # set skyname to filename if none provided 
        skyname = filename
        hdul = fits.open(filename)
        hdu = hdul[0]
        # read data
        fits_data = hdu.data
        fits_wcs = WCS(header=hdu.header)
        # check for frequency
        for key in hdu.header.keys():
            if "unit" in str(key).lower():
                if "hz" in hdu.header[key].lower():
                    freqaxis = str(key.split("CUNIT")[1])
                    fits_freqs = hdu.header["CRVAL"+freqaxis]+np.arange(hdu.header["NAXIS"+freqaxis]*hdu.header["CDELT"+freqaxis])
                    fits_freqs *= u.Unit(hdu.header["CUNIT"+freqaxis])
        # check for radio-style shape
        if len(fits_data.shape)==4:
            fits_npix_x = fits_data.shape[2]
            fits_npix_y = fits_data.shape[3]
            fits_style="radio"
            # read centrecoords from fits
            centrepixcoord_x = round(fits_npix_x/2)
            centrepixcoord_y = round(fits_npix_y/2)
            centreskycoords = fits_wcs.pixel_to_world(centrepixcoord_x,centrepixcoord_y,0,0)
        else:
            fits_npix = max(fits_data.shape[0],fits_data.shape[1])
            fits_style="optical"
            # read centrecoords from fits
            centrepixcoord = round(fits_npix/2)
            centreskycoords = fits_wcs.pixel_to_world(centrepixcoord,centrepixcoord)
        fits_centrecoords = centreskycoords[0]
        # read cellsize
        fits_cellsize = np.abs(hdu.header["CDELT1"]) * u.Unit(hdu.header["CUNIT1"])
        # clean up before instantiating sky
        del(fits_data)
        hdul.close()
        t1=time() # input fits management time
        fits_input_time = t1-t0
        if sky_type=="xarray":
            bench_sky = Sky_xarray(skyname=skyname,
                       centrecoords=fits_centrecoords,
                       npix=fits_npix_x,
                       npix_y=fits_npix_y,
                       cellsize=fits_cellsize,
                       freqs=fits_freqs,
                       nfacets=nfacets,
                       stokes="I")
        elif sky_type=="ndarray":
            bench_sky = Sky_ndarray.from_fits(filename,
                                        nfacets=nfacets,
                                        skyname=benchname)
        else:
            return ValueError
        t2=time() # sky initialisation time
        initialisation_time = t2-t1
        # initialise WCS grids
        bench_sky.initWCSgrids()
        t3=time() # wcs initialisation time
        sky_wcs_init_time = t3-t2
        # initialise data
        bench_sky.initdata()
        t4=time()
        sky_data_init_time = t4-t3
        bench_sky.initfacets()
        t5=time()
        facet_init_time = t5-t4
            
        
            
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
        t6 = time() # facet update time
        facet_update_time = t6-t5
        # finish bench
        total_runtime = t6-t0
        print(f'Bench : {sky_type:8s} | {benchname:8s} - nfacets: {nfacets:3d} - readfits: {fits_input_time:.4f} - skyinit: {initialisation_time:.4f} - skyWCSinit: {sky_wcs_init_time:.4f} - SkyDataInit: {sky_data_init_time:.4f} - FacetsInit: {facet_init_time:.4f} - FacetsUpdate: {facet_update_time:.4f} - total: {total_runtime:.2f}')
              
#              %12s"%(sky_type+" | "+benchname), " - nfacets: %3i"%nfacets," - total time :",dt)
        del(bench_sky)

def bench_facets():
    # build list of fits file paths to iterate over
    fitslist=[pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-1.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-2.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-3.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-4.fits")]
    bench_name = ["656K",
                  "2,5M",
                  "9,9M",
                  "40M",
                  "88M"]
    
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
            bench_test_detailed(path,
                    nfacets,
                    nfacets_edit=nfacets**2,
                    benchname=bench_name[i],
                    sky_type="xarray")
            # bench_test(path,
            #         nfacets,
            #         nfacets_edit=nfacets,
            #         benchname=bench_name[i],
            #         sky_type="ndarray")
        # # print()
        # print("--------- %24s ---------"%(path.as_posix()))
        # for nfacets in nfacetslist:
        #     bench_test(path,
        #             nfacets,
        #             nfacets_edit=5,
        #             benchname=bench_name[i],
        #             sky_type="ndarray")


if __name__=="__main__":
    bench_facets()