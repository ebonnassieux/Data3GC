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
import xarray as xr

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
        t0 = time()
        if sky_type=="xarray":
            bench_sky = Sky_xarray.from_fits(filename,
                                        nfacets=nfacets,
                                        skyname=benchname)
        elif sky_type=="ndarray":
            bench_sky = Sky_xarray.from_fits(filename,
                                        nfacets=nfacets,
                                        skyname=benchname,
                                        datatype=np.ndarray)
        else:
            return ValueError
            
        if nfacets>0:
            keylist = list(bench_sky.facets.keys())
            
            for i, key in enumerate(keylist):
                if i in edit_facets:
                    this_facet_data = bench_sky.facets[key].data["restored"]
                    # add noise to this facet to show we can edit specific areas
                    # noisevals = float(i+1)*np.random.normal(loc=0,
                    #                                     scale=0.001,
                    #                                     size=this_facet_data.shape)
                    bench_sky.facets[key].data["restored"] = this_facet_data + float(i+1)#noisevals
            bench_sky.update_sky(update_facets=keylist)
        # finish bench
        t1 = time()
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
            bench_sky = Sky_xarray(skyname=skyname,
                       centrecoords=fits_centrecoords,
                       npix=fits_npix_x,
                       npix_y=fits_npix_y,
                       cellsize=fits_cellsize,
                       freqs=fits_freqs,
                       nfacets=nfacets,
                       stokes="I",
                       datatype=np.ndarray)
        else:
            return ValueError
        t2=time() # sky initialisation time
        initialisation_time = t2-t1
        # initialise WCS grids
        bench_sky.initWCSgrids()
        t3=time() # wcs initialisation time
        sky_wcs_init_time = t3-t2
        # initialise data
        if bench_sky.datatype is xr.DataArray:
            bench_sky.initdata_xarray()
        elif bench_sky.datatype is np.ndarray:
            bench_sky.initdata_ndarray()
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
            ### TODO: update sky only for edited facets
            bench_sky.update_sky(update_facets=edit_facets) #update_facets=editkeylist)
        t6 = time() # facet update time
        facet_update_time = t6-t5
        # finish bench
        total_runtime = t6-t0
        print(f'Bench : {sky_type:7s} | {benchname:4s} - nfacets: {nfacets:3d} - readfits: {fits_input_time:.4f} - skyinit: {initialisation_time:.4f} - skyWCSinit: {sky_wcs_init_time:.4f} - SkyDataInit: {sky_data_init_time:.4f} - FacetsInit: {facet_init_time:.4f} - FacetsUpdate: {facet_update_time:.4f} - total: {total_runtime:.2f}')
              
#              %12s"%(sky_type+" | "+benchname), " - nfacets: %3i"%nfacets," - total time :",dt)
        del(bench_sky)
        bench_times = [fits_input_time, initialisation_time, sky_wcs_init_time, sky_data_init_time, facet_init_time, facet_update_time, total_runtime]
        return bench_times

def bench_facets(): 
    # build list of fits file paths to iterate over
    fitslist=[pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-1.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-2.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-3.fits"),
              pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-4.fits"),
              pathlib.Path("/home/bonnassieux/Downloads/M31-lowres-LOFAR.fits")]
    bench_name = ["656K",
                  "2,5M",
                  "9,9M",
                  "40M",
                  "88M",
                  "141M - full LoTSS field"]
    # build list of facets to iterate over
    nfacetslist=[1,2,3,4,5,7,9,11]#,15,21,31]
    # initialise bench arrays
    ntests=None
    test_labels=[
        "fits_input_time", 
        "initialisation_time", 
        "sky_wcs_init_time", 
        "sky_data_init_time", 
        "facet_init_time", 
        "facet_update_time", 
        "total_runtime"
    ]
    xarray_all_benches=[]
    ndarray_all_benches=[]
    # launch bench

    for i, path in enumerate(fitslist):
        print()
        print("--------- %24s ---------"%(path.as_posix()))
        xarray_bench_times=[]
        ndarray_bench_times=[]
        for nfacets in nfacetslist:
            times=bench_test_detailed(path,
                    nfacets,
                    nfacets_edit=1,
                    benchname=bench_name[i],
                    sky_type="xarray")
            xarray_bench_times.append(times)
            ntests=None or len(times)
            times=bench_test_detailed(path,
                    nfacets,
                    nfacets_edit=nfacets,
                    benchname=bench_name[i],
                    sky_type="ndarray")
            ndarray_bench_times.append(times)
        xarray_all_benches.append(xarray_bench_times)
        ndarray_all_benches.append(ndarray_bench_times)
    
    xarray_all_benches = np.array(xarray_all_benches)
    ndarray_all_benches = np.array(ndarray_all_benches)

    nfacets = np.array(nfacetslist)**2


    for i in range(ntests):
        plt.subplots(figsize=(8,8))
        plt.title(test_labels[i])
        plt.xlabel("Nfacets")
        plt.ylabel("Runtime [s]")
        for bench_ind in range(len(fitslist)):
            plt.plot(nfacets,xarray_all_benches[bench_ind,:,i],label="xarray "+bench_name[bench_ind])
            plt.plot(nfacets,ndarray_all_benches[bench_ind,:,i],label="ndarray "+bench_name[bench_ind])
        plt.legend()
        ymin = 0.9*np.min(ndarray_all_benches[:,:,i])
        ymax = 1.2*np.max(xarray_all_benches[:,:,i])
        plt.ylim((ymin,ymax))
        plt.grid()
        plt.savefig(test_labels[i])
        print("Saved image as %s.png"%test_labels[i])
        plt.clf()


if __name__=="__main__":
    bench_facets()