import numpy as np
import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import pytest
import pathlib


from data3gc.sky import Sky


def test_sky_initialisation_nofacets():
    # initialise sky object from function calls
    test_sky = Sky(centrecoords=SkyCoord(45.*u.deg,45*u.deg,frame="fk5"),
                   npix=256,
                   cellsize=1*u.arcmin,
                   freqs=[60]*u.MHz,
                   nfacets=0,
                   stokes="I",
                   skyname="test_nenusky"
    )
    test_sky.close()


def test_sky_initialisation_facets():
    test_sky_5facets = Sky(centrecoords=SkyCoord(45.*u.deg,45*u.deg,frame="fk5"),
                   npix=256,
                   cellsize=1*u.arcmin,
                   freqs=[60]*u.MHz,
                   nfacets=3,
                   stokes="I",
                   skyname="test_nenusky"
    )
    test_sky_5facets.close()



def test_read_from_fits():
    # initialise the sky object from test fits file
    p = pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits")
    #p = pathlib.Path("tests/Data/M31-lowres-LOFAR.fits")
    
    filename=p.absolute().as_posix()
    test_sky_m31_cropped = Sky.from_fits(filename,
                                        nfacets=5,
                                        skyname="M31_cropped")
    print("Initialisation done, preparing show.")
    test_sky_m31_cropped.show(vmin=-0.0005,vmax=0.0015)
    test_sky_m31_cropped.close()
    del(test_sky_m31_cropped)


def test_full_functionality():
    # initialise the sky object from test fits file
    p = pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits")
    #p = pathlib.Path("tests/Data/M31-lowres-LOFAR.fits")
    filename=p.absolute().as_posix()
    test_sky_m31_cropped = Sky.from_fits(filename,
                                        nfacets=5,
                                        skyname="M31_cropped")
    print("Sky initialised from fits")
    # show full image
    test_sky_m31_cropped.show(vmin=-0.0005,vmax=0.0015)
    edit_facets = [0,7]#,11,18,23]
    
    # show before noise is added
    test_sky_m31_cropped.show(plot_facets=list(edit_facets),vmin=-0.0005,vmax=0.0015)
    print("Showed sky before noise added")
    keylist = list(test_sky_m31_cropped.facets.keys())#[edit_facets]
    for i, key in enumerate(keylist):
        if i in edit_facets:
            this_facet_data = test_sky_m31_cropped.facets[key].data["restored"].data
            # add noise to this facet to show we can edit specific areas
            noisevals = float(i+1)*np.random.normal(loc=0,
                                                scale=0.01,
                                                size=this_facet_data.shape)
            test_sky_m31_cropped.facets[key].data["restored"].values = 1.2*this_facet_data# + noisevals
    print("Added noise to specified facets")
    test_sky_m31_cropped.show(plot_facets=list(edit_facets),vmin=-0.0005,vmax=0.0015)
    # update sky with facet information
#    test_sky_m31_cropped.update_facets(datakey="restored",update_facets=edit_facets)
    test_sky_m31_cropped.update_sky(datakey="restored",update_facets=edit_facets)
    print("Updated sky")
    test_sky_m31_cropped.show(vmin=-0.0005,vmax=0.0015)
    # test serialisation
    test_sky_m31_cropped.write(basename="tests/serialisation_tests/test",
                            write_facets="all",
                            datakey="all",
                            verbose=True)
    print("Serialisation complete")

    # exit gracefully
    test_sky_m31_cropped.close()

if __name__=="__main__":
    # test initialisation from function call
#    test_sky_initialisation_nofacets()
#    test_sky_initialisation_facets()
#    test_sky()
    # test read from fits
#    test_read_from_fits()
    # test fits file open, facet initialisation + manipulation, then save
    test_full_functionality()
    