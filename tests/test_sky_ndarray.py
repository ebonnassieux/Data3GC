import numpy as np
import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import pytest
import pathlib
# to plot random facets
import random


from data3gc.sky_ndarray import Sky



def test_sky_initialisation():
    # initialise sky object from function calls
    test_sky = Sky(centrecoords=SkyCoord(45.*u.deg,45*u.deg,frame="fk5"),
                   npix=256,
                   cellsize=1*u.arcmin,
                   freqs=[60*u.MHz],
                   nfacets=11,
                   stokes="I",
                   skyname="test_nenusky"
    )
    test_sky.close()

    test_sky_5facets = Sky(centrecoords=SkyCoord(45.*u.deg,45*u.deg,frame="fk5"),
                   npix=256,
                   cellsize=1*u.arcmin,
                   freqs=[60*u.MHz],
                   nfacets=5,
                   stokes="I",
                   skyname="test_nenusky"
    )
    test_sky_5facets.close()

def test_read_from_fits():
    # initialise the sky object from test fits file
    p = pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped.fits")
    filename=p.absolute().as_posix()
    test_sky_m31_cropped = Sky.from_fits(filename,
                                        nfacets=11,
                                        skyname="M31_cropped")
    test_sky_m31_cropped.close()
    del(test_sky_m31_cropped)



def test_full_functionality():
    # initialise the sky object from test fits file
    p = pathlib.Path("tests/Data/M31-lowres-LOFAR-cropped-1.fits")
    #p = pathlib.Path("/home/bonnassieux/Downloads/M31-lowres-LOFAR.fits")
    filename=p.absolute().as_posix()
    test_sky_m31_cropped = Sky.from_fits(filename,
                                        nfacets=11,
                                        skyname="M31_cropped")
    # show full image
#    test_sky_m31_cropped.show(vmin=-0.0005,vmax=0.0015)
    edit_facets = [0,1,3,8,11,18,23]
    # show before noise is added
#    test_sky_m31_cropped.show(plot_facets=list(edit_facets),vmin=-0.0005,vmax=0.0015)
    keylist = list(test_sky_m31_cropped.facets.keys())#[edit_facets]
    for i, key in enumerate(keylist):
        if i in edit_facets:
            this_facet_data = test_sky_m31_cropped.facets[key].data["restored"]
            # add noise to this facet to show we can edit specific areas
            noisevals = float(i+1)*np.random.normal(loc=0,
                                                scale=0.01,
                                                size=this_facet_data.shape)
            test_sky_m31_cropped.facets[key].data["restored"] = this_facet_data + noisevals
#    test_sky_m31_cropped.show(plot_facets=list(edit_facets),vmin=-0.0005,vmax=0.0015)
    # update sky with facet information
    test_sky_m31_cropped.update_sky()
#    test_sky_m31_cropped.show(vmin=-0.0005,vmax=0.0015)
    # test serialisation
    # test_sky_m31_cropped.write(basename="tests/serialisation_tests/test",
    #                         write_facets="all",
    #                         datakey="all",
    #                         verbose=True)

    # exit gracefully
    test_sky_m31_cropped.close()





# # initialise the sky object from variables directly
# test_sky = Sky(skyname="OJ287",
#                centrecoords=SkyCoord(133.703625*u.deg,20.1085*u.deg,frame="fk5"),
#                npix=1000,
#                cellsize=0.1*u.arcsec,
#                freqs=[144e6*u.MHz],
#                nfacets=3,
#                stokes="I"
# )

# visualise full sky object
#test_sky.show()

# plot random facets
#show_facets = random.sample(range(0,test_sky.nfacets**2),15)
#test_sky.show(plot_facets=list(show_facets))

# # edit a facet and overplot it
# edit_facets = [0,1,]
# keylist = list(test_sky.facets.keys())#[edit_facets]
# for i, key in enumerate(keylist):
#     if i in edit_facets:
#         this_facet_data = test_sky.facets[key].data["restored"]
#         # add noise to this facet to show we can edit specific areas
#         noisevals = float(i+1)*np.random.normal(loc=0,
#                                             scale=0.01,
#                                             size=this_facet_data.shape)
#         test_sky.facets[key].data["restored"] = this_facet_data + noisevals
# #test_sky.show(plot_facets=list(edit_facets))
# # update sky with facet information
# test_sky.update()
# test_sky.show(vmin=-0.0005,vmax=0.0015)


# # exit gracefully
# test_sky.close()

if __name__=="__main__":
    # test initialisation from function call
#    test_sky_initialisation()
    # test read from fits
#    test_read_from_fits()
    # test fits file open, facet initialisation + manipulation, then save
    test_full_functionality()
    