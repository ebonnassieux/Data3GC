import numpy as np
import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import pytest
import pathlib


from data3gc.sky_ndarray import Sky

# initialise the sky object from test fits file
p = pathlib.Path("/home/bonnassieux/Downloads/M31-lowres-LOFAR.fits")
filename=p.absolute().as_posix()
test_sky = Sky.from_fits(filename,
                         nfacets=0)



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
import random
#show_facets = random.sample(range(0,test_sky.nfacets**2),15)
#test_sky.show(plot_facets=list(show_facets))

# edit a facet and overplot it
edit_facets = [0,1,]
keylist = list(test_sky.facets.keys())#[edit_facets]
for i, key in enumerate(keylist):
    if i in edit_facets:
        this_facet_data = test_sky.facets[key].data["restored"]
        # add noise to this facet to show we can edit specific areas
        noisevals = float(i+1)*np.random.normal(loc=0,
                                            scale=0.01,
                                            size=this_facet_data.shape)
        test_sky.facets[key].data["restored"] = this_facet_data + noisevals
#test_sky.show(plot_facets=list(edit_facets))
# update sky with facet information
test_sky.update()
test_sky.show()


# exit gracefully
test_sky.close()
