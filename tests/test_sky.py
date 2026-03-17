import numpy as np
import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib.pyplot as plt
import pytest

from data3gc.sky import Sky

test_sky = Sky(skyname="OJ287",
               centrecoords=SkyCoord(133.703625*u.deg,20.1085*u.deg,frame="fk5"),
               npix=1000,
               cellsize=0.1*u.arcsec,
               freqs=[144e6*u.MHz],
               nfacets=5,
               stokes="I"
)

import random

show_facets = random.sample(range(0,test_sky.nfacets**2),15)

#show_facets = [4]

test_sky.show()

test_sky.show(plot_facets=list(show_facets))

#test_sky.close()
