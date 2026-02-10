# class containing sky properties for 3GC
# core dependencies:
# numpy
# regions
# astropy
# python-casacore
# matplotlib

import numpy as np
import regions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
from casacore.images import image
### needed due to way we construct WCS
import os

class Sky:
    ### constructor
    def __init__(self,
                 skyname: str,
                 centrecoords: SkyCoord,
                 npix: int,
                 cellsize: u.Quantity,
                 freqs: list[u.Quantity],
                 nfacets:int,
                 stokes: list[str]="I"
                 ):
        # initialise global sky variables
        self.name = skyname
        self.phasecenter=centrecoords
        self.npix = npix
        self.cellsize = cellsize
        self.freqs = freqs
        self.stokes = stokes
        self.nfacets = nfacets
        # initialise image grid variables
        # base shape on xradio schema 
        # https://github.com/casangi/xradio/blob/470-changes-needed-for-astroviper/docs/source/image_data/tutorials/image_schema_proposal.ipynb
        self.imshape = (len(freqs),len(stokes),npix, npix)
        self.coords  = np.meshgrid(1,1) # TODO get RA, Dec coordinate array explicitly here.        
        self.wcs  = WCS(self.wcs_input_dict())
#        print(self.wcs)
#        stop
        self.data = np.zeros(self.imshape)
        self.data = fits.open("/home/ebonnassieux/OJ287_averaged_outer_uvcut_0.8arcsec-MFS-image.fits")[0].data
        self.hdu = fits.PrimaryHDU(data=self.data, header=self.wcs.to_header())
        self.hdu.writeto('example.fits',overwrite=True)

        stop

#        self.cell = [cellsize, cellsize]
#        self.imdata = self.CasaImageInitialise("data")

    def wcs_input_dict(self):
        # reference pixel is defined as edge of image, first freq, first stokes
        ref_pixels = [int(0.5*self.npix)+1,
                      int(0.5*self.npix)+1,
                      1.,
                      1.]
                
        ref_crvals = [self.phasecenter.ra.to(u.deg).value,
                      self.phasecenter.dec.to(u.deg).value,
                      self.freqs[0].value,
                      1.]

        ### based on wsclean header.
        ### TODO:
        # check if header SIMPLE should be set to True
        # check if header EXTEND should be set to True
        # check how to encode bandwidth properly...
        cdelt_deg = self.cellsize.to(u.deg).value
        wcs_input_dict = {
            "SIMPLE" : "T",
            "BITPIX" : -32,
            "NAXIS"  : 4,
            "NAXIS1" : self.npix,
            "NAXIS2" : self.npix,
            "NAXIS3" : len(self.freqs),
            "NAXIS4" : len(self.stokes),
            "WCSAXES": 4,
            "EXTEND" : "T",
            "BSCALE" : "1",
            "CTYPE1" : "RA---SIN",
            "CRPIX1" : ref_pixels[0],
            "CRVAL1" : ref_crvals[0],
            "CDELT1" : -cdelt_deg,
            "CUNIT1" : "deg",
            "CTYPE2" : "DEC--SIN",
            "CRPIX2" : ref_pixels[1],
            "CRVAL2" : ref_crvals[1],
            "NAXIS2" : self.npix,
            "CDELT2" : cdelt_deg,
            "CUNIT2" : "deg",
            "CTYPE3" : 'FREQ',
            "CRPIX3" : ref_pixels[2],
            "CRVAL3" : ref_crvals[2],
            "CDELT3" : (self.freqs[-1].value - self.freqs[0].value) / (len(self.freqs) - 1) if len(self.freqs) > 1 else 1.0,
            "CUNIT3" : "Hz",
            "CTYPE4" : 'STOKES',
            "CRPIX4" : ref_pixels[3],
            "CRVAL4" : ref_crvals[3],
            "CDELT4" :  1.0,
            "CUNIT4" : '',
            "BTYPE"  : 'Intensity',                                                           
            "BUNIT"  : 'Jy/beam ',                                                          
            "SPECSYS": 'TOPOCENT',
            "RADESYS": 'ICRS'
        }
        return wcs_input_dict

    def CasaImageInitialise(self,imagename):
        ### TODO: remove dependency on casa image. Looks very inefficient + source of errors + extra dependency.
        tmpIm = image(imagename=self.name,
                    shape=self.imshape)#, cell=self.cell, freq=self.freqs, stokes=self.stokes)
        self.coords = tmpIm.coordinates()
        del(tmpIm)
        os.system("rm -Rf %s"%self.name)
        # set coordinate increments to radians
        increments = self.coords.get_increment()
        # default casacore units are arcmin.
        # TODO: do we need to divide the below by 60?
        increments[-1]=[self.cellsize.to(u.radian),-self.cellsize.to(u.radian)] 
        # default casacore units are arcmin. 
        RefVal=self.coords.get_referencevalue()
        RefVal[-1] = [self.phasecenter.ra.to(u.arcmin).value,
                      self.phasecenter.dec.to(u.arcmin).value]
        if self.freqs is not None:
            RefVal[0]=self.freqs[0].value
#            ich,ipol,xy=
            print(self.coords.get_referencepixel())
#            print(ich,ipol,xy)
            stop
            ich=0
            c.set_referencepixel((ich,ipol,xy))

    def create_image(self):
        # Define the image parameters
        shape = [self.npix, self.npix]  
        cell = [self.cellsize, self.cellsize]
        freq = self.freqs
        stokes = self.stokes

        # Create the image
        im = image.Image(shape=shape, cell=cell, freq=freq, stokes=stokes)

        # Save the image
        im.tofile(self.skyname + '.im')

        return im




    def __str__(self):
        test=1

    def show(self):
        # this should imshow the sky and its divisions.
        print(self.name)

    def residuals(self):
        # residual data grid.
        test=1
    
    def model(self):
        # sky model data grid.
        test=1

    def restoringbeam(self,bmaj,bmin,PA):
        # restoring beam gaussian
        self.bmaj=bmaj
        self.bmin=bmin
        self.PA=PA
        test=1

    def coords(self):
        # coordinate grid
        test=1

    def mask(self):
        # clean mask
        test=1

    def region(self):
        # returns region coverage of the sky. Used for overlap determinations.
        test=1

    def write(self,filename):
        # write sky to file
        test=1

    def JonesRegions(self,njonesdir):
        # generates tessels (Jones facets)
        test=1

#    # define facets as sub-skies
#    class Facet(Sky):
#        # subfacet of the sky. Sky needs to know about its facets.
#        def __init__(self,CentreCoords,npix):
#            super().__init__(CentreCoords,gridsize,nfacets=1)

