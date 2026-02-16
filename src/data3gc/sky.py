# class containing sky properties for 3GC
# core dependencies:
# numpy
# astropy
# matplotlib
# regions

import numpy as np
# astropy functions
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
# for diagnostic plots
import matplotlib.pyplot as plt
# for region functionalities
import regions


type StokesEigen = Literal['I','Q','U','V']

from typing import Literal
type Stokes = StokesEigen | tuple[StokesEigen] | tuple[StokesEigen, StokesEigen] | tuple[StokesEigen, StokesEigen, StokesEigen] | tuple[StokesEigen, StokesEigen, StokesEigen, StokesEigen]

class Sky:
    '''
    A class represenging the Sky that we want to work with.
    Attributes:
        centrecoords
        npix
        cellsize
        freqs
        nfacets
        stokes
        skyname
        facets
    '''
    def __str__(self):
        # figure out what the print for the function returns.
        # Name, coords, resolution, shape?
        print(self.phasecenter.to_string('hmsdms'))
        return f"Sky({self.name}, ({self.phasecenter.to_string('hmsdms')}), {self.nfacets} facets"

    def show(self,
             data=None | np.ndarray ,
             facets: list | str="all",
             vmin: float=-8.e-5,
             vmax: float=2.e-4
             ):
        # this should imshow the sky and its divisions.
        if data==None:
            data=self.restored.data
        data = fits.open("/home/ebonnassieux/OJ287_averaged_outer_uvcut_0.8arcsec-MFS-image.fits")[0].data
        # set up the plot axes etc
        fig, ax = plt.subplots(subplot_kw=dict(projection=self.gridwcs))
        ax.set(xlabel='Right Ascension', ylabel='Declination',title=self.name)
        if facets=="all":
            ax.imshow(data[0,0,:,:], vmin=vmin, vmax=vmax, origin='lower')
            #ax.grid(color='white', ls='solid')
            self.grid_reg.plot(ax=ax)
            for facet_reg in self.facet_grid_regs:
                facet_reg.plot(ax=ax)
        else:
            totalmask=np.zeros_like(data).astype(bool)
            for ifacet in facets:
                regmask = self.facet_sky_regs[ifacet].contains(wcs=self.gridwcs,skycoord=self.skycoords)
                totalmask = totalmask + regmask
                self.facet_grid_regs[ifacet].plot(ax=ax)
            self.grid_reg.plot(ax=ax)
            data = data*totalmask
            ax.imshow(data[0,0,:,:], vmin=vmin, vmax=vmax, origin='lower')
        self.grid_reg.plot(ax=ax)
        plt.show()

    ### constructor
    def __init__(self,
                 centrecoords : SkyCoord,
                 npix         : int,
                 cellsize     : u.Quantity,
                 freqs        : list[u.Quantity],
                 nfacets      : int,
                 stokes       : Stokes="I",
                 skyname      : str="Sky",
                 ):
        # initialise global sky variables
        self.name        = skyname
        self.phasecenter = centrecoords
        self.npix        = npix
        self.cellsize    = cellsize
        self.freqs       = freqs
        self.stokes      = stokes
        self.nfacets     = nfacets
        # initialise image grid variables
        # base shape on xradio schema 
        # https://github.com/casangi/xradio/blob/470-changes-needed-for-astroviper/docs/source/image_data/tutorials/image_schema_proposal.ipynb
        self.imshape = (len(freqs),len(stokes),npix, npix)      
        self.wcs     = WCS(self.wcs_input_dict())
        self.gridwcs = self.wcs.dropaxis(2).dropaxis(2)
        # initialise the data grids
        ### TODO: define these as shared xarrays?
        ### TODO: define specific shared-xarray image format?
        self.restored = self.ImageHDU("restored",
                                      np.zeros(self.imshape),
                                      self.wcs)
        self.residual = self.ImageHDU("residual",
                                      np.zeros(self.imshape),
                                      self.wcs)
        self.model    = self.ImageHDU("model",
                                      np.zeros(self.imshape),
                                      self.wcs)
        self.mask     = self.ImageHDU("mask",
                                      np.zeros(self.imshape),
                                      self.wcs)
        # Initialise the coordinate grids in ra, dec and l,m
        coordgrid = np.meshgrid(np.arange(self.npix),np.arange(self.npix))
        # drop the Stokes, Freq axes for this
        self.ras, self.decs = self.gridwcs.all_pix2world(coordgrid[0],coordgrid[1],1)*u.deg
        # defin RA, Dec and l,m coords
        self.skycoords = SkyCoord(self.ras,self.decs)
        self.l,self.m  = self.radec2lm_scalar(self.skycoords,self.phasecenter)
        # initialise regions
        self.sky_reg,self.grid_reg = self.region()
        # initialise facet properties
        self.facet_phasecenters = self.generate_facet_phasecenters()
        self.facet_npix         = self.generate_facet_npix()

        self.facet_sky_regs,self.facet_grid_regs = self.generate_facet_regions()


    def radec2lm_scalar(self,
                        coords      : SkyCoord,
                        phasecenter : None | SkyCoord = None,
                        ):
        # based on DDF function
        '''
        Docstring for radec2lm_scalar
        
        Function based on DDFacet to generate l,m coordinates from RA, Dec SkyCoord

        :param self: based on Sky class
        :param coords: input SkyCoord array object for the full grid 
        :type coords: SkyCoord
        :param phasecenter: Phase center from which to compute l,m values
        :type phasecenter: None | SkyCoord
        '''
        if phasecenter==None:
            phasecenter = self.phasecenter
        refra = self.phasecenter.ra.rad
        refdec = self.phasecenter.dec.rad
        ras    = coords.ra.rad
        decs   = coords.dec.rad
        l = np.cos(decs) * np.sin(ras - refra)
        m = np.sin(decs) * np.cos(refdec) - np.cos(decs) * np.sin(refra) * np.cos(ras - refra)
        return l,m
    
    def generate_facet_phasecenters(self):
        bin_edges   = np.linspace(0,self.npix,self.nfacets+1).astype(int)
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
        facet_centre_grid = np.meshgrid(bin_centers,bin_centers)
        facet_ras, facet_decs = self.gridwcs.all_pix2world(facet_centre_grid[0].ravel(),facet_centre_grid[1].ravel(),1)*u.deg
        return SkyCoord(facet_ras, facet_decs)
    
    def generate_facet_npix(self):
        bin_edges   = np.linspace(0,self.npix,self.nfacets+1).astype(int)
        bin_size    = int(np.mean(bin_edges[1:] - bin_edges[:-1]))
        return bin_size
    
    def generate_facet_regions(self):
        reg_visuals = {'color':"red",
                       'linewidth':1}
        facet_grid_regs = []
        facet_sky_regs  = []
        # returns region coverage of the sky. Used for overlap determinations.
        for i in range(len(self.facet_phasecenters)):
            sky_reg  = regions.RectangleSkyRegion(center=self.facet_phasecenters[i], 
                                                  width=self.cellsize*self.facet_npix, 
                                                  height=self.cellsize*self.facet_npix,
                                                  visual=reg_visuals
                                                  )
            grid_reg = sky_reg.to_pixel(self.gridwcs)
            facet_sky_regs.append(sky_reg)
            facet_grid_regs.append(grid_reg)

        return facet_sky_regs,facet_grid_regs

    def close(self):
        '''
        Docstring for close
        Close all HDU objects, free up virtual memory, exit gracefully.
        '''
        self.restored.close()
        self.residual.close()
        self.model.close()
        self.mask.close()

    def ImageHDU(self,
                 name:str,
                 data:np.ndarray,
                 wcs:WCS
                 ):
        '''
        Docstring for ImageHDU
        
        :param self: class Sky
        :param name: Name of the HDU, restored, residual, model etc
        :type name: str
        :param data: data to put in the HDU
        :type data: np.ndarray
        :param wcs: WCS of the HDU
        :type wcs: WCS
        '''
        this_hdu      = fits.PrimaryHDU(data:=data, 
                                        header=wcs.to_header())
        this_hdu.name = name
        return this_hdu

    def restoringbeam(self,
                      bmaj:u.Quantity,
                      bmin:u.Quantity,
                      PA:float=0):
        # restoring beam gaussian
        # call as a function to initialise on-the-fly
        self.bmaj=bmaj.rad
        self.bmin=bmin.rad
        self.PA=PA
        test=1  

    def region(self):
        reg_visuals = {'color':"blue",
                       'linewidth':5}
        # returns region coverage of the sky. Used for overlap determinations.
        sky_reg  = regions.RectangleSkyRegion(center=self.phasecenter, 
                                              width=self.cellsize*self.npix, 
                                              height=self.cellsize*self.npix,
                                              visual=reg_visuals
                                              )
        grid_reg = sky_reg.to_pixel(self.gridwcs)
        return sky_reg,grid_reg


    def JonesRegions(self,njonesdir):
        # generates tessels (Jones facets)
        test=1

    def wcs_input_dict(self):
        '''
        Docstring for wcs_input_dict
        This is the dictionary which defines the default WCS for our Sky object.
        
        :param self: Sky class
        '''
        # reference pixel is defined as centre of image, first freq, first stokes
        ref_pixels = [int(0.5*self.npix),
                      int(0.5*self.npix),
                      1.,
                      1.]
                
        ref_crvals = [self.phasecenter.ra.to(u.deg).value,
                      self.phasecenter.dec.to(u.deg).value,
                      self.freqs[0].value,
                      1.]
        ### based on wsclean header.
        ### TODO:
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

    def debugWCS(self):
        '''
        Docstring for debugWCS
        This is a function to test our WCS, for debug purposes
        :param self: Description
        '''
        self.centrecoords=SkyCoord(133.703625*u.deg,20.1085*u.deg,frame="fk5")
        self.npix=1000
        self.cellsize=0.1*u.arcsec
        self.imshape = (len(self.freqs),len(self.stokes),self.npix, self.npix)
        self.wcs  = WCS(self.wcs_input_dict())
        self.data = fits.open("/home/ebonnassieux/OJ287_averaged_outer_uvcut_0.8arcsec-MFS-image.fits")[0].data
        self.hdu = fits.PrimaryHDU(data=self.data, header=self.wcs.to_header())
        self.hdu.writeto('example.fits',overwrite=True)
        print("The two images should be exactly the same")
        print("dsm /home/ebonnassieux/OJ287_averaged_outer_uvcut_0.8arcsec-MFS-image.fits Data3GC/example.fits")


#    # define facets as sub-skies
#    class Facet(Sky):
#        # subfacet of the sky. Sky needs to know about its facets.
#        def __init__(self,CentreCoords,npix):
#            super().__init__(CentreCoords,gridsize,nfacets=1)

