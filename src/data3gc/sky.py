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

### functionalities to add:
###  - !!! read/write !!! super important
###  - generate Tessel layer for backwards compatibility
###  - group facets by Voronoi tesselation of Model data slice
###  - multisky (join Sky objects, show becomes healpix?)

### below is experimental for Stokes. over-engineering in practice, but was a personal test.
type StokesEigen = Literal['I','Q','U','V']
from typing import Literal
type Stokes = StokesEigen | tuple[StokesEigen] | tuple[StokesEigen, StokesEigen] | tuple[StokesEigen, StokesEigen, StokesEigen] | tuple[StokesEigen, StokesEigen, StokesEigen, StokesEigen]
### TODO; below, change stokes back to a string of which we look for IQUV subsets...

class Sky:
    '''
    A class represenging the Sky that we want to work with.
    Sky attributes:
        :param centrecoords: input SkyCoord for this sky's center coordinates 
        :type centrecoords: SkyCoord
        :param npix: size of the sky grid, defined as a square 
        :type npix: int
        :param cellsize: angular size of individual pixels, in angles (e.g. arcsec)
        :type cellsize: u.Quantity
        :param freqs: List of frequencies for each spectral cube slice
        :type freqs: list[u.Quantity]
        :param nfacets: Create sub-skies (facets) defined automatically as a square grid. nfacets defines how many facets to create along 1 axis; a total of nfacets^2 facets are created for the sky. 
        :type nfacets: int
        :param stokes: Stokes parameters of the sky, which can be one or more values of I,Q,U,V.
        :type stokes: Stokes="I"
        :param skyname: Name given to this patch of sky. 
        :type skyname: str="Sky"
    If nfacets!=0, facets are an important attribute generated.
        facets
    Facet attributes:
        ...
    '''
    def __str__(self):
        # figure out what the print for the function returns.
        # Name, coords, resolution, shape?
        print(self.phasecenter.to_string('hmsdms'))
        return f"Sky({self.name}, ({self.phasecenter.to_string('hmsdms')}), {self.nfacets} facets"

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
        self.initdata()

#        ### DEBUG
#        self.data["restored"] = fits.open("/home/ebonnassieux/OJ287_averaged_outer_uvcut_0.8arcsec-MFS-image.fits")[0].data

        # Initialise the coordinate grids in ra, dec and l,m
        coordgrid = np.meshgrid(np.arange(self.npix),np.arange(self.npix))
        # drop the Stokes, Freq axes for this
        self.ras, self.decs = self.gridwcs.all_pix2world(coordgrid[0],coordgrid[1],1)*u.deg
        # defin RA, Dec and l,m coords
        self.skycoords = SkyCoord(self.ras,self.decs)
        self.l,self.m  = self.radec2lm_scalar(self.skycoords,self.phasecenter)
        # define region visuals for sky and facet
        self.sky_visuals   = {'color':"blue",
                            'linewidth':5}
        self.facet_visuals = {'color':"red",
                              'linewidth':1}
        # initialise regions
        self.sky_reg,self.grid_reg = self.region(self.sky_visuals)
        if self.nfacets!=0:
            # initialise facet properties
            self.set_facet_pixgrid()
            self.facet_sky_regs,self.facet_grid_regs = self.generate_facet_regions()
            self.facets={}
            for facet_index in range(len(self.facet_grid_regs)):
                facet_phasecenter = self.facet_phasecenters[facet_index]
                print(facet_phasecenter)
                facetname = facet_phasecenter.to_string('hmsdms')
                verts = self.facetvertices[facet_index]
                xmin,xmax = int(np.min(verts.x)),int(np.max(verts.x))
                ymin,ymax = int(np.min(verts.y)),int(np.max(verts.y))
                xlen = xmax-xmin
                ylen = ymax-ymin
                facetnpix=max(xlen,ylen)
                self.facets[facetname]=Sky(skyname=facetname,
                                        centrecoords=facet_phasecenter,
                                        npix=facetnpix,
                                        cellsize=self.cellsize,
                                        freqs=self.freqs,
                                        nfacets=0,
                                        stokes=self.stokes
                )
                ### TODO: set up so we do the below only once...
                # reinitialise regions (update visuals, set gridreg to sky wcs)
                self.facets[facetname].sky_reg =  self.facet_sky_regs[facet_index]
                self.facets[facetname].grid_reg =  self.facet_grid_regs[facet_index]
                # define facet grid properties
                self.facets[facetname].imshape = (len(freqs),len(stokes),xlen, ylen)
                self.facets[facetname].wcs     = WCS(self.facets[facetname].wcs_input_dict())
                self.facets[facetname].gridwcs = self.facets[facetname].wcs.dropaxis(2).dropaxis(2)
                self.facets[facetname].initdata()
                # generate mask from vertices.
                verts = self.facetvertices[facet_index]
                xmin,xmax = int(np.min(verts.x)),int(np.max(verts.x))
                ymin,ymax = int(np.min(verts.y)),int(np.max(verts.y))
                sky_to_facet_regmask = np.zeros(self.imshape).astype(bool)
                sky_to_facet_regmask[:,:,ymin:ymax,xmin:xmax]=True
                # fill data
                for key in self.data.keys():
                    self.facets[facetname].data[key] = self.data[key][sky_to_facet_regmask].reshape(self.facets[facetname].imshape)

    ### data initialiser functions
    def returnHDU(self,
               hduname:str):
        thishdu = self.ImageHDU(hduname,
                        np.zeros(self.imshape),
                        self.wcs)
        return thishdu
    
    def initdata(self):
        '''
        Creates and populations data dictionary.
        '''
        ### TODO: define these as shared xarrays?
        ### TODO: define specific shared-xarray image format?
        ### TODO: figure out if we want these to be HDUS? Probably not actually...
        self.data={}
        keys = {"dirty",
                "restored",
                "residual",
                "model",
                "mask",
                "beam"}
        for key in keys:
            self.data[key] = np.zeros(self.imshape)

    def radec2lm_scalar(self,
                        coords      : SkyCoord,
                        phasecenter : None | SkyCoord = None,
                        ):
        # based on DDF function
        '''
        Docstring for radec2lm_scalar
        
        Function based on DDFacet to generate l,m coordinates from RA, Dec SkyCoord

        :param self: based on Sky c´lass
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
    

    def show(self,
             datakey: str="restored",
             plot_facets: list | str="all",
             channel=0,
             stokes=0,
             vmin: float=-8.e-5,
             vmax: float=2.e-4
             ):
        # this should imshow the sky and its divisions.
        # set up the plot axes etc
        fig, ax = plt.subplots(subplot_kw=dict(projection=self.gridwcs))
        ax.set(xlabel='Right Ascension', ylabel='Declination',title=self.name)
        # get data as numpy array
        data = self.data[datakey]
        # plot the overall sky box region
        self.grid_reg.plot(ax=ax)
        if self.nfacets==0:
            # in this scenario, simply show the data and region
            ax.imshow(data[channel,stokes,:,:], vmin=vmin, vmax=vmax, origin='lower')           
        else:
            if plot_facets=="all":
                # only plot the facet grid
                ax.imshow(data[channel,stokes,:,:], vmin=vmin, vmax=vmax, origin='lower')
                # plot the facet regions only. No facet-based overplot, since we want everything...
                for facet_key in self.facets.keys():
                    self.facets[facet_key].grid_reg.plot(ax=ax)
            else:
                # get the keys associated to the integers requested
                plotkeys=[]
                keylist = list(self.facets.keys())
                for plot_facet in plot_facets:
                    plotkeys.append(keylist[plot_facet])
                # plot the individual facet grid + region
                for facet_key in plotkeys:
                    this_facet = self.facets[facet_key]                    
                    ax.imshow(this_facet.data[datakey][channel,stokes,:,:],
                                    transform=ax.get_transform(this_facet.gridwcs),
                                    vmin=vmin, 
                                    vmax=vmax, 
                                    origin="lower")
                    this_facet.grid_reg.plot(ax=ax)
                # underlay the full sky
                ax.imshow(data[channel,stokes,:,:], vmin=vmin, vmax=vmax, origin='lower',alpha=0.6)
        plt.show()
    
    ### facet initialisation functionalities
    def set_facet_pixgrid(self):
        # define regions by their edges
        self.bin_edges   = np.linspace(0,self.npix,self.nfacets+1).astype(int)
        # build sizes
        self.bin_sizes   = self.bin_edges[1:] - self.bin_edges[:-1]
        self.facetvertices = generate_vertices(self.bin_edges)
        # finish making widths list of len nfacets**2
        self.bin_sizes = np.tile(self.bin_sizes,self.nfacets)
        # build centers
        bin_pixcoord = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        bin_centers_x,bin_centers_y  = np.meshgrid(bin_pixcoord,bin_pixcoord)
        self.bin_centers = regions.PixCoord(x=bin_centers_x.ravel(),y=bin_centers_y.ravel())
        self.facet_phasecenters = self.bin_centers.to_sky(self.gridwcs)
        


    def generate_facet_regions(self):
        reg_visuals = {'color':"red",
                       'linewidth':1}
        facet_grid_regs = []
        facet_sky_regs  = []
        # returns region coverage of the sky. Used for overlap determinations.
        for i in range(self.nfacets**2):
            grid_reg = regions.RectanglePixelRegion(center = self.bin_centers[i],
                                                   width  = self.bin_sizes[i],
                                                   height = self.bin_sizes[i],
                                                   visual=reg_visuals
                                                   )
            sky_reg = grid_reg.to_sky(self.gridwcs)
            facet_sky_regs.append(sky_reg)
            facet_grid_regs.append(grid_reg)
        return facet_sky_regs,facet_grid_regs


    def close(self):
        '''
        Docstring for close
        Close all HDU objects, free up virtual memory, exit gracefully.
        '''
        print("todo")
#        for key in self.data.keys():
#            self.data[key]._close()
#        self.restored.close()
#        self.residual.close()
#        self.model.close()
#        self.mask.close()

    def ImageHDU(self,
                 name:str,
                 data:np.ndarray,
                 wcs:WCS
                 ):
        '''
        Generates HDU object from data and WCS for writing purposes
        
        :param self: class Sky
        :param name: Name of the HDU, restored, residual, model etc
        :type name: str
        :param data: data to put in the HDU
        :type data: np.ndarray
        :param wcs: WCS of the HDU
        :type wcs: WCS
        '''
        this_hdu      = fits.PrimaryHDU(data=data, 
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

    def region(self, 
               reg_visuals:dict):
        # returns region coverage of the sky. Used for overlap determinations.
        ### !!! CAREFUL !!! when initialising facets, the ORIGINAL SKY PHASECENTER gets used.......
        ### TODO: fix this eventually.
        sky_reg  = regions.RectangleSkyRegion(center=self.phasecenter, 
                                              width=self.cellsize*(self.npix), 
                                              height=self.cellsize*(self.npix),
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

    def write(self,filename):
        '''docstring'''
        print("TODO")

    def read(filename):
        '''docstring'''
        print("TODO")


def generate_vertices(edges):
    '''docstring'''
    vertices=[]
    for ifacet in range(len(edges)-1):
        for jfacet in range(len(edges)-1):
            vertx = [edges[ifacet],
                    edges[ifacet+1],
                    edges[ifacet+1],
                    edges[ifacet]
                    ]
            verty = [edges[jfacet],
                    edges[jfacet],
                    edges[jfacet+1],
                    edges[jfacet+1]
                    ]
            vertices.append(regions.PixCoord(x=verty,y=vertx))
    return vertices

def MaskFromVertices(data,vertices):
    '''docstring'''
    if len(data)==2:
        d = np.zeros_like(data).astype(bool)
    else:
        d = np.zeros_like(data[0,0,:,:]).astype(bool)
    x,y = np.array(vertices.xy).astype(int)
    d[np.min(x):np.max(x),
      np.min(y):np.max(y)] = True
    return d