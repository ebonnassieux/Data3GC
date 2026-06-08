# class containing sky properties for 3GC
# core dependencies:
# numpy
# astropy
# matplotlib
# regions
# pathlib
# json

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
# for pathing functionalities
import pathlib
# for JSON serialisation
import json
# for xarray functionalities
import xarray as xr

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
        '''
        Docstring for __str__
        :param self: Description
        '''
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
                 stokes       : str="I",
                 skyname      : str="Sky",
                 data         : dict=None,
                 npix_y       : int=None
                 ):
        '''
        Docstring for __init__
        
        :param self: Description
        :param centrecoords: Description
        :type centrecoords: SkyCoord
        :param npix: Description
        :type npix: int
        :param npix_y : used in case your grid is rectangular. Default is None, sets it to npix
        :type npix_y: int
        :param cellsize: Description
        :type cellsize: u.Quantity
        :param freqs: Description
        :type freqs: list[u.Quantity]
        :param nfacets: Description
        :type nfacets: int
        :param stokes: Description
        :type stokes: Stokes
        :param skyname: Description
        :type skyname: str
        :param data: the data dict to initialise the class with. By default, is None, initialing all data to 0.
        :type data: dict 
        '''
        # initialise global sky variables
        self.name        = skyname
        self.phasecenter = centrecoords
        self.npix        = npix
        if npix_y is None:
            self.npix_y  = self.npix
        else:
            self.npix_y  = npix_y        
        self.cellsize    = cellsize
        self.freqs       = freqs
        self.stokes      = list(set(stokes))
        self.nfacets     = nfacets
        # initialise image grid variables
        # base shape on xradio schema 
        # https://github.com/casangi/xradio/blob/470-changes-needed-for-astroviper/docs/source/image_data/tutorials/image_schema_proposal.ipynb
        self.imshape = (len(self.freqs),len(self.stokes),self.npix, self.npix_y)    
        self.wcs     = WCS(self.wcs_input_dict())
        self.gridwcs = self.wcs.dropaxis(2).dropaxis(2)
        # initialise coords of data grids
        ### the axes are switched here; this is by design. This is due to meshgrid behaviour.
        coordgrid = np.meshgrid(np.arange(self.npix_y),np.arange(self.npix))
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
        # initialise the data grids
        self.initdata()
        # initialise regions
        self.sky_reg,self.grid_reg = self.region(self.sky_visuals)
        if self.nfacets!=0:
            # initialise facet properties
            self.set_facet_pixgrid()
            self.facet_sky_regs,self.facet_grid_regs = self.generate_facet_regions()
            self.facets={}
            for facet_index in range(len(self.facet_grid_regs)):
                # read facet initialisation params
                facet_phasecenter = self.facet_phasecenters[facet_index]
                facetname = facet_phasecenter.to_string('hmsdms')
                verts = self.facetvertices[facet_index]
                xmin,xmax = int(np.min(verts.x)),int(np.max(verts.x))
                ymin,ymax = int(np.min(verts.y)),int(np.max(verts.y))
                xlen = xmax-xmin
                ylen = ymax-ymin
                # initialise the facet as sky object
                self.facets[facetname]=Sky(skyname=facetname,
                                        centrecoords=facet_phasecenter,
                                        npix=xlen,
                                        npix_y=ylen,
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
            
                self.init_facet(facetname)
                # generate mask from vertices.
                verts = self.facetvertices[facet_index]
                xmin,xmax = int(np.min(verts.x)),int(np.max(verts.x))
                ymin,ymax = int(np.min(verts.y)),int(np.max(verts.y))
                sky_to_facet_regmask = np.zeros(self.imshape).astype(bool)
                sky_to_facet_regmask[:,:,ymin:ymax,xmin:xmax]=True
                self.facets[facetname].sky_to_facet_regmask = sky_to_facet_regmask
                # add vertices info
                self.facets[facetname].xmin = xmin
                self.facets[facetname].xmax = xmax
                self.facets[facetname].ymin = ymin
                self.facets[facetname].ymax = ymax
            # read data from sky
            self.update_facets()


    ### update sky with facet information
    def update_sky(self,
                    datakey: list | str="all",
                    update_facets: list | str="all",
                    channel=0,
                    stokes=0,
                    ) -> None:
        '''
        Function to update sky data with facet data.
        Can be done per data key and per facet section of the sky.
        Does all datakeys and all sky by default
        '''
        # if no facets present, skip compute
        if self.nfacets == 0:
            return
        # for "all" keys, update full lists
        if update_facets == "all":
            update_facets = list(self.facets.keys())
        # if the update_facets are not in the facet keys, assume 
        # a list of indices are provided, and update accordingly
        elif update_facets[0] not in self.facets.keys() and type(update_facets[0])==int:
            keylist = list(self.facets.keys())
            update_facets = [keylist[i] for i in update_facets]
        else:
            raise ValueError("update_facets should be either 'all', a list of correct facet keys, or a corresponding list of indices.")
            
        if datakey == "all":
            datakeys = list(self.datakeys)
            print("Datakeys : ",datakeys)
            stop
        else:
            datakeys = [datakey] if isinstance(datakey, str) else list(datakey)
        # update sky data from facet data using xarray indexing

        vmin=-8.e-5
        vmax=2.e-4

        plt.imshow(self.data[datakey][channel,stokes,:,:].data, vmin=vmin, vmax=vmax, origin='lower') 
        plt.show()

        for facet_key in update_facets:
            facet = self.facets[facet_key]
            for datakey in datakeys:
                # Select matching dimensions from both sky and facet
                # self.data[datakey].data[channel,
                #                         stokes,
                #                         facet.xmin:facet.xmax,
                #                         facet.ymin:facet.ymax] = (
                #     facet.data[datakey].data[channel, stokes,:,:]
                # )


                ### this works
                # self.data[datakey].isel(freq=channel,
                #         stokes=stokes,
                #         x=slice(facet.xmin,facet.xmax),
                #         y=slice(facet.ymin,facet.ymax)
                # )[:] = 0

                self.data[datakey].isel(freq=channel,
                        stokes=stokes,
                        x=slice(facet.xmin,facet.xmax),
                        y=slice(facet.ymin,facet.ymax)
                )[:] = facet.data[datakey].isel(freq=channel,stokes=stokes,x=slice(0,facet.npix),y=slice(0,facet.npix_y)).data


                del(facet)




        plt.imshow(self.data[datakey][channel,stokes,:,:].data, vmin=vmin, vmax=vmax, origin='lower') 
        plt.show()


    def update_facets(self,
                    datakey: list | str="all",
                    update_facets: list | str="all",
                    channel=0,
                    stokes=0,
                    ) -> None:
        '''
        Function to update facet data with sky data.
        Can be done per data key and per facet.
        Does all datakeys and all facets by defaults
        '''
        # if no facets present, skip compute
        if self.nfacets == 0:
            return
        # for "all" keys, update full lists
        if update_facets == "all":
            update_facets = list(self.facets.keys())
        # if the update_facets are not in the facet keys, assume 
        # a list of indices are provided, and update accordingly
        elif update_facets[0] not in self.facets.keys() and type(update_facets[0])==int:
            keylist = list(self.facets.keys())
            update_facets = [keylist[i] for i in update_facets]
        else:
            raise ValueError("update_facets should be either 'all', a list of correct facet keys, or a corresponding list of indices.")
        if datakey == "all":
            datakeys = self.datakeys
        else:
            datakeys = [datakey] if isinstance(datakey, str) else list(datakey)
        print("update_facet datakeys",datakeys)
        # update facet data from sky data using the underlying ndarray buffer
        for facet_key in update_facets:
            facet = self.facets[facet_key]
            for datakey in datakeys:
                # flippums needed to correctly add the data. mindbreaking
                facet.data[datakey].data[channel,
                                          stokes,
                                          :,
                                          :] = self.data[datakey].data[channel,
                                                                       stokes,
                                                                       facet.ymin:facet.ymax,
                                                                       facet.xmin:facet.xmax].T
                

    def radec2lm_scalar(self,
                        coords      : SkyCoord,
                        phasecenter : SkyCoord = None,
                        ) -> list[float, float]:
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
        m = np.sin(decs) * np.cos(refdec) - np.cos(decs) * np.sin(refdec) * np.cos(ras - refra)
        return l,m

    def show(self,
             datakey: str="restored",
             plot_facets: list | str="all",
             channel=0,
             stokes=0,
             vmin: float=-8.e-5,
             vmax: float=2.e-4
             ) -> None:
        '''
        Function to show the current sky, requesting specific facet subsets
        
        :param self: Description
        :param datakey: Description
        :type datakey: str
        :param plot_facets: Description
        :type plot_facets: list | str
        :param channel: Description
        :param stokes: Description
        :param vmin: Description
        :type vmin: float
        :param vmax: Description
        :type vmax: float
        '''
        # this should imshow the sky and its divisions.
        # set up the plot axes etc
        fig, ax = plt.subplots(subplot_kw=dict(projection=self.gridwcs))
        ax.set(xlabel='Right Ascension', ylabel='Declination',title=self.name)
        # get data as numpy array
        # plot the overall sky box region
        self.grid_reg.plot(ax=ax)
        if self.nfacets==0:
            # in this scenario, simply show the data and region
            ax.imshow(self.data[datakey][channel,stokes,:,:].data, vmin=vmin, vmax=vmax, origin='lower')       
        else:
            if plot_facets=="all":
                # only plot the facet grid
                ax.imshow(self.data[datakey][channel,stokes,:,:].data, vmin=vmin, vmax=vmax, origin='lower')
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
                    ax.imshow(this_facet.data[datakey][channel,stokes,:,:].data,
                                    transform=ax.get_transform(this_facet.gridwcs),
                                    vmin=vmin, 
                                    vmax=vmax, 
                                    origin="lower")
                    this_facet.grid_reg.plot(ax=ax)
                # underlay the full sky
                ax.imshow(self.data[datakey][channel,stokes,:,:].data, vmin=vmin, vmax=vmax, origin='lower',alpha=0.1)
        plt.show()












    def initdata(self) -> None:
        '''
        Creates and populates data dictionary.
        
        :param self: Sky object
 
        '''
        self.data={}
        self.datakeys = ["dirty",
                        "restored",
                        "residual",
                        "model",
                        "mask",
                        "beam"]
        for key in self.datakeys:
            self.data[key] = xr.DataArray(
                                        data=np.zeros(self.imshape), 
                                        dims=["freq", 
                                              "stokes", 
                                              "x", 
                                              "y"],
                                        coords={
                                            "freq": self.freqs.value,
                                            "stokes": self.stokes,
                                            "x":np.arange(self.npix),
                                            "y":np.arange(self.npix_y),
                                            "ra":(("x", "y"), self.ras.value),
                                            "dec":(("x", "y"), self.decs.value),
                                            "l":(("x", "y"), self.l),
                                            "m":(("x", "y"), self.m)
                                            },
                                        name=key
                                        )

    def init_facet(self,facetname) -> None:
        '''
        Initialises all facet properties for requested facet.
        
        :param self: Sky object
 
        '''
        
        self.facets[facetname].wcs     = WCS(self.facets[facetname].wcs_input_dict(facet=True))
        self.facets[facetname].gridwcs = self.facets[facetname].wcs.dropaxis(2).dropaxis(2)

        self.facets[facetname].data={}
        self.facets[facetname].datakeys = ["dirty",
                                        "restored",
                                        "residual",
                                        "model",
                                        "mask",
                                        "beam"]
        for key in self.datakeys:
            self.facets[facetname].data[key] = xr.DataArray(
                                        data=np.zeros(self.facets[facetname].imshape), 
                                        dims=["freq", 
                                              "stokes", 
                                              "x", 
                                              "y"],
                                        coords={
                                            "freq": self.freqs.value,
                                            "stokes": self.stokes,
                                            "x":np.arange(self.facets[facetname].npix),
                                            "y":np.arange(self.facets[facetname].npix_y),
                                            "ra":(("x", "y"), self.facets[facetname].ras.value),
                                            "dec":(("x", "y"), self.facets[facetname].decs.value),
                                            "l":(("x", "y"), self.facets[facetname].l),
                                            "m":(("x", "y"), self.facets[facetname].m)
                                            },
                                        name=key
                                        )

    
    def region(self, 
               reg_visuals:dict) -> list[regions.RectangleSkyRegion, regions.PixelRegion]:
        '''
        Returns region coverage of the sky. Used for overlap determinations.
        !!! CAREFUL !!! when initialising facets, the ORIGINAL SKY PHASECENTER gets used.......

        :param self: Description
        :param reg_visuals: Description
        :type reg_visuals: dict
        '''
        sky_reg  = regions.RectangleSkyRegion(center=self.phasecenter, 
                                              width=self.cellsize*(self.npix), 
                                              height=self.cellsize*(self.npix_y),
                                              visual=reg_visuals
                                              )
        grid_reg = sky_reg.to_pixel(self.gridwcs)
        return sky_reg,grid_reg
    
    def close(self) -> None:
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
    
    def set_facet_pixgrid(self) -> None:
        '''
        Docstring for set_facet_pixgrid
        
        :param self: Description
        '''
        # define regions by their edges
        self.bin_edges_x   = np.linspace(0,self.npix,self.nfacets+1).astype(int)
        if self.npix_y is None:
            self.bin_edges_y = self.bin_edges_x
        else:
            self.bin_edges_y = np.linspace(0,self.npix_y,self.nfacets+1).astype(int)
        # build sizes
        self.bin_sizes_x   = self.bin_edges_x[1:] - self.bin_edges_x[:-1]
        self.bin_sizes_y   = self.bin_edges_y[1:] - self.bin_edges_y[:-1]
        
        self.facetvertices = generate_vertices(self.bin_edges_y,self.bin_edges_x)
        # finish making widths list of len nfacets**2
        self.bin_sizes_x = np.tile(self.bin_sizes_x,self.nfacets)
        self.bin_sizes_y = np.tile(self.bin_sizes_y,self.nfacets)
        
        # build centers
        bin_pixcoord_x = (self.bin_edges_x[:-1] + self.bin_edges_x[1:]) / 2
        bin_pixcoord_y = (self.bin_edges_y[:-1] + self.bin_edges_y[1:]) / 2
        bin_centers_x,bin_centers_y  = np.meshgrid(bin_pixcoord_x,bin_pixcoord_y)
        self.bin_centers = regions.PixCoord(x=bin_centers_x.ravel(),y=bin_centers_y.ravel())
        self.facet_phasecenters = self.bin_centers.to_sky(self.gridwcs)

    def generate_facet_regions(self) -> list[tuple, tuple]:
        '''
        Function to generate region object for a given facet
        
        :param self: Description
        '''
        reg_visuals = {'color':"red",
                       'linewidth':1}
        facet_grid_regs = []
        facet_sky_regs  = []
        # returns region coverage of the sky. Used for overlap determinations.
        for i in range(self.nfacets**2):
            grid_reg = regions.RectanglePixelRegion(center = self.bin_centers[i],
                                                   width  = self.bin_sizes_x[i],
                                                   height = self.bin_sizes_y[i],
                                                   visual=reg_visuals
                                                   )
            sky_reg = grid_reg.to_sky(self.gridwcs)
            facet_sky_regs.append(sky_reg)
            facet_grid_regs.append(grid_reg)
        return facet_sky_regs,facet_grid_regs

    def radec2lm_scalar(self,
                        coords      : SkyCoord,
                        phasecenter : None | SkyCoord = None,
                        ) -> list[float, float]:
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
        m = np.sin(decs) * np.cos(refdec) - np.cos(decs) * np.sin(refdec) * np.cos(ras - refra)
        return l,m

    def wcs_input_dict(self,facet=False) -> dict:
        '''
        Docstring for wcs_input_dict
        This is the dictionary which defines the default WCS for our Sky object.
        
        :param self: Sky class
        '''
        # reference pixel is defined as centre of image, first freq, first stokes
        ### add 1 due to FITS convention count starting at 1 rather than 0
        ### i.e. Fortran-style rather than C-style. This is also why the other
        ### values are set to 1 rather than 0.
        if facet==True:
             ref_pixels = [round(0.5*self.npix)+1,
                           round(0.5*self.npix_y)+1,
                           1.,
                           1.]
        else:
            ref_pixels = [round(0.5*self.npix)+1,
                        round(0.5*self.npix_y)+1,
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
            "NAXIS2" : self.npix_y,
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
#todo            "NAXIS2" : self.npix,
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
    


    @classmethod
    def from_json(cls, json_string: str) -> 'Sky':
        '''
        Instantiate Sky object from JSON string.
        
        :param json_string: JSON string representation of Sky
        :type json_string: str
        :return: Sky object
        :rtype: Sky
        '''
        sky_dict = json.loads(json_string)
        return cls.from_dict(sky_dict)

    @classmethod
    def from_json_file(cls, filename: str) -> 'Sky':
        '''
        Load Sky object from JSON file.
        
        :param filename: Input JSON filename
        :type filename: str
        :return: Sky object
        :rtype: Sky
        '''
        filepath = pathlib.Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} not found.")
        
        with open(filepath, 'r') as f:
            sky_dict = json.load(f)
        
        return cls.from_dict(sky_dict)

    @classmethod
    def from_fits(cls,
                  filename:str,
                  hdu_n:int=0,
                  nfacets:int=5,
                  skyname:str=None,
                  stokes:str="I",
                  default_data_type:str="restored"):
        '''
        Class method to initialise Sky object from a .fits file.
        Only works on a single hdu at a time at present, defined by hdu_n.
        If no skyname is provided, it will be set to the filename absolute path.
        '''
        # check path
        filepath = pathlib.Path(filename)
        if filepath.exists()==False:
            raise FileNotFoundError
        # set skyname to filename if none provided 
        if skyname is None:
            skyname = filename
        hdul = fits.open(filename)
        hdu = hdul[hdu_n]
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
            fits_npix = max(fits_data.shape[2],fits_data.shape[3])
            fits_style="radio"
            # read centrecoords from fits
            centrepixcoord = round(fits_npix/2)
            centreskycoords = fits_wcs.pixel_to_world(centrepixcoord,centrepixcoord,0,0)
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
        this_sky = Sky(skyname=skyname,
                       centrecoords=fits_centrecoords,
                       npix=fits_npix,
                       cellsize=fits_cellsize,
                       freqs=fits_freqs,
                       nfacets=nfacets,
                       stokes=stokes
        )
        fits_datakeys = {"dirty",
                "restored",
                "residual",
                "model",
                "mask",
                "beam",
                "image"}
        baseimagename=None
        for key in fits_datakeys:
            if str(key) in filename:
                 baseimagename=filename.split(key)[0]
                 print("Image identified as containing %s; deriving base image name and populating sky with all available fits files."%key)
        if baseimagename==None:
            print("Image string does not include data keyword; assume it is a %s image. All other data arrays will be empty."%default_data_type)
            hdul = fits.open(filename)
            this_sky.data["restored"].data=hdul[hdu_n].data
            hdul.close()


        this_sky.update_facets()

        return this_sky


    
def generate_vertices(edges,
                      edges_y=None) -> tuple:
    '''
    Generates vertices for given list of region edges.
    
    :param edges: Description
    
    '''
    vertices=[]
    for ifacet in range(len(edges)-1):
        for jfacet in range(len(edges)-1):
            vertx = [edges[ifacet],
                    edges[ifacet+1],
                    edges[ifacet+1],
                    edges[ifacet]
                    ]
            if edges_y is None:
                verty = [edges[jfacet],
                        edges[jfacet],
                        edges[jfacet+1],
                        edges[jfacet+1]
                        ]
            else:
                verty = [edges_y[jfacet],
                        edges_y[jfacet],
                        edges_y[jfacet+1],
                        edges_y[jfacet+1]
                        ]
            vertices.append(regions.PixCoord(x=verty,y=vertx))
    return vertices

def MaskFromVertices(data,vertices) -> np.ndarray:
    '''
    Make a mask from given array + vertices.
    
    :param data: Description
    :param vertices: Description
    '''
    if len(data)==2:
        d = np.zeros_like(data).astype(bool)
    else:
        d = np.zeros_like(data[0,0,:,:]).astype(bool)
    x,y = np.array(vertices.xy).astype(int)
    d[np.min(x):np.max(x),
      np.min(y):np.max(y)] = True
    return d