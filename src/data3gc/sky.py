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

from time import time
def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__:16s} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

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
    @timer  
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
        # split stokes string into list
        self.stokes      = list(set(stokes))
        self.nfacets     = nfacets
        # initialise image grid variables
        # base shape on xradio schema 
        # https://github.com/casangi/xradio/blob/470-changes-needed-for-astroviper/docs/source/image_data/tutorials/image_schema_proposal.ipynb
        self.imshape = (len(freqs),len(stokes),npix, self.npix_y)
        # # initialise coords of data grids
        self.make_coord_grids()
        # initialise the data grids
        self.initdata()
        self.facets={}
        if self.nfacets!=0:
            # initialise facet properties
            self.set_facet_pixgrid()
            for facet_index in range(len(self.facet_phasecenters)):
                # debug
                print()
                print("Initialisation of facet",facet_index)
                # read facet initialisation params
                facet_phasecenter = self.facet_phasecenters[facet_index]
                facetname = facet_phasecenter.to_string('hmsdms')
                verts = self.facetvertices[facet_index]
                xmin,xmax = int(np.min(verts.x)),int(np.max(verts.x))
                ymin,ymax = int(np.min(verts.y)),int(np.max(verts.y))
                # initialise the facet as sky object
                self.facets[facetname]=Sky(skyname=facetname,
                                        centrecoords=facet_phasecenter,
                                        npix=xmax-xmin,
                                        npix_y=ymax-ymin,
                                        cellsize=cellsize,
                                        freqs=freqs,
                                        nfacets=0,
                                        stokes=self.stokes
                )
                # initialise region visuals
                self.facets[facetname].sky_reg,self.facets[facetname].grid_reg = \
                    self.facets[facetname].region(self.facet_phasecenters[facet_index],
                                                  self.gridwcs,
                                                  facet_visuals())
                # save vertices info
                self.facets[facetname].xmin = xmin
                self.facets[facetname].xmax = xmax
                self.facets[facetname].ymin = ymin
                self.facets[facetname].ymax = ymax
        
                
    ### update sky with facet information
    @timer
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
            update_facets = self.facets.keys()
        # if the update_facets are not in the facet keys, assume 
        # a list of indices are provided, and update accordingly
        elif update_facets[0] not in self.facets.keys() and type(update_facets[0])==int:
            keylist = list(self.facets.keys())
            update_facets = [keylist[i] for i in update_facets]
        else:
            raise ValueError("update_facets should be either 'all', a list of correct facet keys, or a corresponding list of indices.")
            
        if datakey == "all":
            datakeys = list(self.datakeys)
        else:
            datakeys = [datakey] if isinstance(datakey, str) else list(datakey)
        # update sky data from facet data
        for facet_key in update_facets:
            facet = self.facets[facet_key]
            for datakey in datakeys:
                self.data[datakey][channel,
                                   stokes,
                                   self.facets[facet_key].xmin:self.facets[facet_key].xmax,
                                   self.facets[facet_key].ymin:self.facets[facet_key].ymax] = self.facets[facet_key].data[datakey][channel,stokes,:,:].data
        del(facet)



    @timer
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
        # update facet data from sky data using the underlying ndarray buffer
        for facet_key in update_facets:
            facet = self.facets[facet_key]
            for datakey in datakeys:
                facet.data[datakey].data[channel,
                                          stokes,
                                          :,
                                          :] = self.data[datakey].data[channel,
                                                                       stokes,
                                                                       facet.xmin:facet.xmax,
                                                                       facet.ymin:facet.ymax]
        del(facet)
                

    #@timer
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
        # calculate l,m
        l = np.cos(decs) * np.sin(ras - refra)
        m = np.sin(decs) * np.cos(refdec) - np.cos(decs) * np.sin(refdec) * np.cos(ras - refra)
        # return values
        return l,m
    
    @timer
    def make_coord_grids(self):
        self.wcs     = WCS(self.wcs_input_dict())
        self.gridwcs = self.wcs.dropaxis(2).dropaxis(2)
        # initialise coords of data grids
        coordgrid = np.indices((self.npix,self.npix_y))
        # drop the Stokes, Freq axes for this
        x = coordgrid[0].ravel()
        y = coordgrid[1].ravel()
        ra_flat, dec_flat = self.gridwcs.wcs_pix2world(x, y, 1)   # origin=1 like your code
        self.ras = ra_flat.reshape(self.npix, self.npix_y) * u.deg
        self.decs = dec_flat.reshape(self.npix, self.npix_y) * u.deg
        self.l,self.m  = self.radec2lm_scalar(SkyCoord(self.ras,self.decs),self.phasecenter)
        # initialise sky regions
        self.sky_reg,self.grid_reg = self.region(self.phasecenter,self.gridwcs,sky_visuals())


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
            ax.imshow(self.data[datakey][channel,stokes,:,:].data, 
                      vmin=vmin, 
                      vmax=vmax, 
                      origin='lower',
                      transform=ax.get_transform(this_facet.gridwcs))    
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
                    # add the facet region
                    this_facet.grid_reg.plot(ax=ax)
                # underlay the full sky
                ax.imshow(self.data[datakey][channel,stokes,:,:].data, 
                          vmin=vmin, 
                          vmax=vmax, 
                          origin='lower',
                          alpha=0.1,
                          transform=ax.get_transform(this_facet.gridwcs))
        plt.show()


    @timer
    def initdata(self) -> None:
        '''
        Creates and populates data dictionary.
        
        :param self: Sky object
 
        '''
        # TODO: this should be a single DataSet!!!
        self.data={}
        self.datakeys = ["dirty",
                        "restored",
                        "residual",
                        "model",
                        "mask",
                        "beam"]
        dims=["freq", 
            "stokes", 
            "x", 
            "y"]
        coords={
                "freq": self.freqs.value,
                "stokes": self.stokes,
                "x":np.arange(self.npix),
                "y":np.arange(self.npix_y),
                "ra":(("x", "y"), self.ras.value),
                "dec":(("x", "y"), self.decs.value),
                "l":(("x", "y"), self.l),
                "m":(("x", "y"), self.m)
                }
        facet_xarr = xr.DataArray(data=np.zeros(self.imshape), 
                                  dims=dims,
                                  coords=coords,
                                  name="temp"
                                  )
        for key in self.datakeys:
            ### careful - this might need to become a copy
            self.data[key] = facet_xarr#.copy()
            self.data[key].name=key
        del(facet_xarr)

    #@timer
    def region(self,
               center_coords:SkyCoord,
               sky_gridwcs:WCS,
               reg_visuals:dict) -> list[regions.RectangleSkyRegion, regions.PixelRegion]:
        '''
        Returns region coverage of the sky. Used for overlap determinations.

        :param self: Description
        :param reg_visuals: Description
        :type reg_visuals: dict
        '''
        pix_x, pix_y = sky_gridwcs.wcs_world2pix(
                                center_coords.ra.deg,
                                center_coords.dec.deg,
                                0,          # origin=0 because we want it given in python-style counting
                            )

        pix_center = regions.PixCoord(x=pix_x, y=pix_y)
        grid_reg = regions.RectanglePixelRegion(center=pix_center,
                                                width =self.npix,
                                                height=self.npix_y,
                                                visual=reg_visuals)
        print("DEBUG - grid_reg",grid_reg)
        sky_reg = grid_reg.to_sky(sky_gridwcs)
        return sky_reg,grid_reg
    
    def JonesRegions(self,njonesdir) -> None:
        '''
        Function to associate regions for given Jones-direction.
        TO BE DONE
        
        :param self: Description
        :param njonesdir: Description
        '''
        # generates tessels (Jones facets)
        test=1
    
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


    def write(self,
              basename:str=None,
              write_facets:list|str=None,
              datakey:list|str="all",
              object_serialisation:str="json",
              verbose:bool=True,
              overwrite:bool=True
              ) -> None:
        '''
        Function to write sky object as .fits files, with metadata
        preserved in JSON format. Individual facets are not written
        to fits by default; can write either all or some by requesting
        their facetnames. If you wish to write metadata ONLY (e.g. instantiate
        sky object from output of imaging run), set datakey to None.
        '''
        # initialise write parameters for this call
        if basename==None:
            basename=self.name
        if datakey=="all":
            datakeys=self.datakeys
        elif datakey==None:
            datakeys=[]
        else:
            datakeys=datakey
        if self.nfacets==0:
            write_facets=None
        # check if basename includes path; create it if absent
        if "/" in basename:
            # if basename *is* a path, append skyname
            if basename[-1]=="/":
                writepath=pathlib.Path(basename)
                basename+=self.name
            else:
                # otherwise, get the writepath out of basename
                writepath=pathlib.Path(basename[0:-len(basename.split("/")[-1])])
            if pathlib.Path.is_dir(writepath)==False:
                pathlib.Path.mkdir(writepath)
        if verbose:
            print("Serialising %s in %s"%(self.name,writepath))
        # serialise metadata
        if object_serialisation=="json":
            self.to_json_file(filename=basename+".json")
        ### write requested data to fits
        # check for facet output
        if write_facets is not None:
            # build directory for facet fits files
            facets_dir = pathlib.Path(basename+"_facets")
            if pathlib.Path.is_dir(facets_dir)==False:
                pathlib.Path.mkdir(facets_dir)
            facets_dir=facets_dir.as_posix()
            if verbose:
                print("Writing facets fits files to: %s"%facets_dir)
        for datakey in datakeys:
            outfilename=basename+"."+datakey+".fits"
            self.WriteFits(filename=outfilename,
                           key=datakey,
                           overwrite=overwrite)
            self.filename=outfilename
            if verbose:
                print("Writing -- %8s -- sky data to  : %s"%(datakey,outfilename))
            if write_facets is not None:
                # if all facets requested, write all facets
                if write_facets=="all":
                    write_facets=self.facets.keys()
                # write requested facets
                for facet in write_facets:
                    if facet in self.facets:
                        outfilename=facets_dir+"/"+facet.replace(" ","_")+"."+datakey+".fits"
                        self.facets[facet].WriteFits(filename=outfilename,
                                                     key=datakey,
                                                     overwrite=overwrite)
                        self.facets[facet].filename=outfilename
                        if verbose:
                            print("Writing -- %8s -- facet data to: %s"%(datakey,outfilename))
                    elif verbose==True:
                        print("Count not find requested facet %s for sky %s"%(facet,self.name))
        # print output to notify user
        if verbose:
            print("Serialisation of %s complete."%self.name)

    def WriteFits(self,
                  filename:str,
                  key:str,
                  overwrite=False
                  ) -> None:
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
        this_hdu      = fits.PrimaryHDU(data=self.data[key].data, 
                                        header=self.wcs.to_header())
        writepath = pathlib.Path(filename)
        this_hdu.writeto(writepath,overwrite=overwrite)
        del(this_hdu)

    def to_dict(self, 
                include_data:bool=False) -> dict:
        '''
        Converts Sky object to a dictionary for JSON serialization.
        
        :param self: Sky object
        :param include_data: set to True to include image data in the dict itself. Defaults to False
        :type include_data: bool
        :return: Dictionary representation current Sky object
        :rtype: dict
        '''
        sky_dict = {
            "name": self.name,
            "phasecenter": {
                "ra": self.phasecenter.ra.deg,
                "dec": self.phasecenter.dec.deg,
                "unit": "deg"
            },
            "npix": int(self.npix),
            "cellsize": {
                "value": float(self.cellsize.value),
                "unit": str(self.cellsize.unit)
            },
            "freqs": [
                {"value": float(freq.value), "unit": str(freq.unit)}
                for freq in self.freqs
            ],
            "nfacets": int(self.nfacets),
            "stokes": list(self.stokes) if isinstance(self.stokes, tuple) else [self.stokes],
            "imshape": list(self.imshape),
            "wcs": self.wcs.to_header().tostring() if self.wcs else None,
        }
        
        if include_data:
            # also serialize image data arrays. Bad idea but better to futureproof...
            sky_dict["data"] = {}
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    sky_dict["data"][key] = {
                        "array": value.tolist(),
                        "dtype": str(value.dtype),
                        "shape": list(value.shape)
                    }
        
        # serialise facets. Add data here too if requested.
        if self.nfacets != 0 and self.facets:
            sky_dict["facets"] = {
                facet_name: facet.to_dict(include_data=include_data)
                for facet_name, facet in self.facets.items()
            }
        
        return sky_dict

    def to_json(self, 
                include_data:bool=False, 
                indent:int=2) -> str:
        '''
        Generate JSON.dumps string for this Sky
        
        :param self: Sky object
        :param include_data: Whether to include image data arrays. Default True.
        :type include_data: bool
        :param indent: JSON indentation level. Default 2.
        :type indent: int
        :return: JSON string representation
        :rtype: str
        '''
        sky_dict = self.to_dict(include_data=include_data)
        return json.dumps(sky_dict, indent=indent)

    def to_json_file(self, 
                     filename: str, 
                     include_data:bool=False,
                     overwrite:bool=True,
                     indent:int=2) -> None:
        '''
        Write Sky object to JSON file.
        
        :param self: Sky object
        :param filename: Output filename
        :type filename: str
        :param include_data: include data values in JSON file. Default False.
        :type include_data: bool
        :param overwrite: overwrite existing file if present. Default True.
        :type overwrite: bool
        :param indent: JSON indentation level. Default 2.
        :type indent: int
        '''
        filepath = pathlib.Path(filename)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Set overwrite=True to overwrite.")
        sky_dict = self.to_dict(include_data=include_data)
        with open(filepath, 'w') as f:
            json.dump(sky_dict, f, indent=indent)

    @classmethod
    def from_dict(cls, sky_dict: dict) -> 'Sky':
        '''
        Create Sky object from dictionary.
        Image data must be re-loaded separately from fits, or saved in serialisation.
        Fits filenames will be present in the dict.
        
        :param sky_dict: Dictionary with Sky parameters
        :type sky_dict: dict
        :return: Sky object
        :rtype: Sky
        '''
        # Extract basic parameters
        centrecoords = SkyCoord(
            ra=sky_dict["phasecenter"]["ra"] * u.deg,
            dec=sky_dict["phasecenter"]["dec"] * u.deg
        )
        npix = sky_dict["npix"]
        cellsize = u.Quantity(sky_dict["cellsize"]["value"], sky_dict["cellsize"]["unit"])
        
        # Reconstruct frequencies
        freqs = [
            u.Quantity(freq["value"], freq["unit"])
            for freq in sky_dict["freqs"]
        ]
        
        nfacets = sky_dict["nfacets"]
        stokes = tuple(sky_dict["stokes"]) if len(sky_dict["stokes"]) > 1 else sky_dict["stokes"][0]
        skyname = sky_dict.get("name", "Sky")
        
        # Create Sky object
        this_sky = cls(
            centrecoords=centrecoords,
            npix=npix,
            cellsize=cellsize,
            freqs=freqs,
            nfacets=nfacets,
            stokes=stokes,
            skyname=skyname
        )
        
        # Load image data if present
        if "data" in sky_dict and sky_dict["data"]:
            for key, data_info in sky_dict["data"].items():
                if isinstance(data_info, dict) and "array" in data_info:
                    this_sky.data[key] = np.array(
                        data_info["array"],
                        dtype=np.dtype(data_info["dtype"])
                    )
        
        # Load facets if present
        if "facets" in sky_dict and sky_dict["facets"]:
            for facet_name, facet_dict in sky_dict["facets"].items():
                facet_sky = cls.from_dict(facet_dict)
                this_sky.facets[facet_name] = facet_sky
        
        return this_sky

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


    def restoringbeam(self,
                      bmaj:u.Quantity,
                      bmin:u.Quantity,
                      PA:float=0) -> None:
        '''
        Generates restoring beam value for given grid
        
        :param self: Description
        :param bmaj: Description
        :type bmaj: u.Quantity
        :param bmin: Description
        :type bmin: u.Quantity
        :param PA: Description
        :type PA: float
        '''
        # restoring beam gaussian
        # call as a function to initialise on-the-fly
        self.bmaj=bmaj.rad
        self.bmin=bmin.rad
        self.PA=PA
        test=1  
    
    
    def set_facet_pixgrid(self) -> None:
        '''
        Docstring for set_facet_pixgrid
        
        :param self: Description
        '''
        # define regions by their edges
        bin_edges_x   = np.linspace(0,self.npix,self.nfacets+1).astype(int)
        if self.npix_y is None:
            bin_edges_y = bin_edges_x
        else:
            bin_edges_y = np.linspace(0,self.npix_y,self.nfacets+1).astype(int)
        # build facet vertices. flippems correctly handles recangular skies
        self.facetvertices = generate_vertices(bin_edges_y,bin_edges_x)
        # build centers
        bin_pixcoord_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
        bin_pixcoord_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        ### indices are switched below: this is normal, and due to np.meshgrid behaviour
        bin_centers_y,bin_centers_x  = np.meshgrid(bin_pixcoord_x,bin_pixcoord_y)
        bin_centers = regions.PixCoord(x=bin_centers_x.ravel(),y=bin_centers_y.ravel())
        self.facet_phasecenters = bin_centers.to_sky(self.gridwcs)
        self.facet_phasecenters_pix = bin_centers

  
    def wcs_input_dict(self) -> dict:
        '''
        Docstring for wcs_input_dict
        This is the dictionary which defines the default WCS for our Sky object.
        
        :param self: Sky class
        '''
        # reference pixel is defined as centre of image, first freq, first stokes
        ### add 1 due to FITS convention count starting at 1 rather than 0
        ### i.e. Fortran-style rather than C-style. This is also why the other
        ### values are set to 1 rather than 0.
        ref_pixels = [round(0.5*self.npix)+1,
                    round(0.5*self.npix_y)+1,
                    1.,
                    1.]
        print("ref_pixels",ref_pixels)
                
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
        this_sky = Sky(skyname=skyname,
                       centrecoords=fits_centrecoords,
                       npix=fits_npix_x,
                       npix_y=fits_npix_y,
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
#            print("Image string does not include data keyword; assume it is a %s image. All other data arrays will be empty."%default_data_type)
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
    if edges_y is None:
        edges_y=edges
    for ifacet in range(len(edges)-1):
        for jfacet in range(len(edges_y)-1):
            vertx = [edges[ifacet],
                    edges[ifacet+1],
                    edges[ifacet+1],
                    edges[ifacet]
                    ]
            verty = [edges_y[jfacet],
                    edges_y[jfacet],
                    edges_y[jfacet+1],
                    edges_y[jfacet+1]
                    ]
            vertices.append(regions.PixCoord(x=verty,y=vertx))
    return vertices

@timer
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


def sky_visuals():
        return {'color':"blue",
                'linewidth':5}

def facet_visuals():
        return {'color':"red",
                'linewidth':1}