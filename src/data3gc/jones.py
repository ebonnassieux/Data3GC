# Defines Jones object properties.
from __future__ import annotations
from typing import Any, Optional, NotRequired, Protocol, Self, cast, Tuple, TypeAlias, runtime_checkable
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy.typing
import xarray as xr
from attrs import define, field
import numpy as np
from pathlib import Path
# for serialisation
import json



def validate_axis(input_var,
                    desired_physical_type:str,
                    output_unit:u.Unit|u.IrreducibleUnit) -> u.Quantity:
    '''
    Function to take in sloppy inputs for xjones coordinate axes
    and reformat them to build nice xarray coordinates.
    
    :param input_var: coord value or array 
    :param desired_physical_type: physical type for this axis, such as time, frequency, Stokes...
    :type desired_physical_type: str
    :param output_unit: Fundamental unit to return the coord in, such as s, Hz...
    :type output_unit: u.Unit | u.IrreducibleUnit
    :return: Outputs a reformatted astropy Unit array using the provided values and requested units.
    :rtype: Quantity
    '''
    # since u.Quantity doesn't say if it's scalar or array, check that first.
    if isinstance(input_var,u.Quantity):
        output_var = cast(u.Quantity, input_var) # inform pylance
        # check if it is scalar or vector
        if output_var.isscalar:
            output_var = cast(u.Quantity,np.array([output_var]))
        # check if it has correct unit physical type
        if isinstance(output_var, u.Quantity) \
            and output_var.unit is not None \
                and output_var.unit.physical_type != desired_physical_type:
            output_var = output_var << output_unit
        # if it does, cast to default unit (e.g. seconds rather than minutes)
        output_var = input_var.to(output_unit)
    else:
        output_var=np.asarray(input_var) # check this works. pylance complains less at least.
        # if it is not a quantity, make it into an array if it is not already
        if np.isscalar(input_var):
            output_var=np.array([input_var])
        elif not isinstance(input_var, np.ndarray):
            output_var=np.asarray(input_var)
        # not a quantity to begin with; just add requested units
        output_var = cast(u.Quantity, output_var << output_unit)
    return output_var

class MissingDicoSolsKey(Exception):
    """
    Exception raised due to missing coordinates when serialising xJones as DicoSols.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@define
class xJones:
    gains: xr.Dataset
    '''xarray dataset containing the Jones terms with itss coordinates. These may be parametrised.'''
    name:str
    '''Name of these gains.'''
    gaintype:str
    '''Description of the gain type (eg scalar, full-jones, phase, amp...)'''
    msname:str
    '''Name of the Measurement Set associated to these gains.'''
    comments:np.typing.NDArray[np.str_]
    '''Optional comments'''

    def __init__(self,
                 name:str,
                 gaintype:str,
                 antennas:np.typing.NDArray[np.str_],
                 times:u.Quantity,
                 freqs:u.Quantity,
                 params:np.typing.NDArray[np.str_],
                 directions:Optional[Any] = None,
                 msname:str="Undefined",
                 comments:np.typing.NDArray[np.str_]=np.array([""])
                 ):
        '''
        This class takes in the coordinate values for xJones arrays.
        These are expected to be non-scalar astropy SkyCoords (for directions) or quantities (for all other axes).
        It builds and returns an empty xarray.
        '''
        # required metadata
        self.name = name
        self.gaintype = gaintype
        # optional metadata
        self.msname = msname
        self.comments=comments
        # check dirs, params dimensionality
        if np.isscalar(directions) or directions is None:
            dirs=np.array([directions])
        else:
            dirs=directions
            # cast(u.Quantity, input_var)
        if np.isscalar(params):
            params=np.array([params])
        # validate times, freqs dimensionality
        times = validate_axis(times, 'time', u.s)
        freqs = validate_axis(freqs, 'frequency', u.Hz)
        # build xarray shape
        xshape = (len(dirs),len(antennas),len(times),len(freqs),len(params))
        # build xarray coords
        dims=["Direction",
              "Antennas",
              "Times",
              "Freqs",
              "Params"]
        # build dict
        coords={"Direction":dirs,
                "Antennas":antennas,
                "Times":times,
                "Freqs":freqs,
                "Params":params
                }
        # build xarray dataset
        self.gains = xr.Dataset(
                        {self.name: (dims, np.zeros(xshape, dtype=np.float32))},
                        coords=coords,
                        attrs=dict(gaintype=gaintype,
                                   msname=msname,
                                   comments=comments),
                    )
    
    def __repr__(self) -> str:
        '''Return the underlying xarray dataset representation'''
        return self.gains.__repr__()


    def add_coords(self,
                   coordname:str,
                   coordvals:u.Quantity,
                   physical_type:str,
                   unit:u.Unit|u.IrreducibleUnit) -> None:
        '''
        Add coordinate axis specified by coordname with coordvals values. 
        :param self: xJones object
        :param coordname: Name of new coordinate axis
        :type coordname: str
        :param coordvals: Values of new coordinates in axis
        :type coordvals: u.Quantity
        '''
        self.gains = self.gains.assign_coords({coordname:validate_axis(coordvals,
                                                                       physical_type,
                                                                       unit)})


    def del_coords(self, coordname:str) -> None:
        '''
        Remove coordinate axis specified by coordname.
        
        :param self: xJones object
        :param coordname: name of coordinate axis to remove
        :type coordname: str
        '''
        self.gains=self.gains.drop_dims(coordname)

    def add_attr(self,
                 attrname:str,
                 attrvals:Any,
                 ) -> None:
        self.gains = self.gains.assign_attrs({attrname:attrvals})

    def del_attr(self,
                 attrname:str) -> None:
        attrs = self.gains.attrs
        del attrs[attrname]
        self.gains = self.gains.assign_attrs(attrs)

    @classmethod
    def from_dicosols(cls,
                     filename:str):
        """Read a killMS .sols.npz file and create an xSols object."""
        ### load dicosols object
        fname = Path(filename).absolute().as_posix()
        data = np.load(fname, allow_pickle=True)
        ### extract dicosols metadata
        # xsols metadata
        msname = str(data['MSName'])
        # new metadata items. add to xjones metadata in post.
        msname_time0 = float(data['MSNameTime0']) 
        beam_times = data['BeamTimes']
        ### extra dicosols data
        sols_struct = data['Sols']
        ### axes information
        # build dummy direction information, as pointings are not contained in dicosols afaik
        ndir = sols_struct['G'].shape[3] ### NOT!!! SHAPE 0 !!!! for ease of application, I suspect
        directions=[]
        for i in range(ndir):
            directions.append(f"dir{i:02d}")
        directions=np.array(directions)
        # extract antenna information.
        station_names = np.array(data['StationNames'],dtype=np.str_)
        ### time
        # Extract time intervals (t0 and t1 are scalars in the structured array)
        gains_t0 = validate_axis(np.array([float(s['t0']) for s in sols_struct]),'time',u.s)
        gains_t1 = validate_axis(np.array([float(s['t1']) for s in sols_struct]),'time',u.s)
        # assume timestamps are halfway between t0 and t1 values.
        times = validate_axis(0.5 * (gains_t0 + gains_t1),'time',u.s)
        ### freq
        freqs = validate_axis(np.array(data['FreqDomains']),'freq',u.Hz)
        # extract freq intervals
        freqs_center = np.mean(freqs, axis=1)
        gains_nu0 = validate_axis(freqs[:, 0],'freq',u.Hz)
        gains_nu1 = validate_axis(freqs[:, 1],'freq',u.Hz)
        ### read gains and stats
        gains = sols_struct['G']
        stats = sols_struct['Stats']        
        ### Reshape gain, stats values
        # Sols has fields: t0, t1, G, Stats
        # t0 and t1 are scalar floats for each time slot
        # G has shape (n_freqs, n_antennas, n_dirs, 2, 2)  per time slot
        # Stats has shape ....
        # We want shape (n_dirs, n_times, n_freqs, n_antennas, n_params)
        gains_list = []
        stats_list = []
        for s in sols_struct:
            gains = s['G']     # shape (n_freqs, n_antennas, n_dirs, 2, 2)
            stats = s['Stats'] # shape (n_times, n_antennas, 4)
            gains_list.append(gains)
            stats_list.append(stats)
        gains = np.array(gains_list) # shape (n_times, n_freqs, n_ants, n_dirs, 2,2 )
        stats = np.array(stats_list) # shape (n_times, n_freqs, n_ants, 4 )
        # reshape to xarray style
        gains = gains.transpose(3, 2, 0, 1, 4, 5).reshape(len(directions), len(station_names), len(times), len(freqs), 4)
        stats = stats.transpose(2, 0, 1, 3) # (len(station_names), len(times), len(freqs), 4)
        # Set defaults for optional fields
        name = Path(filename).stem.replace("killMS.","").replace(".sols","")
        comments = np.array([f"Gains loaded from dicosol {filename}."])
        # do not specify solve mode, because it is not inferred.
        # DicoSols are always 2,2 Jones matrices, even when solved in Scalar or IDiag.
        gaintype="JonesMatrix" 
        # There does not seem to be metadata recording the correlator basis.
        # I guess, for now, let's assume it's XY, since it's LOFAR software.
        params=np.array(["XX","XY","YX","YY"])
        # Create initial xjonesinstance with attrs
        output = cls(name=name,
                    gaintype=gaintype,
                    directions=directions,
                    antennas=station_names,
                    times=times,
                    freqs=freqs_center,
                    params=params,
                    msname=msname,
                    comments=comments)
        # assign dicosols-specific coordinates the times and freq axes
        output.gains = output.gains.assign_coords({
                                                "gains_t0": ("Times", gains_t0),
                                                "gains_t1": ("Times", gains_t1),
                                                "gains_nu0": ("Freqs", gains_nu0),
                                                "gains_nu1": ("Freqs", gains_nu1)
                                                })
        # add dicosols-specific metadata
        output.add_attr("t0",msname_time0)
        output.add_attr("BeamTimes",beam_times)
        ### TODO: the three below will require making a nice SkyModel dataset.
        output.add_attr("SkyModel",data["SkyModel"]) # TODO: make this into a nice dataset.
        output.add_attr("ClusterCat",data["ClusterCat"]) # TODO: make this into a nice dataset.
        output.add_attr("SourceCatSub",data["SourceCatSub"]) # TODO: make this into a nice dataset.
        output.add_attr("ModelName",data["ModelName"])
        # remove the default empty DataArray, unused here
        del(output.gains[name])
        # save the gains and stats in the dataset
        output.gains[name+"_gains"] = (("Direction", "Antennas", "Times", "Freqs", "Params"), gains)
        output.gains[name+"_stats"] = (("Antennas", "Times", "Freqs", "Params"), stats)

        return output


    def to_dicosols(self,
                    filename:str="",
                    return_dico:bool=True,
                    write_to_file:bool=False
                    ) -> Optional[dict]:
        '''
        Function to convert this instance of an
        xJones class to a DicoSols object. This object
        can be written it to a file, with the default
        using the xJones name as a DicoSols basename.
        Otherwise, the function returns the DicoSols
        to be manipulated in memory.

        Description of a DicoSols object:
          It is an NpzFile object containing multiple files.
          These are, in order:
            MSName: np.ndarray[np.str_]
              contains the name of the MS these gains are derived from
            MSNameTime0: np.ndarray[float]
              contains the lowest time value of the MS for these gains, in MJDs
            Sols: np.ndarray
              contains the gain solutions. 
              dtype=[('t0', '<f8'), 
                     ('t1', '<f8'), 
                     ('G', '<c8', (n_freqs, n_antennas, n_dirs, 2, 2)), 
                     ('Stats', '<f4', (n_times, n_antennas, 4))]=
            StationNames: np.ndarray[np.str_]
              array of station name strings
            SkyModel: np.ndarray
              contains the pybdsf-like SkyModel array
                dtype=[('Name', 'S200'), ('ra', '<f4'), ('dec', '<f4'), ('SumI', '<f4'), ('Cluster', '<i8'), ('l', '<f4'), ('m', '<f4')]
            ClusterCat: np.ndarray
              contains the pybdsf-like ClusterCat array
                b'str', float, float, float, int, float, float
                dtype=[('Name', 'S200'), ('ra', '<f4'), ('dec', '<f4'), ('SumI', '<f4'), ('Cluster', '<i8'), ('l', '<f4'), ('m', '<f4')]
            SourceCatSub: np.ndarray[Object]
              contains the catalog of sources to be subtracted during FreeFullSub
            ModelName: np.ndarray[np.str_]
              contains the filename for the model used to compute these gains
            FreqDomains: np.ndarray[np.float64]
              contains the nu0, nu1 values for the gains
            BeamTimes: np.ndarray[np.float64]
              contains the times at which the instrument beam values are computed

        If DicoSols fields are missing from the xJones
        object, they are filled with dummy values.
        
        :param self: xJones object
        :param return_dico: Flag to return the DicoSols object. Defaults to True.
        :type return_dico: bool
        :param write_to_file: flag to write the DicoSols object to file. Defaults to False.
        :type write_to_file: bool
        :param filename: Provide a name for your DicoSols. Only used if write_to_file is True. xJones named is used if this is not provided.
        :type filename: str
        '''
        ### check for the specific values we cannot autogenerate:
        ### gains t0, t1, nu0, nu1. Return an exception if they
        ### are missing.
        # gain sols
        if 'gains_t0' not in self.gains.coords:
            raise MissingDicoSolsKey(f"Serialisation of {self.name} requires \'gains_t0\' coordinate.")
        if 'gains_t1' not in self.gains.coords:
            raise MissingDicoSolsKey(f"Serialisation of {self.name} requires \'gains_t1\' coordinate.")
        if 'gains_nu0' not in self.gains.coords:
            raise MissingDicoSolsKey(f"Serialisation of {self.name} requires \'gains_nu0\' coordinate.")
        if 'gains_nu1' not in self.gains.coords:
            raise MissingDicoSolsKey(f"Serialisation of {self.name} requires \'gains_nu1\' coordinate.")
        ### read values from DataSet, add defaults if missing.
        # msname
        MSName = np.array(self.msname)
        # t0
        if "t0" in self.gains.attrs:
            MSNameTime0=np.array(self.gains.attrs['t0'])
        else:
            MSNameTime0 = np.array([])
        ### Reshape gain, stats values
        # Sols has fields: t0, t1, G, Stats
        # t0 and t1 are scalar floats for each time slot
        # G has shape (n_freqs, n_antennas, n_dirs, 2, 2)  per time slot
        # We want shape (n_dirs, n_times, n_freqs, n_antennas, n_params)
        ## read t0, t1 from the time coordinate axis
        t0=self.gains.coords['gains_t0'].values
        t1=self.gains.coords['gains_t1'].values
        # gains_list = []
        # stats_list = []
        # for s in sols_struct:
        #     gains = s['G']     # shape (n_freqs, n_antennas, n_dirs, 2, 2)
        #     stats = s['Stats'] # shape (n_times, n_antennas, 4)
        #     gains_list.append(gains)
        #     stats_list.append(stats)
        # gains = np.array(gains_list) # shape (n_times, n_freqs, n_ants, n_dirs, 2,2 )
        # stats = np.array(stats_list) # shape (n_times, n_freqs, n_ants, 4 )
        # # reshape to xarray style
        # gains = gains.transpose(3, 2, 0, 1, 4, 5).reshape(len(directions), len(station_names), len(times), len(freqs), 4)
        # stats = stats.transpose(2, 0, 1, 3) # (len(station_names), len(times), len(freqs), 4)

        # target gain shape in the dicosols:
        # times, freqs, ants, corrs
        Stats = self.gains[self.name+"_stats"].transpose('Times','Freqs','Antennas','Params').values
        ntimes, nfreqs, nants, _ = Stats.shape
        ndir = len(self.gains['Direction'])
        # target gain shape in the dicosols:
        # times, freqs, ants, dir, 2,2
        G = self.gains[self.name+"_gains"].transpose('Times', 'Freqs','Antennas','Direction','Params').values.reshape(ntimes,nfreqs,nants,ndir,2,2)
        # build the dicosols Sol structure
        sol_dtype = np.dtype([
                            ('t0', np.float64),
                            ('t1', np.float64),
                            ('G', np.complex64, (nfreqs, nants, ndir, 2, 2)),
                            ('Stats', np.float32, (nfreqs, nants, 4))
                            ])
        # build the Sols key
        Sols = np.array(
                        [(t0[i], t1[i], G[i], Stats[i]) for i in range(ntimes)],
                        dtype=sol_dtype
                        )
        # station names
        ## get from attributes
        StationNames=self.gains.coords['Antennas'].to_numpy()
        ### skymodel-like dtype
        cat_dtype = np.dtype([
                            ('Name', '|S200'),
                            ('ra', np.float32),
                            ('dec', np.float32),
                            ('SumI', np.float32),
                            ('Cluster', np.int64),
                            ('l', np.float32),
                            ('m', np.float32)
                            ])
        # skymodel
        if "SkyModel" in self.gains.attrs:
            SkyModel = np.asarray(self.gains.attrs["SkyModel"], dtype=cat_dtype)
        else:
            SkyModel = np.array(None, dtype=object)
        # clustercat
        if "ClusterCat" in self.gains.attrs:
            ClusterCat = np.asarray(self.gains.attrs["ClusterCat"], dtype=cat_dtype)
        else:
            ClusterCat=np.array(None, dtype=object)
        # sourcecatsub - this one can be empty!
        if "SourceCatSub" in self.gains.attrs:
            if self.gains.attrs["SourceCatSub"]==None:
                SourceCatSub=np.array(None, dtype=object)
            else:
                SourceCatSub = np.asarray(self.gains.attrs["SourceCatSub"], dtype=cat_dtype)
        else:
            SourceCatSub=np.array(None, dtype=object)
        # modelname
        if "ModelName" in self.gains.attrs:
            ModelName=np.array(self.gains.attrs["ModelName"])
        else:
            ModelName=np.array(NotImplemented)
        # freqdomains
        ## read these from the freq coordinate axis
        nu0=self.gains.coords['gains_nu0'].values
        nu1=self.gains.coords['gains_nu1'].values
        FreqDomains=np.array([nu0,nu1]).transpose(1,0)
        # beamtimes
        if "BeamTimes" in self.gains.attrs:
            BeamTimes=np.array(self.gains.attrs["BeamTimes"])
        else:
            BeamTimes=np.array()
        # then, build the dicosols
        dico = kMSDicoSols(MSName,
                           MSNameTime0,
                           Sols,
                           StationNames,
                           SkyModel,
                           ClusterCat,
                           SourceCatSub,
                           ModelName,
                           FreqDomains,
                           BeamTimes)

        if write_to_file:
            # build default filename if not provided
            if filename!="":
                filename = f"{self.name}.sols.npz"
            np.savez(filename, **dico, pickle=True)
        if return_dico:
            return dico


################### we are here ######################
#     def to_solsnpz(self,
#                    filename):
#         """Write to killMS sols.npz format."""
#         fname = Path(filename).absolute().as_posix()
        
#         # Prepare the structured array for Sols
#         # Sols has fields: t0, t1, G, Stats
#         n_times = len(self.times)
#         n_antennas = len(self.antennas)
        
#         # Create dtype for Sols structured array
#         # G has shape (1, n_antennas, 1, 2, 2)
#         # Stats has shape (1, n_antennas, 4) - we'll use zeros for now
#         sol_dtype = np.dtype([
#             ('t0', np.float64),
#             ('t1', np.float64),
#             ('G', np.complex64, (1, n_antennas, 1, 2, 2)),
#             ('Stats', np.float32, (1, n_antennas, 4))
#         ])
        
#         # Build Sols array
#         sols_list = []
#         for t_idx in range(n_times):
#             t0 = self.gains_t0[t_idx]
#             t1 = self.gains_t1[t_idx]
            
#             # Extract gains for this time slot: shape (n_antennas, 2, 2)
#             g = self.gains[t_idx, :, :, :]  # shape (n_antennas, 2, 2)
            
#             # Reshape to (1, n_antennas, 1, 2, 2)
#             g = g[np.newaxis, :, np.newaxis, :, :]  # shape (1, n_antennas, 1, 2, 2)
            
#             # Convert to complex64
#             g = g.astype(np.complex64)
            
#             # Stats - use zeros for now
#             stats = np.zeros((1, n_antennas, 4), dtype=np.float32)
            
#             sols_list.append((t0, t1, g, stats))
        
#         sols_array = np.array(sols_list, dtype=sol_dtype)
        
#         # Prepare other arrays
#         # FreqDomains: shape (n_freqs, 2)
#         if hasattr(self, 'freqs') and self.freqs is not None:
#             if hasattr(self, 'gains_nu0') and hasattr(self, 'gains_nu1'):
#                 freqs = np.stack([self.gains_nu0, self.gains_nu1], axis=1)
#             else:
#                 # Use freqs as center and create a small bandwidth
#                 freqs = np.column_stack([self.freqs - 1e6, self.freqs + 1e6])
#         else:
#             freqs = np.array([[1.0e8, 2.0e8]], dtype=np.float64)  # default
        
#         # Save to npz file
#         np.savez(
#             fname,
#             MSName=self.msname,
#             MSNameTime0=np.float64(0.0),  # We don't have this info
#             Sols=sols_array,
#             StationNames=self.antennas,
#             SkyModel=np.array([], dtype=object),  # Placeholder
#             ClusterCat=np.array([], dtype=object),  # Placeholder
#             SourceCatSub=np.array([], dtype=object),  # Placeholder
#             ModelName=np.array(self.name, dtype='<U18'),
#             FreqDomains=freqs,
#             BeamTimes=np.array([], dtype=np.float64)
#         )
        
#         return fname

    def to_dict(self, 
                include_data:bool=False) -> dict:
        '''
        Converts xJones object to a dictionary for JSON serialization.
        
        :param self: xJones object
        :param include_data: set to True to include image data in the dict itself. Defaults to False
        :type include_data: bool
        :return: Dictionary representation current xJones object
        :rtype: dict
        '''
        jones_dict = {
            "metadata": {
                "name": self.name,
                "gaintype": self.gaintype,
                "msname": self.msname,
                "comments": self.comments.tolist(),
            },
            "coordinates": {},
            "data": {},
            "attrs": {k: str(v) for k, v in self.gains.attrs.items()}
        }

        # Serialize coordinates
        for coord_name, coord in self.gains.coords.items():
            if hasattr(coord, 'values'):
                if hasattr(coord, 'unit'):
                    jones_dict["coordinates"][coord_name] = {
                                "values": serialize_complex(coord.values),
                                "unit": str(coord.unit)
                    }
                else:
                    jones_dict["coordinates"][coord_name] = serialize_complex(coord.values)
            else:
                jones_dict["coordinates"][coord_name] = str(coord)

        # Serialize data variables if requested
        if include_data==True:
            for var_name in self.gains.data_vars:
                var = self.gains[var_name]
                jones_dict["data"][var_name] = {
                            "values": serialize_complex(var.values),
                            "dims": list(var.dims)
                }
        return jones_dict








    def to_json(self,
                filename: str, 
                include_data:bool=False,
                overwrite:bool=True,
                indent:int=2) -> None:
        '''
         Serialise xJones object to JSON file.
        
        :param self: xJones object
        :param filename: Output filename
        :type filename: str
        :param include_data: include data values in JSON file. Default False.
        :type include_data: bool
        :param overwrite: overwrite existing file if present. Default True.
        :type overwrite: bool
        :param indent: JSON indentation level. Default 2.
        :type indent: int
        '''
        filepath = Path(filename)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Set overwrite=True to overwrite.")
        jones_dict = self.to_dict(include_data=include_data)
        with open(filepath, 'w') as f:
            json.dump(jones_dict, f, indent=indent)
        

def serialize_complex(arr:np.ndarray[Any]):
    '''
    Splits complex numpy array to allow JSON serialisation.
    Returns a list, split into real and imag if values are complex.
    
    :param arr: Numpy-like array of complex values to split
    :type arr: np.ndarray[np.complexfloating]
    '''
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return {
            "real": np.real(arr).tolist(),
            "imag": np.imag(arr).tolist(),
            "_complex_": True
        }
    return arr.tolist()


# ### build sols protocol
# @runtime_checkable
# class SolsArrayProtocol(Protocol):
#     '''
#     Protocol describing the contents of DicoSols Sols
#     fields: t0, t1, G, Stats
#     '''
#     dtype: np.dtype
#     @property
#     def t0(self) -> np.typing.NDArray[np.float64]: ...
#     @property
#     def t1(self) -> np.typing.NDArray[np.float64]: ...
#     @property
#     def G(self) -> np.typing.NDArray[np.complex64]: ...
#     @property
#     def Stats(self) -> np.typing.NDArray[np.float32]: ...

# def validate_sols(arr:np.typing.NDArray[np.record]) -> bool:
#     """
#     Validate a numpy array conforms to Sols structure.
    
    
#     :param arr: Input DicoSols array to be validated
#     :type arr: np.typing.NDArray[np.record]
#     :return: True if DicoSols is compliant
#     :rtype: bool
#     """
#     return (arr.dtype.names == ('t0', 't1', 'G', 'Stats') and
#             arr.dtype['t0'] == np.float64 and
#             arr.dtype['t1'] == np.float64 and
#             arr.dtype['G'].kind == 'c' and  # complex
#             arr.dtype['Stats'].kind == 'f')  # float


# ### build dicocat protocol
# @runtime_checkable
# def DicoCatProtocol(Protocol):
#     '''
#     Protocol for DicoSols catalog-like structured array
#     fields: Name, ra, dec, SumI, Cluster, l, m
#     '''
#     dtype: np.dtype
#     @property
#     def Name(self) -> np.typing.NDArray[np.str_]: ...
#     @property
#     def ra(self) -> np.typing.NDArray[np.float32]: ...
#     @property
#     def dec(self) -> np.typing.NDArray[np.float32]: ...
#     @property
#     def SumI(self) -> np.typing.NDArray[np.float32]: ...
#     @property
#     def Cluster(self) -> np.typing.NDArray[np.int64]: ...
#     @property
#     def l(self) -> np.typing.NDArray[np.float32]: ...
#     @property
#     def m(self) -> np.typing.NDArray[np.float32]: ...

# def ValidateDicoCat(arr:np.typing.NDArray[np.record]) -> bool:
#     '''
#     Validator for DicoCat-like structured arrays
    
#     :param arr: Input array to validate
#     :type arr: np.typing.NDArray[np.record]
#     :return: True if it's compliant
#     :rtype: bool
#     '''
#     return (arr.dtype.names == ('Name', 'ra', 'dec', 'SumI', 'Cluster', 'l', 'm') and
#             arr.dtype['Name'].kind == 'U' and  # unicode string
#             arr.dtype['ra'] == np.float32 and
#             arr.dtype['dec'] == np.float32 and
#             arr.dtype['SumI'] == np.float32 and
#             arr.dtype['Cluster'] == np.int64 and
#             arr.dtype['l'] == np.float32 and
#             arr.dtype['m'] == np.float32)

def kMSDicoSols(MSName: np.typing.NDArray[np.str_],
             MSNameTime0: np.typing.NDArray[np.float64],
             Sols:np.typing.NDArray[np.void], # solstype
             StationNames: np.typing.NDArray[np.str_],
             SkyModel: np.typing.NDArray[np.record], # cattype
             ClusterCat: np.typing.NDArray[np.record], # cattype
             SourceCatSub: np.typing.NDArray[np.record], # cattype
             ModelName: np.typing.NDArray[np.str_],
             FreqDomains:np.typing.NDArray[np.float64],
             BeamTimes: np.typing.NDArray[np.float64]
             ):
    
    '''
    Builds a killMS DicoSols object out of its input ndarrays.
    It is an NpzFile object containing multiple files, each of
    which corresponds to one of the parameters listed, in order.
    
    :param MSName: contains the name of the MS these gains are derived from
    :type MSName: np.ndarray[np.str_]
    :param MSNameTime0: contains the lowest time value of the MS for these gains, in MJDs
    :type MSNameTime0: np.ndarray[np.float64]
    :param Sols: contains the gain solutions and statistics. 
                 Should follow SolsArrayProtocol, i.e. the following: 
                    dtype=[('t0', '<f8'), 
                            ('t1', '<f8'), 
                            ('G', '<c8', (n_freqs, n_antennas, n_dirs, 2, 2)), 
                            ('Stats', '<f4', (n_times, n_antennas, 4))]
    :type Sols: np.ndarray[np.float64, np.float64, np.ndarray[np.complex64], np.ndarray[np.float32]]
    :param StationNames: array of station name strings
    :type StationNames: np.ndarray[np.str_]

    :param SkyModel: contains the pybdsf-like SkyModel array.
                     Should follow DicoCatProtocol, i.e. the following:
                       b'str', float, float, float, int, float, float
                       dtype=[('Name', 'S200'), ('ra', '<f4'), ('dec', '<f4'), ('SumI', '<f4'), ('Cluster', '<i8'), ('l', '<f4'), ('m', '<f4')]
    :type SkyModel: np.ndarray[Tuple[str, float, float, float, int, float, float]]
    :param ClusterCat: contains the pybdsf-like ClusterCat array
                       Should follow DicoCatProtocol, i.e. the following:
                         b'str', float, float, float, int, float, float
                         dtype=[('Name', 'S200'), ('ra', '<f4'), ('dec', '<f4'), ('SumI', '<f4'), ('Cluster', '<i8'), ('l', '<f4'), ('m', '<f4')]
    :type ClusterCat: np.ndarray
    :param SourceCatSub: contains the catalog of sources to be subtracted during FreeFullSub
                         Should follow DicoCatProtocol, i.e. the following:
                           b'str', float, float, float, int, float, float
                           dtype=[('Name', 'S200'), ('ra', '<f4'), ('dec', '<f4'), ('SumI', '<f4'), ('Cluster', '<i8'), ('l', '<f4'), ('m', '<f4')]
    :type SourceCatSub: np.ndarray[...]
    :param ModelName: contains the filename for the model used to compute these gains
    :type ModelName: np.ndarray[np.str_]
    :param FreqDomains: contains the nu0, nu1 values for the gains
    :type FreqDomains: np.ndarray[np.float64]
    :param BeamTimes: contains the times at which the instrument beam values are computed
    :type BeamTimes: np.ndarray[np.float64]
    
    '''
    return {
        'MSName': MSName,
        'MSNameTime0': MSNameTime0,
        'Sols': Sols,
        'StationNames': StationNames,
        'SkyModel': SkyModel,
        'ClusterCat': ClusterCat,
        'SourceCatSub': SourceCatSub,
        'ModelName': ModelName,
        'FreqDomains': FreqDomains,
        'BeamTimes': BeamTimes
    }
