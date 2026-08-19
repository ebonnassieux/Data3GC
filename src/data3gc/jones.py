# Defines Jones object properties.
from __future__ import annotations
from typing import Any, Optional, NotRequired, Protocol, Self, cast
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy.typing
import xarray as xr
from attrs import define, field
import numpy as np
from pathlib import Path



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
        ndir = sols_struct.shape[0]
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
        # extract param intervals. TODO
        gains = sols_struct['G']
        stats = sols_struct['Stats']        
        ### Reshape gain, stats values
        # Sols has fields: t0, t1, G, Stats
        # t0 and t1 are scalar floats for each time slot
        # G has shape (n_freqs, n_antennas, n_dirs, 2, 2)  per time slot
        # We want shape (n_dirs, n_times, n_freqs, n_antennas, n_params)
        gains_list = []
        for s in sols_struct:
            g = s['G'][0]  # shape (n_antennas, 1, 2, 2)
            g = g[:, 0, :, :]  # shape (n_antennas, 2, 2)
            gains_list.append(g)
        gains = np.stack(gains_list, axis=0)  # shape (n_times, n_antennas, 2, 2)
        
        # Set defaults for optional fields
        name = Path(filename).stem
        comments = np.array([f"Gains loaded from dicosols {name}."])
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
        
        # add stats xr.DataArray to dataset
        ...
        
        # add dicosols-specific axes and coords
        output.add_coords("gains_t0",
                           gains_t0,
                           physical_type='time',
                           unit=u.s)
        output.add_coords("gains_t1",
                           gains_t1,
                           physical_type='time',
                           unit=u.s)
        output.add_coords("gains_nu0",
                           gains_nu0,
                           physical_type='freq',
                           unit=u.Hz)
        
        output.add_coords("gains_nu1",
                           gains_nu1,
                           physical_type='freq',
                           unit=u.Hz)
        
        # add dicosols-specific metadata
        output.add_attr("t0",msname_time0)
        output.add_attr("BeamTimes",beam_times)

        return output



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