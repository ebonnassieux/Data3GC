# Defines Jones object properties.
from __future__ import annotations
from typing import TypedDict, Required, Optional, NotRequired, Protocol, Self, cast
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy.typing
import xarray as xr
from attrs import define, field
import numpy as np
from pathlib import Path


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
                 directions:Optional[np.str_] = None,
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
        if np.isscalar(directions) :
            dirs=np.array([directions])
            dirs = cast(np.ndarray,dirs)
            # cast(u.Quantity, input_var)
        if np.isscalar(params):
            params=np.array([params])
        # validate times, freqs dimensionality
        times = self.validate_axis(times, 'time', u.s)
        freqs = self.validate_axis(freqs, 'frequency', u.Hz)
        # build xarray shape
        xshape = (len(dirs),len(antennas),len(times),len(freqs),len(params))
        print(xshape)
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

    def validate_axis(self,
                      input_var,
                      desired_physical_type:str,
                      default_unit:u.Unit) -> u.Quantity:
        # since u.Quantity doesn't say if it's scalar or array, check that first.
        if isinstance(input_var,u.Quantity):
            output_var = cast(u.Quantity, input_var) # inform pylance
            # check if it is scalar or vector
            if output_var.isscalar:
                output_var = np.array([output_var])
            # check if it has correct unit physical type
            if output_var.unit.physical_type != desired_physical_type:
                output_var << default_unit
            # if it does, cast to default unit (e.g. seconds rather than minutes)
            output_var = input_var.to(default_unit)
        else:
            # if it is not a quantity, make it into an array if it is not already
            if np.isscalar(input_var):
                output_var=np.array([input_var])
            elif not isinstance(input_var, np.ndarray):
                output_var=np.asarray(input_var)
            # not a quantity to begin with; just add requested units
            output_var = cast(u.Quantity, output_var << default_unit)
        return output_var


################### we are here ######################









#     @classmethod
#     def from_solsnpz(cls,
#                      filename):
#         """Read a killMS .sols.npz file and create an xSols object."""
#         fname = Path(filename).absolute().as_posix()
#         data = np.load(fname, allow_pickle=True)
        
#         # Extract metadata
#         msname = str(data['MSName'])
#         msname_time0 = float(data['MSNameTime0'])
#         station_names = data['StationNames']
#         freqs = data['FreqDomains']
#         beam_times = data['BeamTimes']
        
#         # Extract solution data
#         sols_struct = data['Sols']
        
#         # Parse the structured array
#         # Sols has fields: t0, t1, G, Stats
#         # t0 and t1 are scalar floats for each time slot
#         # G has shape (1, n_antennas, 1, 2, 2) - complex64 Jones matrices
        
#         n_times = len(sols_struct)
#         n_antennas = len(station_names)
        
#         # Extract time intervals (t0 and t1 are scalars in the structured array)
#         gains_t0 = np.array([float(s['t0']) for s in sols_struct])
#         gains_t1 = np.array([float(s['t1']) for s in sols_struct])
        
#         # Extract gain values
#         # G has shape (1, n_antennas, 1, 2, 2) per time slot
#         # We want shape (n_times, n_antennas, 2, 2)
#         gains_list = []
#         for s in sols_struct:
#             g = s['G'][0]  # shape (n_antennas, 1, 2, 2)
#             g = g[:, 0, :, :]  # shape (n_antennas, 2, 2)
#             gains_list.append(g)
#         gains = np.stack(gains_list, axis=0)  # shape (n_times, n_antennas, 2, 2)
        
#         # For times coordinate, use midpoint of intervals
#         times = (gains_t0 + gains_t1) / 2
        
#         # For freqs, use midpoint of frequency domains
#         # freqs shape is (n_freq_domains, 2)
#         freqs_center = np.mean(freqs, axis=1)
        
#         # Set defaults for optional fields
# #        directions = None  # Direction-independent gains for now
#         name = Path(filename).stem
#         comments = ""
        
#         # Add frequency interval info if available
#         if freqs.shape[1] == 2:
#             gains_nu0 = freqs[:, 0]
#             gains_nu1 = freqs[:, 1]
#         else:
#             gains_nu0 = freqs_center
#             gains_nu1 = freqs_center
        
#         # Create xSols instance with attrs
#         return cls(
#             name=name,
#             msname=msname,
#             comments=comments,
#             gaintype="full-jones",
# #            directions=directions,
#             antennas=station_names,
#             times=times,
#             gains_t0=gains_t0,
#             gains_t1=gains_t1,
#             freqs=freqs_center,
#             gains_nu0=gains_nu0,
#             gains_nu1=gains_nu1,
#             gains=gains,
#         )
        
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