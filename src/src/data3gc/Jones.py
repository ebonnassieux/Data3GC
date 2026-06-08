# Defines Jones object properties.

class Jones:
    '''
    A class that describes the various antenna-based signal propagation effects 
    which we want to either model or solve for.
    '''
    ### constructor
    def __init__(self,
                 values       : np.ndarray,
                 freqs        : list[u.Quantity],
                 times        : list[u.Quantity],
                 antnames     : list[str],
                 basis        : str="XY"|"RL",
                 name         : str="Jones",
                 jonestype    : str="Scalar"|"Diag"|"Full"
#                 directions   : Sky
                 ):
        self.values    = values
        self.antnames  = antnames
        self.nants     = len(antnames)
        self.freqs     = freqs
        self.nchan     = len(freqs)
        self.times     = times
        self.ntimes    = len(times)
        self.basis     = basis
        self.name      = name
        self.jonestype = jonestype
        self.ndir      = ndir