
try:
    import tables as tb
except:
    pass
import numpy as np


# Support files classes        
    

class MemmapHelper:
    """Class wrapping and easing memory-map management.
    """
    
    def __init__(self, filename, dtype, shape):
        """Constructor
        """
        
        self.filename = filename
        self.dtype = dtype
        self.shape = shape
        
    
    def create(self):
        """Create and return a memory-map object.
        """
        
        return np.memmap(self.filename, dtype=self.dtype, mode='w+', shape=self.shape)
        
        
    def read(self):
        """Read and return an existing memory-map object from file.
        """
        
        return np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=self.shape)
    
    
    def setDType(self, dtype):
        """Set dtype.
        
        :param str dtype: anydata type supperted by Numpy arrays.
        """
        
        self.dtype = dtype
        



class CArrayHelper:
    """Class wrapping and easing PyTables CArray management.
    """
    
    def __init__(self, filename, dtype, shape, chunkshape=None):
        """Constructor
        """
        
        self.filename = filename
        self.dtype = dtype
        self.shape = shape
        self.chunkshape = chunkshape
        self.h5file = None
        
    
    def create(self):
        """Create and return a CArray object.
        """
        
        self.h5file = tb.openFile(self.filename, mode='w', title="Support array")
        atom = self._dtype2atom(self.dtype)
        x = self.h5file.createCArray(self.h5file.root, 'x', atom, shape=self.shape, chunkshape=self.chunkshape)
        return x
        
        
    def read(self):
        """Read and return an existing CArray object from file.
        """
        
        self.h5file = tb.openFile(self.filename, mode='r+')
        x = self.h5file.root.x
        return x
        
    
    def close(self):
        """Close support file (opened by creating or reading it)
        """
        
        self.h5file.close()
        
        
    def _dtype2atom(self, dtype):
        
        if dtype == 'uint8':
            atom = tb.UInt8Atom()
        
        return atom