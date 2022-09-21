import numpy as np


def read_npy(path: str) -> np.ndarray:
    """Reads file in binary format and loads it using np.open function"""
    
    
    with open(path, "rb") as handler:
        data = np.load(handler)
        
    return data