"""Declared MEG phase extraction; no input discovery or downloads."""
import numpy as np
from scipy.signal import resample_poly
def meg_current_windows(data,times):
    data=np.asarray(data);times=np.asarray(times)
    assert data.ndim==3 and data.shape[1]==306 and data.shape[2]==len(times)
    assert np.isfinite(data).all() and np.all(np.diff(times)>0)
    phases=[]
    for low,high in [(0,2),(2,4)]:
        ix=np.flatnonzero((times>=low)&(times<high));assert len(ix)==200
        a=np.asarray(data[:,:,ix],dtype=np.float64)
        phases.append(resample_poly(a,up=2,down=5,axis=2).reshape(len(a),24480).astype(np.float32))
    return tuple(phases)
