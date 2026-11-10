"""Positive velocity-dependent damping approximation of directional dry resistance."""
import numpy as np


def damping(velocity,positive,negative,scale):
    """Friction is h(v)*tanh(v/scale), evaluated as nonnegative implicit damping."""
    v,positive,negative,scale=np.broadcast_arrays(velocity,positive,negative,scale)
    magnitude=np.where(v>0,positive,np.where(v<0,negative,(positive+negative)/2))
    ratio=np.empty_like(v,dtype=float)
    np.divide(np.tanh(v/scale),v,out=ratio,where=abs(v)>1e-10)
    ratio=np.where(abs(v)>1e-10,ratio,1./scale)
    return magnitude*ratio
