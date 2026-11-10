"""Bounded conservative load correction for fixed-base diagnostic candidates."""
import numpy as np


def evaluate(q,coefficients):
    """Return U(q) and dU/dq; coefficients never depend on time or future poses."""
    q=np.asarray(q);c=np.broadcast_to(np.asarray(coefficients),q.shape[:-1]+(2,8))
    energy=np.zeros(q.shape[:-1]);force=np.zeros_like(q,dtype=float)
    for side in range(2):
        a=q[...,2*side];b=q[...,2*side+1];p=c[...,side,:]
        terms=[a,b,a-b,a+b];derivatives=[(1.,0.),(0.,1.),(1.,-1.),(1.,1.)]
        for j,(angle,(da,db)) in enumerate(zip(terms,derivatives)):
            # 周期势能有界；其梯度只改变保守负载，不冒充耗散摩擦。
            energy+=-p[...,2*j]*np.cos(angle)+p[...,2*j+1]*np.sin(angle)
            torque=p[...,2*j]*np.sin(angle)+p[...,2*j+1]*np.cos(angle)
            force[...,2*side]+=da*torque;force[...,2*side+1]+=db*torque
    return energy,force
