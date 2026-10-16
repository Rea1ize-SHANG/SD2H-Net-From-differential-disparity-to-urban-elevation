import numpy as np
from scipy.optimize import least_squares
import rpcm.rpc_model as rm

def localize(self, col, row, alt, return_normalized=False):
    ca,ra,aa=np.broadcast_arrays(np.asarray(col,float),np.asarray(row,float),np.asarray(alt,float)); shape=ca.shape
    out=np.empty(ca.size*2); eps=1e-5
    for i,(c,r,a) in enumerate(zip(ca.ravel(),ra.ravel(),aa.ravel())):
        nc=(c-self.col_offset)/self.col_scale; nr=(r-self.row_offset)/self.row_scale; na=(a-self.alt_offset)/self.alt_scale
        def fun(v):
            return np.array([rm.apply_rfm(self.col_num,self.col_den,v[1],v[0],na)-nc,rm.apply_rfm(self.row_num,self.row_den,v[1],v[0],na)-nr])
        sol=least_squares(fun,[0.,0.],jac='3-point',xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
        if not sol.success or np.linalg.norm(fun(sol.x))>1e-7: raise rm.MaxLocalizationIterationsError('GF7 inverse failed')
        out[2*i:2*i+2]=sol.x
    lon=out[::2].reshape(shape); lat=out[1::2].reshape(shape)
    if not return_normalized: lon=lon*self.lon_scale+self.lon_offset; lat=lat*self.lat_scale+self.lat_offset
    return lon,lat
rm.RPCModel.localization=localize
