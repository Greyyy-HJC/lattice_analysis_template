"""
All prior dicts for z-dependence extrapolation should be defined here.
"""
import gvar as gv

def large_z_extrapolation_prior():
    priors = gv.BufferDict()
    priors['c1'] = gv.gvar(1, 10)
    priors['c2'] = gv.gvar(1, 10)
    priors['n1'] = gv.gvar(1, 10)
    priors['n2'] = gv.gvar(1, 10)
    priors['lam0'] = gv.gvar(100, 100)

    return priors