from fgbuster.observation_helpers import get_instrument, get_sky, get_observation
from fgbuster.component_model import CMB, Dust, Synchrotron

from fgbuster.separation_recipes import  adaptive_comp_sep

import healpy as hp
import numpy as np
import pysm


from .utils import check_nside

def fitting_parameters(string, nside , idpatches  ):

    npix = hp.nside2npix(nside) #len(skyvar['dust'][0]['temp'])



    if string=='Bs':
        sky =   get_sky(nside, 's1d0')
        param = ( sky.components[0] .pl_index )
        patchlist =[np.int_( idpatches ), np.zeros( npix, dtype=np.int_ ),  np.zeros( npix, dtype=np.int_ )]

    elif string=='Bd':
        sky =get_sky(nside, 's0d1')
        param = ( sky.components[1] .mbb_index  )
        sky.components[1].mbb_temperature  = 20 * sky.components[1].mbb_temperature.unit
        patchlist  = [np.zeros( npix, dtype=np.int_ ),np.int_( idpatches ),np.zeros( npix, dtype=np.int_ )]

    elif string=='Td':
        sky =get_sky(nside, 's0d1')
        sky.components[1].mbb_index  = 1.6 *sky.components[1].mbb_index.unit 

        param  = ( sky.components[1] .mbb_temperature  ).value
        patchlist =[np.zeros( npix, dtype=np.int_ ),np.zeros( npix, dtype=np.int_ ), np.int_( idpatches ) ]



    return sky, patchlist


def estimate_Stat_and_Sys_residuals( idpatches, galactic_binmask ,
                                    parameter_string ,   randomseed = 1234567,
                                     version='v28' , instrument_conf='LiteBIRD'):

    nside =hp.get_nside(galactic_binmask)
    v={'v27': np.array([39.76, 25.76, 20.69, 12.72, 10.39, 8.95,
                        6.43, 4.3, 4.43, 4.86, 5.44, 9.72, 12.91, 19.07, 43.53]),
        'v28': np.array([59.29, 32.78, 25.76, 15.91, 13.10, 11.25, 7.74, 5.37,
                        5.65, 5.81, 6.48, 15.16, 17.98, 24.99, 49.90]) }

    sens_I_LB =np.array([25.60283688, 13.90070922, 14.32624113,  8.0141844 ,  7.30496454,
         5.95744681,  4.96453901,  4.11347518,  3.33333333,  4.96453901,
         4.11347518,  5.67375887,  6.45390071,  8.08510638, 13.90070922])
    skyconst =get_sky(nside, 'd0s0')


    instrument = get_instrument(instrument_conf)
    instrument.depth_i= sens_I_LB
    instrument.depth_p=v['v28']
    patches = np.zeros_like(galactic_binmask   , dtype=np.int_)

    patches[galactic_binmask] = np.int_(idpatches)+1

    skyvar , patchlist =  fitting_parameters(parameter_string, nside , patches )

    np.random.seed(seed= randomseed )

    signalvar  = get_observation(instrument, skyvar , noise=False   )

    signoisemaps =  get_observation(instrument, skyconst , noise=True   )


    signalvar [:,:,~galactic_binmask]= hp.UNSEEN
    signoisemaps [:,:,~galactic_binmask ]= hp.UNSEEN
    components = [CMB(), Synchrotron(20), Dust(353) ]

    sysresult   = adaptive_comp_sep(components, instrument, signalvar[:,1:]   ,   patchlist )
    statresult   = adaptive_comp_sep(components, instrument, signoisemaps[:,1:] , patchlist )

    msys = np.zeros_like (signalvar[0])
    mstat = np.zeros_like (signoisemaps[0]  )

    #Mask eventually unconstrained pixels
    for i in range(2):
        nan = np.ma.masked_invalid ( sysresult.s[0,i] ) .mask

        msys[i+1,:] = sysresult.s[0,i]
        msys[i+1,nan] = hp.UNSEEN

        nan = np.ma.masked_invalid ( statresult.s[0,i] ) .mask

        mstat[i+1,:] = statresult.s[0,i]
        mstat[i+1,nan] = hp.UNSEEN
    return msys, mstat


def estimate_spectra (msys, mstat ):
    nside= hp.get_nside(msys)

    clstat = hp.anafast(   mstat    ,lmax=3*nside  )
    clsys = hp.anafast(msys   ,lmax=3*nside  )
    ell=np.arange(  clstat.shape[1]   )
    variance_sys=np.sum (clsys[2, :] * (2.*ell+1) )
    variance_stat= np.sum (clstat[2, :] * (2.*ell+1) )

    #print(f"Syst. :{variance_sys}, Stat: { variance_stat}, Tot.: {np.sqrt(variance_sys**2+ variance_stat**2)} " )
    return  clsys , clstat,variance_sys, variance_stat
