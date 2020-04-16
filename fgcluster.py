from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster.component_model import CMB, Dust, Synchrotron

from fgbuster.separation_recipes import basic_comp_sep,adaptive_comp_sep

import healpy as hp 
import numpy as np 
import pysm


def check_nside (nsideout , mapin): 
    nside2 =hp.get_nside(mapin)
    if nside2 != nsideout : 
        print ("running ud_grade ")
        return hp.ud_grade(nside_out=nsideout , map_in=mapin) 
    else: 
        return mapin 
    
def fitting_parameters(string, nside , idpatches  ): 
    
    npix = hp.nside2npix(nside) #len(skyvar['dust'][0]['temp'])
    if string=='Bd' : 
        skyvar=get_sky(nside, 's0d1')

        skyvar['dust'][0]['temp']=20 #*np.ones(npix)
        #skyvar['synchrotron'] [0]['spectral_index']=-3 *np.ones(npix)
        #patchlist[0] = 
        patchlist =[np.zeros( npix, dtype=np.int_ ), np.int_( idpatches ), np.zeros( npix, dtype=np.int_ )]  
    elif string=='Td': 
        skyvar=get_sky(nside, 's0d1')
        
        skyvar['dust'][0]['spectral_index']=1.6 *np.ones(npix)
        #skyvar['synchrotron'] [0]['spectral_index']=-3 *np.ones(npix)
        #patchlist[1] =  np.int_( idpatches )
        patchlist =[np.zeros( npix, dtype=np.int_ ),np.zeros( npix, dtype=np.int_ ), np.int_( idpatches ), ]  
        
    elif string=='Bs': 
        skyvar=get_sky(nside, 's1d0')
        
        #skyvar['dust'][0]['temp']=20 *np.ones(npix)
        #skyvar['dust'][0]['spectral_index']=1.6 *np.ones(npix)
        patchlist  = [np.int_( idpatches ),np.zeros( npix, dtype=np.int_ ),np.zeros( npix, dtype=np.int_ )]
        
        
    return skyvar, patchlist 
                
        
def estimate_Stat_and_Sys_residuals( idpatches, parameter_string , instrument_conf,
                                    galactic_binmask ,   randomseed = 1234567 ): 
    
    nside =hp.get_nside(idpatches) 
 
    sky_conf, patchlist =  fitting_parameters(parameter_string, nside , idpatches )
    
    skyvar  = pysm.Sky(sky_conf)
    
    
    instrument_conf ['add_noise'] =True
    instrument_conf['noise_seed'] = randomseed 
    instrument = pysm.Instrument(instrument_conf)
    
    signalvar = instrument.observe(skyvar, write_outputs=False, ) [0] 
    nside2=nside 
    
    skyconst = pysm.Sky (get_sky(nside2, 's0d0'))
    signal,noise = instrument.observe(skyconst , write_outputs=False, ) 

    signoisemaps  = signal  + noise 
    
    
    signalvar [:,:,~galactic_binmask]= hp.UNSEEN
    signoisemaps [:,:,~galactic_binmask ]= hp.UNSEEN
    components = [CMB(), Synchrotron(20), Dust(353) ]
    
    cov =np.ones_like(signalvar)
    for i in range(np.array(instrument.Frequencies) .shape[0]): 
        cov[i,0,:]*=(instrument.Sens_I[i] )**2/hp.nside2resol(nside, arcmin=True)
        cov[i,1,:]*=(instrument.Sens_P[i] )**2/hp.nside2resol(nside, arcmin=True)
    cov[:,2,:]= cov[:,1,:] 
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

    
    
    
    