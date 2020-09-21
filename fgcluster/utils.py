import warnings
import pylab as pl
import healpy as hp
import astropy.units as u

import numpy as np



def smooth_and_rotate_map (input_map ,lmax=None , fwhm=None , rot= None  ):
    nside= hp.get_nside(input_map)
    alm = hp.map2alm(
            input_map, lmax=lmax, use_pixel_weights=True if nside > 16 else False)
    if fwhm is not None:
        hp.smoothalm(
               alm, fwhm=fwhm.to_value(u.rad), verbose=False, inplace=True, pol=True
            )
    if rot is not None:
        alm = rot.rotate_alm(alm )
    smoothed_map = hp.alm2map(alm, nside=nside, verbose=False, pixwin=False)
    if hasattr(input_map, "unit"):
        smoothed_map <<= input_map.unit
    return smoothed_map

def plotclusters(labels,imap):
    outm= pl.zeros_like(imap)
    for i in range(labels.max()+1):
        pixs=pl.where(labels==i)
        outm[pixs]=imap[pixs].mean()

    return outm

def check_nside (nsideout , mapin):
    nside2 =hp.get_nside(mapin)
    if nside2 != nsideout :
        print ("running ud_grade ")
        return hp.ud_grade(nside_out=nsideout , map_in=mapin)
    else:
        return mapin

def hellinger_distance(x,y ) :
    #estimating hellinger distance from https://en.wikipedia.org/wiki/Hellinger_distance
    mu1= x[0]; sigma1=x[1];
    mu2= y[0]; sigma2=y[1] ;


    BC = (pl.sqrt(2. *sigma1*sigma2/(sigma1**2 +sigma2**2))
          * pl.exp(-1/4. *(mu1-mu2)**2/(sigma1**2 +sigma2**2)))

    return pl.sqrt(1- BC )
