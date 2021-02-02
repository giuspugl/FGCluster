import numpy as np

import pylab as pl
pl.rcParams['figure.figsize'] = 12, 16

import healpy as hp


from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import fgbuster.separation_recipes as sr
from fgbuster.visualization import corner_norm
from fgbuster.observation_helpers import get_instrument, get_sky,get_noise_realization
# Imports needed for component separation
from fgbuster import (CMB, Dust, Synchrotron,  # sky-fitting model
                      MixingMatrix)  # separation routine


from fgcluster .utils  import check_nside
import warnings
warnings.filterwarnings("ignore")
import argparse

import os

def main(args) :
    try :
        os.makedirs(args.output_dir )
    except  FileExistsError:
        print (f"Warning: Overwriting files in {args.output_dir}")

    v={}
    #v['v27'] = np.array([39.76, 25.76, 20.69, 12.72, 10.39, 8.95, 6.43, 4.3, 4.43, 4.86, 5.44, 9.72, 12.91, 19.07, 43.53])

    v['v28'] = np.array([59.29, 32.78, 25.76, 15.91, 13.10, 11.25, 7.74, 5.37, 5.65, 5.81, 6.48, 15.16, 17.98, 24.99, 49.90])
    sens_I_LB =np.array([25.60283688, 13.90070922, 14.32624113,  8.0141844 ,  7.30496454,
         5.95744681,  4.96453901,  4.11347518,  3.33333333,  4.96453901,
         4.11347518,  5.67375887,  6.45390071,  8.08510638, 13.90070922])
    nside =args.nside
    Galmask=hp.read_map(args.galmask  ,  verbose=False )
    Galmask =check_nside(nsideout=nside, mapin=Galmask )

    Galmask  =  pl.ma.masked_not_equal(Galmask,0 ).mask

    ones= pl.ma.masked_greater(Galmask,0 ).mask
    fsky = (Galmask[ones].shape[0] / Galmask.shape[0])
    if  args.signalonly  :

        model ='d1s1'
        sky =get_sky(nside, model )

        sky.components[0].mbb_temperature.value[:] = hp.smoothing(
            sky.components[0].mbb_temperature.value, fwhm=np.radians(2)
            )
        sky.components[0].mbb_index.value[:] = hp.smoothing(
            sky.components[0].mbb_index.value, fwhm=np.radians(2)
            )
    else:
        model=  'd0s0'
        sky =get_sky(nside, model  )


    instrument = get_instrument('LiteBIRD')
    instrument.depth_i= sens_I_LB
    instrument.depth_p=v['v28']

    print(f'model: {model} , Signal-only sims: {args.signalonly}, #ID {args.id }')
    print(f'Performing  Component Separation on fsky =   {fsky:.2f}%   ')
    np.random.seed(seed=1234567+args.id  )

    freq_maps = get_observation(instrument, sky, noise= (not args.signalonly )  )
    if not args.signalonly :
        freq_maps += get_noise_realization(nside=nside, instrument = instrument )
    for i, m in enumerate( freq_maps ):
        if args.common_fwhm !=0 :
            freq_maps[i] = hp.smoothing (m, fwhm= np.radians(args.common_fwhm/60. ))

    Bs_patches=hp.read_map(args.Bs_clusters, dtype=pl.int_, verbose=False)
    Bd_patches=hp.read_map(args.Bd_clusters, dtype=pl.int_, verbose=False)
    Td_patches=hp.read_map(args.Td_clusters, dtype=pl.int_, verbose=False)
    Bs_patches=check_nside(nsideout=nside, mapin=Bs_patches )
    Bd_patches=check_nside(nsideout=nside, mapin=Bd_patches )
    Td_patches=check_nside(nsideout=nside, mapin=Td_patches )
    freq_maps[:,:,~Galmask] =hp.UNSEEN

    components = components = [CMB(),  Synchrotron(20 , nu_pivot=70., running=0.0), Dust(353) ]
    options={'disp':False, 'gtol': 1e-18, 'eps': 1e-18, 'maxiter': 10000, 'ftol': 1e-18 }
    tol = 1e-18
    method='TNC'
    if args.polarization and not args.include_temperature:
        results    = sr.adaptive_comp_sep(components, instrument, freq_maps[:,1:]  ,
                                   [Bs_patches, Bd_patches, Td_patches] , method=method,tol = tol, options=options  )
        string ='_pol_'
    elif args.polarization and  args.include_temperature:
        results  = sr.adaptive_comp_sep(components, instrument, freq_maps   ,
                                   [Bs_patches, Bd_patches, Td_patches],  method=method,tol = tol, options=options  )
        string ='_pol_temp_'

    elif  not args.polarization and  args.include_temperature:
        results   = sr.adaptive_comp_sep(components, instrument, freq_maps[:,0]   ,
                                   [Bs_patches, Bd_patches, Td_patches] ,  method=method,tol = tol, options=options  )
        string ='_temp_'

    elif not args.polarization and not args.include_temperature:
        raise ValueError("Unset arguments set at least one between  --polarization or --include-temperature")

    np.savez(f'{args.output_dir}/res__{model}__{args.output_string}__BdTdBs__{args.id}__{string}__nside{nside}.npz',
                    **{n: a for n, a in results.items()})


if __name__=="__main__":
        parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
        parser.add_argument("--Bs-clusters" ,    help='path of Bs cluster patches', required=True)
        parser.add_argument("--Bd-clusters" ,   help='path of Bd cluster patches', required=True)
        parser.add_argument("--Td-clusters" ,    help='path of Td cluster patches', required=True)
        parser.add_argument("--output-dir" ,    help='path for outputs', default='./')
        parser.add_argument("--output-string" ,    help='label for outputs', default='')
        parser.add_argument("--polarization", help='compsep on polarization data ',
                                        action='store_true')

        parser.add_argument("--include-temperature",
                            help='add temperature data to compsep', action='store_true')
        parser.add_argument("--signalonly",
                            help='run on signal only data ', action='store_true')
        parser.add_argument('--common-fwhm',
                                help='convolve frequency maps to a common resolution', default=0 , type=np.float )
        parser.add_argument("--nside", help="nside of output maps" ,required=True ,  type=np.int_)
        parser.add_argument("--id", help="" ,required=True ,  type=np.int_)
        parser.add_argument("--galmask", help = 'path to the  healpix galactic mask ', )
        args = parser.parse_args()


        main( args)

"""
    mpirun -np 4 python  fg_w_clustering.py  \
            --Bs-clusters "/Users/peppe/work/adaptive_compsep/clusterpatches/clusters_hellinger_galmask_Bs_32_noise_partition.fits"  \
            --Bd-clusters "/Users/peppe/work/adaptive_compsep/clusterpatches/Bd_250_noise.fits"  \
            --Td-clusters "/Users/peppe/work/adaptive_compsep/clusterpatches/Td_250_noise.fits"\
            --polarization \
            --nside  32 \
            --galmask  "/Users/peppe/work/heavy_maps/HFI_Mask_GalPlane-apo2_32_R2.00.fits" \
            --output-dir results_test_noise

"""