

import healpy as hp
import numpy as np
import argparse
import pylab as pl

import pysm
import pysm.units as u

from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import fgbuster.separation_recipes as sr
from fgbuster.visualization import corner_norm
from fgbuster.observation_helpers import get_instrument, get_sky
# Imports needed for component separation
from fgbuster import (CMB, Dust, Synchrotron,  # sky-fitting model
                      MixingMatrix)  # separation routine

from fgcluster import (
     residuals as res ,
     utils as cu  )

from fgcluster.clusters  import ClusterData
from fgcluster.residuals import estimate_Stat_and_Sys_residuals, estimate_spectra

import warnings
warnings.filterwarnings("ignore")

def main(args):

    angles=args.haversine_distance

    affinity =args.affinity


    string =  args.spectral_parameter
    nside =args.nside
    sky =get_sky(nside, 's1d1')
    if string=='Bs':

        param = ( sky.components[0] .pl_index )
    elif string=='Bd':
        param = ( sky.components[1] .mbb_index  )
    elif string=='Td':
        param  = ( sky.components[1] .mbb_temperature  ).value


    sigmaparam = hp.read_map(args.parameter_uncertainties , verbose=False )
    Galmask=hp.read_map(args.galmask  ,  verbose=False )
    Galmask =cu.check_nside(nsideout=nside, mapin=Galmask )
    sigmaparam =  cu.check_nside(nsideout=nside, mapin=sigmaparam )

    Galmask  =  pl.ma.masked_not_equal(Galmask,0 ).mask

    ones= pl.ma.masked_greater(Galmask,0 ).mask
    fsky = (Galmask[ones].shape[0] / Galmask.shape[0])
    if args.verbose  : print(f' Clustering on fsky =   {fsky:.2f}%   ')

    #param[~Galmask]=0
    #sigmaparam [~Galmask ] =0
    noisestring=''
    if args.add_noise :
        noise_param = np.random.normal(loc=pl.zeros_like(param) , scale =sigmaparam/10.  )
        param += noise_param
        param = hp.smoothing(param, fwhm=pl.radians(args.parameter_resolution), verbose=False  )

        noisestring='_noise'

    save_affinity = True

    anglestring=''
    if angles :
        anglestring='_haversine'

    fmap= ('/Users/peppe/work/adaptive_compsep/clusterpatches/'+
                                f'clusters_{affinity}{anglestring}_galmask_{string}_{nside}{noisestring}_{args.optimization}.fits')
    file_affinity= (f'/Users/peppe/work/adaptive_compsep/affinities/'+
                            f'{affinity}{anglestring}_galmask_{string}_{nside}{noisestring}.npy')

    Cluster   = ClusterData( [param, sigmaparam ]  ,  nfeatures=2,
                              nside=nside,  affinity=affinity  ,file_affinity =file_affinity ,
                              include_haversine=angles, verbose=args.verbose  ,save_affinity=save_affinity ,
                               scaler=None,
                           feature_weights=[1, 1] , galactic_mask = Galmask  )

    Kmin=2
    Kmax=200
    Cluster(nvals= args.num_cluster_evaluation , Kmax=Kmax-1  ,Kmin=Kmin,
                minimize= args.optimization ,parameter_string =string  )

    if args.optimization == 'partition':
        label1 =r'Under partition  '
        label2 =r'Over partition '
        ylabel ='Partition measure '

    elif args.optimization == 'residuals':
        label1 =r'Syst. residuals'
        label2 =r'Stat. residuals'
        ylabel =r'Residuals [ $\mu K^2$ ]  '

    pl.title(string)
    pl.plot(Cluster.Kvals,
            Cluster.Vu ,'.',label=label1  )
    pl.plot(Cluster.Kvals, Cluster.Vo ,'.',label=label2 )
    pl.plot(Cluster.Kvals, pl.sqrt(Cluster.Vo**2 +Cluster.Vu**2) ,'-', label=r'Root Squared Sum ')
    pl.legend()
    pl.xlabel('K',fontsize=15)

    pl.ylabel(ylabel ,fontsize=15)
    pl.show()

    patches = pl.zeros_like( param  , dtype=pl.int_)

    patches[Galmask] = pl.int_(Cluster.clusters.labels_)+1

    hp.mollview( patches,cmap=pl.cm.tab20 ); pl.show()

    hp.write_map(fmap,  patches, overwrite=True )


if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--spectral-parameter" , help='spectral  parameter (Bd, Td, Bs) to perform clustering' )
	parser.add_argument("--parameter-uncertainties", help = 'path to the  healpix  map of uncertainties',   )
	parser.add_argument("--parameter-resolution", help = 'resolution in deg  to convolve the parameter map ', default= 4. )
	parser.add_argument("--affinity", help="identifier of Affinity distance ('hellinger, haversine, euclidean')"
                                    , default='hellinger')
	parser.add_argument("--haversine-distance", help='include spatial similarity ',
                                action='store_true')
	parser.add_argument("--add-noise",
                            help='add noise to parameter map', action='store_true')
	parser.add_argument("--no-verbose",help='', action='store_true' )
	parser.add_argument("--verbose",help='', action='store_true', default=True  )

	parser.add_argument("--optimization" , default='partition',
                        help=" Metric to estimate optimal number of clusters, (partition, residuals, KL, etc... )")
	parser.add_argument("--nside", help="nside of output maps" , default=32,  type=np.int_)
	parser.add_argument("--galmask", help = 'path to the  healpix galactic mask ', )
	parser.add_argument("--num-cluster-evaluation", default=10,  type=np.int_)
	args = parser.parse_args()
	if args.no_verbose : args.verbose=False
	main( args)


"""
    python  estimate_clusters.py   \
    --spectral-parameter  "Td"  \
    --parameter-uncertainties  "sigma_Td_dave_32.fits" \
    --nside 32  \
    --galmask   "/Users/peppe/work/heavy_maps/HFI_Mask_GalPlane-apo2_32_R2.00.fits" \
    --add-noise
"""
