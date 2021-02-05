import healpy as hp
import pylab as pl
import numpy as np
import time
from mpi4py import MPI
import argparse
import os
from os import path
import fgcluster as fgc
from fgcluster.spectral_clustering_mpi import ( build_adjacency_from_heat_kernel, build_adjacency_from_heat_kernel_gather ,
                                    build_adjacency_from_nearest_neighbours,
                                    build_adjacency_from_KS_distance,build_adjacency_from_KS_distance_gather ,
                                    build_adjacency_from_KS_distance_savedata ,
                                    build_adjacency_from_heat_kernel_savedata ,
                                    estimate_Laplacian_matrix,
                                    estimate_Ritz_eigenpairs,
                                    from_index_to_ell ,
                                    from_ell_to_index,
                                    build_distance_matrix_from_eigenvectors, kolmogorov_smirnov_distance)

from fgbuster import   get_sky
from fgcluster import (
     utils as cu  )
from sklearn.cluster import AgglomerativeClustering


def main(args) :
    comm    = MPI.COMM_WORLD
    workdir =os. getcwd()
    rank =comm.Get_rank()
    nprocs =comm.Get_size()
    nside=args.nside

    string =  args.spectral_parameter
    sky =get_sky(nside, 's1d1')

    if string=='Bs':

        param = ( sky.components[0] .pl_index )
    elif string=='Bd':
        param = ( sky.components[1] .mbb_index  )
    elif string=='Td':
        param  = ( sky.components[1] .mbb_temperature  ).value

    sigmaparam = hp.read_map(args.parameter_uncertainties , verbose=False )
    sigmaparam  = cu.check_nside(nsideout=nside, mapin=sigmaparam  )
    mask =np.ma.masked_less(sigmaparam,1e-7 ).mask
    sigmaparam  [mask] =param [ mask]*.0005
    if rank==0:
        hp.mollview(sigmaparam, norm='hist', sub=122)
        hp.mollview(param, norm='hist', sub=121);pl.show()

    if np.bool(args.KS_weight)  :
        if path.exists(f'{workdir}/affinities/KS_distance_{string}_{args.nside}.npz'):
            Q= np.load(f'{workdir}/affinities/KS_distance_{string}_{args.nside}.npz')['affinity']
        else :
            Q   = build_adjacency_from_KS_distance(  nside=nside , comm=comm,
                                X= param, sigmaX=sigmaparam , ntests =5, nresample=100   )
            if rank ==0 :
                np.savez(f'{workdir}/affinities/KS_distance_{string}_{args.nside}.npz',
                        affinity  =Q  )

    else:
        Q=None

    A = build_adjacency_from_heat_kernel (nside, comm,  KS_weighted= np.bool(args.KS_weight) ,
                                                    Q=Q  ,alpha=args.KS_weight )
    if rank ==0:
        pl.subplot(121)
        pl.title('Heat Kernel matrix  ')
        pl.imshow(np.log(A));
        pl.subplot(122)
        pl.title('KS distance matrix ')
        pl.imshow(np.log(Q)) ;
        pl.show()
    L = estimate_Laplacian_matrix(A ,kind='unnormalized')
    lmax= nside -1
    Nmax= np.int_(from_ell_to_index(lmax )[1])
    if rank==0 : print(f"Estimating eigenvalues up to lmax= {nside  -1 }, i.e. the first {Nmax} eigenvectors of the Laplacian ")
    l, W = estimate_Ritz_eigenpairs (L, n_eig = Nmax  )

    E = build_distance_matrix_from_eigenvectors(W[:,1:] ,comm=comm )
    if rank ==0 :
        np.savez(f'{workdir}/affinities/{string}_euclidean_distance_eigenvectors_{args.KS_weight:.2f}_{args.nside}.npz',
                    distance =E ,eigenvectors=W [:,1:] , eigenvalues=l [ 1:]   )

    clusters  = AgglomerativeClustering(distance_threshold=args.distance_threshold,affinity= 'precomputed',
                                   linkage='average', compute_full_tree=True, n_clusters=None   ).fit(E  )

    patches = pl.zeros_like( param  , dtype=pl.int_)

    patches = pl.int_( clusters.labels_)
    if rank ==0 :
        hp.mollview( patches,cmap=pl.cm.tab20 ); pl.show()
        fmap= f'{workdir}/clusterpatches/{string}_clusters_spectralclus_{args.KS_weight:.2f}_{args.nside}.fits'
        hp.write_map(fmap,  patches, overwrite=True )

    comm.Disconnect 




    pass





if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )

	parser.add_argument("--spectral-parameter" , help='spectral  parameter (Bd, Td, Bs) to perform clustering' )
	parser.add_argument("--parameter-uncertainties", help = 'path to the  healpix  map of uncertainties',   )

	parser.add_argument("--nside", help="nside of output maps" ,required=True ,  type=np.int_)
	parser.add_argument("--KS-weight", help="weight to the KS distance to be combined with heat kernel" ,
                            required=True ,  type=np.float)
	parser.add_argument("--distance-threshold", help = 'threshold to define connectivity in AgglomerativeClustering',
                                    default=0.3,  type=np.float)

	args = parser.parse_args()


	main( args)
