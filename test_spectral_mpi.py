
import healpy as hp
import pylab as pl
import numpy as np
import time
from mpi4py import MPI

from fgcluster.spectral_clustering_mpi import ( build_adjacency_from_heat_kernel,
                                    build_adjacency_from_nearest_neighbours,
                                    build_adjacency_from_KS_distance,
                                    estimate_Laplacian_matrix,
                                    estimate_Ritz_eigenpairs,from_ell_to_index,from_index_to_ell,
                                    build_distance_matrix_from_eigenvectors, kolmogorov_smirnov_distance)


import os

def test_utils () :
    l =200
    l1,l2= from_ell_to_index(l )

    ell = (from_index_to_ell(.5*(l1+l2)))
    assert l ==ell
def test_spectral_clustering( comm,workdir ):
    rank =comm.Get_rank()
    nside=2
    s =time.perf_counter ()

    Ann   = build_adjacency_from_nearest_neighbours(  nside=nside , neighbour_order=0,
                                comm=comm,  )
    e = time.perf_counter ()

    if rank==0:
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_NN.npy'), Ann, atol=1e-5)
        print(f"build_adjacency_from_nearest_neighbours, execution time {e-s }sec")


    s =time.perf_counter ()
    A = build_adjacency_from_heat_kernel (nside, comm  )
    e = time.perf_counter ()
    if rank==0 :
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_heat.npy'), A, atol=1e-5)
        print(f"build_adjacency_from_heat_kernel, execution time {e-s }sec")


    L = estimate_Laplacian_matrix(A ,kind='normalized')
    if rank==0:
        assert np.allclose(pl.load(f'{workdir}/test/laplacian_normalized.npz',allow_pickle=True )['laplacian'] , L.todense() )
    L = estimate_Laplacian_matrix(A ,kind='symmetric')
    if rank==0:
        assert np.allclose(pl.load(f'{workdir}/test/laplacian_symmetric.npz',allow_pickle=True )['laplacian'] , L.todense() )
    L = estimate_Laplacian_matrix(A ,kind='unnormalized')

    if rank==0:
        assert np.allclose(pl.load(f'{workdir}/test/laplacian_unnormalized.npz',allow_pickle=True )['laplacian'] , L.todense() )
    s =time.perf_counter ()
    l, W = estimate_Ritz_eigenpairs (L, n_eig = 6 )
    e = time.perf_counter ()

    if rank==0 :
        print(f"estimate_Ritz_eigenpairs, execution time {e-s }sec")

    #we don't consider the smallest eigenvectors since it's the constant vector
    s=time.perf_counter()
    E = build_distance_matrix_from_eigenvectors(W[:,1:] ,comm=comm )
    e = time.perf_counter ()

    if rank==0 :
        #np.savez(f'{workdir}/test/euclidean_evecs.npz', distance= E ) 
        assert np.allclose(np.load(f'{workdir}/test/euclidean_evecs.npz' )['distance']  ,E  )
        #pl.imshow(E);pl.colorbar();pl.show()
        print(f"build_distance_matrix_from_eigenvectors, execution time {e-s }sec")

def test_KS_distances(comm,workdir ):
    rank =comm.Get_rank()

    nside=2
    randseed = 1234567
    np.random.seed(randseed)
    X= np.random.uniform(size= hp.nside2npix(nside ))
    sigmaX=np.ones_like(X)*.2
    if rank ==0 :
        assert  (kolmogorov_smirnov_distance((0.3,1.3),(0,1), ntests=50,nsamp=100) ==0.04514318340310804  )


    s =time.perf_counter ()
    Ann_w  = build_adjacency_from_nearest_neighbours(  nside=nside , neighbour_order=0,
                                        comm=comm,
                                        KS_weighted=True, X=X, sigmaX=X  )
    e = time.perf_counter ()
    if rank==0 :
        #pl.save(f'{workdir}/test/adjacency_wNN.npy',Ann_w)
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_wNN.npy'), Ann_w, atol=1e-5)
        print(f"build_adjacency_from_nearest_neighbours (weighted), execution time {e-s }sec")

    s =time.perf_counter ()
    Q   = build_adjacency_from_KS_distance(  nside=nside , comm=comm,
                                    X= X, sigmaX=sigmaX  )
    e = time.perf_counter ()
    if rank==0:
        #pl.save(f'{workdir}/test/adjacency_KS.npy', Q)
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_KS.npy'), Q, atol=1e-5)
        print(f"build_adjacency_from_KS_distance, execution time {e-s }sec")

    s =time.perf_counter ()

    A   = build_adjacency_from_heat_kernel(  nside=nside , comm=comm, KS_weighted=True ,
                                                    Q=Q  ,alpha=.5)
    e = time.perf_counter ()

    if rank==0:
        #pl.save(f'{workdir}/test/adjacency_weighted_heat.npy',A)
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_weighted_heat.npy'), A, atol=1e-5)
        print(f"build_adjacency_from_heat_kernel (weighted), execution time {e-s }sec")

comm    = MPI.COMM_WORLD
workdir =os. getcwd()
test_spectral_clustering (comm, workdir  )
test_KS_distances(comm , workdir )
test_utils()
comm.Disconnect
