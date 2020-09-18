
import healpy as hp
import pylab as pl
import numpy as np
import time
from mpi4py import MPI

from spectral_clustering_mpi import ( build_adjacency_from_heat_kernel,
                                    build_adjacency_from_nearest_neighbours,
                                    estimate_Laplacian_matrix,
                                    estimate_Ritz_eigenpairs,
                                    build_distance_matrix_from_eigenvectors, KS_distance)


import os

def test_spectral_clustering( ):
    workdir =os. getcwd()

    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    nside=2
    randseed = 1234567
    np.random.seed(randseed)

    X= np.random.uniform(size= hp.nside2npix(nside ))
    sigmaX=np.ones_like(X)*.2
    if rank ==0 :
        assert  (KS_distance((0.3,1.3),(0,1)) ==0.01  )

    s =time.perf_counter ()

    Ann   = build_adjacency_from_nearest_neighbours(  nside=nside , neighbour_order=0,
                                comm=comm,  )
    e = time.perf_counter ()

    if rank==0:
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_NN.npy'), Ann, atol=1e-5)
        print(f"build_adjacency_from_nearest_neighbours, execution time {e-s }sec")

    s =time.perf_counter ()

    Ann_w  = build_adjacency_from_nearest_neighbours(  nside=nside , neighbour_order=0,
                                    comm=comm,
                                    KS_weighted=True, X=X, sigmaX=X  )


    e = time.perf_counter ()
    if rank==0 :
        assert np.allclose(pl.load(f'{workdir}/test/adjacency_wNN.npy'), Ann_w, atol=1e-5)
        print(f"build_adjacency_from_nearest_neighbours (weighted), execution time {e-s }sec")
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
        assert np.allclose(np.load(f'{workdir}/test/eigenpairs.npz')['evals'], l )
        assert np.allclose(np.load(f'{workdir}/test/eigenpairs.npz')['evecs'], W )
        print(f"estimate_Ritz_eigenpairs, execution time {e-s }sec")

    #we don't consider the smallest eigenvectors since it's the constant vector
    s=time.perf_counter()
    E = build_distance_matrix_from_eigenvectors(W[:,1:] ,comm=comm )
    e = time.perf_counter ()

    if rank==0 :
        assert np.allclose(np.load(f'{workdir}/test/euclidean_evecs.npz' )['distance']  ,E  )
        #pl.imshow(E);pl.colorbar();pl.show()
        print(f"build_distance_matrix_from_eigenvectors, execution time {e-s }sec")

    comm.Disconnect


test_spectral_clustering ()
