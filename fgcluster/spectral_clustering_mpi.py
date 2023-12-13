#
#   spectral_clustering_mpi.py
#   module encoding the spectral clustering.
#
#   date: 2021-09-21
#   author: GIUSEPPE PUGLISI
#
#   Copyright (C) 2021  Giuseppe Puglisi
#


import healpy as hp
import pylab as pl
import numpy as np

from scipy.stats import norm, ks_2samp
from scipy import sparse
from scipy.special import legendre
from sklearn.metrics import pairwise_distances
from .utils import (
    get_lmax,
    get_neighbours,
    split_data_among_processors,
    minmaxrescale,
    from_ell_to_index,
    from_index_to_ell,
)
from mpi4py import MPI

import warnings

warnings.filterwarnings("ignore")


def kolmogorov_smirnov_distance(x, y, ntests, nsamp):
    """
    Estimate for a pair `x` and `y` the Kolmogorov Smirnov Test, for `ntests` times,
    by resampling the measurements :math:`x\pm dx ` and :math:`y\pm dy ` with a
    normal distribution `nsamp` times.
    """
    mu1 = x[0]
    sigma1 = x[1]
    mu2 = y[0]
    sigma2 = y[1]
    D = []
    # We  repeat the KS test ntest times (it takes time) on two samples with same size n=m
    # np.random.seed(1234)

    for test in range(ntests):
        rvs1 = norm.rvs(size=nsamp, loc=mu1, scale=sigma1)
        rvs2 = norm.rvs(size=nsamp, loc=mu2, scale=sigma2)
        _, pval = ks_2samp(rvs1, rvs2)
        D.append(pval)
    med = np.median(D)
    return med


def heat_kernel(Theta, l, sigma):
    """
    Returns the functional Heat kernel given the cosine matrix `Theta` at a given `l`
    and `sigma`.
    """
    return (2 * l + 1) / 4.0 / np.pi * np.exp(-sigma * l * (l + 1)) * legendre(l)(Theta)


def build_adjacency_from_heat_kernel_gather(
    nside, comm, stopping_threshold=1e-7, KS_weighted=False, Q=None, alpha=0.5
):
    """
    Estimate the Adjacency matrix in parallel from the Heat kernel.

    **Parameters**

    - `nside`:{int}
    resolution parameter of the healpix map
    - `comm` : {MPI_COMM}
    the MPI communicator
    - `stopping_threshold`:{float}
    the threshold to stop the sum over `l` of the heat kernel
    - `KS_weighted`:{bool}
    boolean to distort the adjacency with the KS distance
    - `Q`:{np.matrix}
    matrix encoding the KS distance
    - `alpha`:{float}
    the weight to set the amplitude of the KS distortion
    """
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    npix = hp.nside2npix(nside)
    p = np.arange(npix)
    V = np.array(hp.pix2vec(ipix=p, nside=hp.get_nside(p))).T
    scalprod = minmaxrescale(V.dot(V.T), a=-1, b=1)

    if KS_weighted:
        Theta = np.arccos(scalprod)
        Psi = Theta + alpha * (1 - Q) * np.pi / 2
        scalprod = np.cos(Psi)

    lmax, sigmabeam = get_lmax(nside, stopping_threshold)
    start_row, end_row = split_data_among_processors(
        size=npix, rank=rank, nprocs=nprocs
    )
    # Create loc array on each core
    Gloc = np.zeros((end_row - start_row, npix), dtype=np.float_)
    offset_hpx = start_row
    for l in np.arange(lmax):
        Gloc += heat_kernel(
            scalprod[offset_hpx : offset_hpx + Gloc.shape[0], :], l, sigmabeam
        )
    comm.Barrier()
    G= np.zeros(npix*npix)
    comm.Allgather(Gloc,G)
    G = (G).reshape((npix, npix))
    return G


def build_adjacency_from_KS_distance_gather(
    nside, comm, X, sigmaX, ntests=50, nresample=100
):
    """
    Build the adjacency matrix  given the Kolmogorov Smirnov (KS) distance.
    /!\ This is one of the most computationally expensive routine in the whole package.

    - `nside`:{int}
    resolution parameter of the healpix map
    - `comm` : {MPI_COMM}
    the MPI communicator
    - `X`:{np.array}
    Healpix map  encoding the features to cluster
    - `sigmaX`:
    {np.array}
    Healpix map  encoding the uncertainties of the  features to cluster
    - `ntests`:{int}
    number of KS tests   for a given pairwise evaluation  (X1,sigmaX1) and (X2, sigmaX2)
    - `nresample`:{int}
    size of the bootstrap resampling of a measurement  (X1,sigmaX1)

    """
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    # npix = hp.nside2npix(nside)
    npix = X.size
    start_row, end_row = split_data_among_processors(
        size=npix, rank=rank, nprocs=nprocs
    )
    Qloc = np.zeros(
        (end_row - start_row, npix), dtype=np.float_
    )  # Create Qloc array on each core
    offset_hpx = start_row
    for i in range(Qloc.shape[0]):
        X_i = (X[i + offset_hpx], sigmaX[i + offset_hpx])
        for j in range(npix):
            X_j = (X[j], sigmaX[j])
            q = kolmogorov_smirnov_distance(
                x=X_i, y=X_j, ntests=ntests, nsamp=nresample
            )
            Qloc[i, j] = q

    comm.Barrier()
    Q=np.zeros(npix*npix)
    comm.Allgather(Qloc, Q)
    Q = (Q).reshape((npix, npix))
    Q[np.diag_indices(npix)] = 0.0
    return minmaxrescale(Q, a=0, b=1)


def build_adjacency_from_nearest_neighbours(nside, comm, neighbour_order=1):

    """
    Estimate the Adjacency matrix in parallel from the Nearest neighbour pixels.

    **Parameters**

    - `nside`:{int}
    resolution parameter of the healpix map
    - `comm` : {MPI_COMM}
    the MPI communicator
    - `neighbour_order`:{int}
    set the degree of pixel proximity, e.g. `order=0` returns the 8 closest pixels

    """

    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    npix = hp.nside2npix(nside)
    Dloc = np.zeros((npix, npix))
    # Identify for each pixels what are the nearest neighbours,
    # the order specifies how many nearest neighbours to include:
    # the higher the order, the more are the neighbours included
    start, stop = split_data_among_processors(size=npix, rank=rank, nprocs=nprocs)

    for i in np.arange(start, stop):
        neighbours = get_neighbours(ipix=i, nside=nside, order=neighbour_order)
        Dloc[i, neighbours] = 1
    D = np.zeros_like(Dloc)
    comm.Allreduce(Dloc, D, op=MPI.SUM)

    return D


def estimate_Laplacian_matrix(W, kind="unnormalized"):
    """
    Estimate the Laplacian matrix from the matrix of weights`W`.
    3 implementation of the Laplacian are available :
    - unnormalized
    - normalized
    - symmetric


    """

    ### Matrices that we are dealing with are sparse, it's more efficient to defined them as
    ### Compressed Sparse Row  (CSR ) matrices with scipy .

    Ws = sparse.csr_matrix(W)
    D = W.dot(np.ones_like(W[0, :]))

    Dinv = 1.0 / (D)
    Dinv2 = 1.0 / pl.sqrt(D)
    Ds = sparse.spdiags(D, diags=0, m=W.shape[0], n=W.shape[1])
    L = sparse.csr_matrix(Ds - Ws)

    if kind == "unnormalized":
        return L
    elif kind == "normalized":
        Dinvs = sparse.spdiags(Dinv, diags=0, m=W.shape[0], n=W.shape[1])
        Lnorm = Dinvs.dot(L)
        return Lnorm
    elif kind == "symmetric":
        Dinv2s = sparse.spdiags(Dinv2, diags=0, m=W.shape[0], n=W.shape[1])
        Lsym = Dinv2s.dot(L.dot(Dinv2s))
        return Lsym
    else:
        raise SyntaxError(
            ' "kind" can   be one among :  "unnormalized" ,  "normalized" , "symmetric"  '
        )


def estimate_Ritz_eigenpairs(L, n_eig, tolerance=1e-12):
    """
    Estimate the first `n_eig` eigenpairs via the ARPACK library of the
    Laplacian matrix `L`. `tolerance` set the precision of the approximated
    eigenvalues.
    """

    eval, evec = sparse.linalg.eigsh(
        L, k=n_eig, return_eigenvectors=True, which="SM", tol=tolerance
    )
    return eval, evec


def build_distance_matrix_from_eigenvectors(W, comm):
    """
    Estimate the Euclidean distance from the matrix built from the
    Laplacian eigenvectors  `W`. We pass the MPI_COMM `comm` to perform the estimate
    in parallel.
    """
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    npix = W.shape[0]
    Indices = np.array(np.triu_indices(npix, 1))
    # assigning a symmetric matrix values in mpi
    # globalsize = Indices.shape[1]
    # start, stop = split_data_among_processors(size=globalsize, rank=rank, nprocs=nprocs)
    start, stop = split_data_among_processors(size=npix, rank=rank, nprocs=nprocs)
    Dloc = np.zeros((npix, npix))
    for i in range(start, stop):
        i_indx = np.ma.masked_equal(Indices[0], i).mask
        j_indx = Indices[1][i_indx]
        # the euclidean distance is estimated considering npix samples of m features
        # (each row of W  is an  m-dimensional (m being the number  of columns of W)
        try:
            Dloc[i, j_indx] = np.linalg.norm(W[i, :] - W[j_indx, :], axis=1)
            Dloc[j_indx, i] = Dloc[i, j_indx]
        except ValueError:
            # this exception is to avoid the ValueError raising
            # when there ain't no j-elements in correspondence of the
            # last row
            Dloc[i, i] = 0.0
    Dout = np.zeros_like(Dloc )
    comm.Allreduce(Dloc,Dout , op=MPI.SUM)
    return Dout


def build_adjacency_from_KS_distance_nearest_neighbours(
    nside, comm, X, sigmaX, order_nn, ntests=50, nresample=100
):

    """
    Build the adjacency matrix  given the Kolmogorov Smirnov (KS) distance.
    With respect to `build_adjacency_from_KS_distance`, we build it by restricting the
    evaluation onto only the nearest neighbours of a given pixel.

    - `nside`:{int}
    resolution parameter of the healpix map
    - `comm` : {MPI_COMM}
    the MPI communicator
    - `X`:{np.array}
    Healpix map  encoding the features to cluster
    - `sigmaX`:
    {np.array}
    Healpix map  encoding the uncertainties of the  features to cluster
    - `order_nn`:{int}
    set the degree of pixel proximity, e.g. `order=0` returns the 8 closest pixels
    - `ntests`:{int}
    number of KS tests   for a given pairwise evaluation  (X1,sigmaX1) and (X2, sigmaX2)
    - `nresample`:{int}
    size of the bootstrap resampling of a measurement  (X1,sigmaX1)

    """
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    npix = hp.nside2npix(nside)
    Indices = np.triu_indices(npix, 1)
    Qloc = np.zeros(npix * npix)
    start, stop = split_data_among_processors(
        size=Indices[0].size, rank=rank, nprocs=nprocs
    )

    for i in range(start, stop):
        i_indx = np.ma.masked_equal(Indices[0], i).mask
        j_indx = Indices[1][i_indx]
        listpix = get_neighbours(ipix=i, nside=nside, order=order_nn)
        intersect = np.intersect1d(j_indx, listpix)
        for j in intersect:

            X_i = (X[i], sigmaX[i])
            X_j = (X[j], sigmaX[j])
            q = kolmogorov_smirnov_distance(
                x=X_i, y=X_j, ntests=ntests, nsamp=nresample
            )
            Qloc[i * npix + j] = q
            Qloc[j * npix + i] = q
    Qout= np.zeros_like(Qloc)
    comm.Allreduce(Qloc, Qout, op=MPI.SUM)
    Qout= Qout.reshape((npix,npix))
    return minmaxrescale(Qout , a=0, b=1)
