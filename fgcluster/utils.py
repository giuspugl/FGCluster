import warnings
import pylab as pl
import healpy as hp
import astropy.units as u

import numpy as np


def plotclusters(labels, imap):
    outm = pl.zeros_like(imap)
    for i in range(labels.max() + 1):
        pixs = pl.where(labels == i)
        outm[pixs] = imap[pixs].mean()

    return outm


def check_nside(nsideout, mapin, verbose=False):
    nside2 = hp.get_nside(mapin)
    if nside2 != nsideout:
        if verbose:
            print("running ud_grade ")
        return hp.ud_grade(nside_out=nsideout, map_in=mapin)
    else:
        return mapin


def get_scatter_info(N, nprocs):
    """
    Split input array with size `N` by the number of processes `nprocs`.
    """
    datatoscatter = np.zeros((N, N))
    split = np.array_split(datatoscatter, nprocs, axis=0)
    split_sizes = np.array([split[i].shape[0] for i in range(nprocs)], dtype="int")

    split_sizes_in = split_sizes * N
    offset_in = np.insert(np.cumsum(split_sizes_in), 0, 0)[0:-1]

    return offset_in, split_sizes


def extend_matrix(mask, compressed_matr, fill_value=0):
    """
    Given a compressed affinity matrix (evaluated outside a masked region defined by mask)
    we expand into a Npix X Npix affinity matrix  with non zero elements mapped consistently
    from the compressed matrix .

    - `mask`: {`np.array`}
        binary mask in the healpix format. It's True (False) outside (inside) the masked region
    - `compressed_matr` : {`np.array`}
        matrix with adjacency defined outside the masked region,   encoding less pixels than
        a matrix built from a full sky healpix map.
    - `fill_value`:{float }
        value to fill the matrix elements in correspondence of the masked area

    - `expanded_matr` :{`np.array`}
        matrix _expanded_ with full size with all the pixels encoded in a Healpix map.
        Matrix elements in the masked area are set to zero by default.(see `fill_value`)
    """
    Npix = hp.nside2npix(hp.get_nside(mask))
    mask = np.expand_dims(mask, axis=1)
    mask2d = np.bool_(mask.dot(mask.T))

    expanded_matr = np.zeros((Npix, Npix))
    expanded_matr[mask2d] = compressed_matr.flatten()
    expanded_matr[~mask2d] = fill_value
    return expanded_matr


def get_under_over_partition_measures(K, labels, W):
    """
    Estimate the within- and between-cluster variance  for a configuration with
    `K` clusters, labelled with `labels` and estimated from the distance matrix `W`.
     See definitions in sect. 7.2 of
    http://scholar.google.com/scholar?hl=en&btnG=Search&q=intitle:Chapter+15+-+Clustering+Methods#4

    **Returns**
    - `under`: {float}
    quantifies the under-partition measure, i.e. the within-cluster variance
    - `over`:{float }
    quantifies the over-partition measure, i.e. the between-cluster variance
    """
    mu = pl.zeros(K * W.shape[1]).reshape(K, W.shape[1])
    mean_intra_cluster_D = pl.zeros(K)
    for k in range(K):
        ck = pl.where(labels == k)[0]
        Xk = W[ck, :]
        mu[k] = Xk.mean(axis=0)
        Nk = len(ck)
        if Nk <= 1:
            continue
        E = pairwise_distances(X=Xk, Y=mu[k].reshape(1, -1), metric="euclidean")
        mean_intra_cluster_D[k] = E.sum() / Nk
    inter_cluster_D = pairwise_distances(mu, metric="euclidean")
    pl.fill_diagonal(inter_cluster_D, pl.inf)
    under = mean_intra_cluster_D.sum() / K
    over = K / inter_cluster_D.min()
    return under, over


def minmaxrescale(x, a=0, b=1):
    """
    Performs  a MinMax Rescaling on an array `x` to a generic range :math:`[a,b]`.
    """
    xresc = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
    return xresc


def get_lmax(nside, stop_threshold):
    """
    Estimate  to which `lmax`    the heat kernel estimation  is stopped at a
    given `nside` resolution and a threshold `stop_threshold`.

    """
    pixscale = hp.nside2resol(nside=nside)
    sigmabeam = (pixscale) / pl.sqrt(8 * np.log(2))
    ell = np.arange(3 * nside - 1)
    arr = np.array(
        [(2 * l + 1) / 4.0 / np.pi * np.exp(-sigmabeam * l * (l + 1)) for l in ell]
    )
    lmax = np.argmin(np.fabs(arr - stop_threshold))

    return lmax, sigmabeam


def get_neighbours(ipix, nside, order):
    """
    Given a pixel index in the Healpix pixelization and the nside parameter, estimated
    the indices of nearest neighbour pixels. The larger is   `order`,
     the farther neighbours are included.
    """

    if order == 0:
        return np.unique(hp.get_all_neighbours(theta=ipix, nside=nside))
    else:
        ipix = np.unique(hp.get_all_neighbours(theta=ipix, nside=nside))
        return get_neighbours(ipix, nside, order - 1)


def split_data_among_processors(size, rank, nprocs):
    """
    Split in a load-balanced way the size of an array equally among the processes

    """
    localsize = np.int_(size / nprocs)
    remainder = size % nprocs

    if rank < remainder:
        start = np.int_(rank * (localsize + 1))
        stop = np.int_(start + localsize + 1)

    else:
        start = np.int_(rank * localsize + remainder)
        stop = np.int_(start + (localsize))
    return start, stop


def from_ell_to_index(ell):
    """
    Returns the range of column values assuming  a matrix with columns ordered
    with the m multipole , m ranging from -ell to +ell
    """
    return ell ** 2, ell ** 2 + 2 * ell + 1


def from_index_to_ell(index):
    """
    given an index in the matrix with columns ordered
    with the m multipole, returns the  ell multipole
    """
    ell = np.floor(np.sqrt(index))
    return ell
