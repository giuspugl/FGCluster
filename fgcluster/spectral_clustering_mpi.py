import healpy as hp
import pylab as pl
import numpy as np

from scipy.stats import norm, ks_2samp
from scipy import sparse
from scipy.special import legendre
from sklearn.metrics import pairwise_distances

from mpi4py import MPI

import warnings

warnings.filterwarnings("ignore")


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


def kolmogorov_smirnov_distance(x, y, ntests, nsamp):

    mu1 = x[0]
    sigma1 = x[1]
    mu2 = y[0]
    sigma2 = y[1]
    D = []
    # We  repeat the KS test ntest times (it takes time) on two samples with same size n=m
    #np.random.seed(1234)

    for test in range(ntests):
        rvs1 = norm.rvs(size=nsamp, loc=mu1, scale=sigma1)
        rvs2 = norm.rvs(size=nsamp, loc=mu2, scale=sigma2)
        _, pval = ks_2samp(rvs1, rvs2)
        D.append(pval)
    med = np.median(D)
    return med


def get_lmax(nside, stop_threshold):
    pixscale = hp.nside2resol(nside=nside)
    sigmabeam = (pixscale) / pl.sqrt(8 * np.log(2))
    ell = np.arange(3 * nside - 1)
    arr = np.array(
        [(2 * l + 1) / 4.0 / np.pi * np.exp(-sigmabeam * l * (l + 1)) for l in ell]
    )
    lmax = np.argmin(np.fabs(arr - stop_threshold))

    return lmax, sigmabeam


def heat_kernel(Theta, l, sigma):
    return (2 * l + 1) / 4.0 / np.pi * np.exp(-sigma * l * (l + 1)) * legendre(l)(Theta)


def minmaxrescale(x, a=0, b=1):
    """
    Performs  a MinMax Rescaling on an array `x` to a generic range :math:`[a,b]`.
    """
    xresc = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
    return xresc


def build_adjacency_from_heat_kernel(     nside, comm, stopping_threshold=1e-7, KS_weighted=False, Q=None, alpha=0.5):

    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    p = np.arange(hp.nside2npix(nside))
    V = np.array(hp.pix2vec(ipix=p, nside=hp.get_nside(p))).T
    scalprod = minmaxrescale(V.dot(V.T), a=-1, b=1)

    if KS_weighted:
        Theta = np.arccos(scalprod)
        Psi = Theta + alpha * (1 - Q) * np.pi / 2
        scalprod = np.cos(Psi)

    lmax, sigmabeam = get_lmax(nside, stopping_threshold)

    Gloc = np.zeros_like(scalprod)

    start_ell, stop_ell = split_data_among_processors(lmax, rank, nprocs)
    for l in np.arange(start_ell, stop_ell):
        Gloc += heat_kernel(scalprod, l, sigmabeam)
    Gloc = comm.allreduce(Gloc, op=MPI.SUM)

    return Gloc


def build_adjacency_from_KS_distance(nside, comm, X, sigmaX, ntests=50, nresample=100):
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    npix = hp.nside2npix(nside)
    Indices = np.array(np.triu_indices(npix, 1))
    # Qloc = np.zeros( (npix,npix ))
    Qloc = np.zeros(npix * npix)
    start, stop = split_data_among_processors(
        size=Indices[0].size, rank=rank, nprocs=nprocs
    )

    for i, j in Indices[:, start:stop].T:
        X_i = (X[i], sigmaX[i])
        X_j = (X[j], sigmaX[j])
        q = kolmogorov_smirnov_distance(x=X_i, y=X_j, ntests=ntests, nsamp=nresample)
        Qloc[i * npix + j] = q
        Qloc[j * npix + i] = q

    Qloc = comm.allreduce(Qloc, op=MPI.SUM)
    Qloc = Qloc.reshape((npix, npix))
    return minmaxrescale(Qloc, a=0, b=1)

def build_adjacency_from_KS_distance2(nside, comm, X, sigmaX, ntests=50, nresample=100):
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    npix = hp.nside2npix(nside)
    Indices = (np.triu_indices(npix, 1))
    Qloc = np.zeros(npix * npix)
    #    start, stop = split_data_among_processors(
    #        size=Indices[0].size, rank=rank, nprocs=nprocs
    #    )
    for i in range(npix):
        i_indx =np.ma.masked_equal(Indices[0],i ) .mask
        j_indx =Indices[1][i_indx]
        listpix = get_neighbours(ipix=i,nside=nside,order=6 )
        intersect = np.intersect1d(j_indx, listpix )
        for j in j_indx :

            X_i = (X[i], sigmaX[i])
            X_j = (X[j], sigmaX[j])
            q = kolmogorov_smirnov_distance(x=X_i, y=X_j, ntests=ntests, nsamp=nresample)
            Qloc[i * npix + j] = q
            Qloc[j * npix + i] = q

    #Qloc = comm.allreduce(Qloc, op=MPI.SUM)
    Qloc = Qloc.reshape((npix, npix))
    return minmaxrescale(Qloc, a=0, b=1)

def build_adjacency_from_heat_kernel_gather(nside, comm, stopping_threshold=1e-7, KS_weighted=False, Q=None, alpha=0.5):

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
    start_row, end_row = split_data_among_processors(size=npix, rank=rank, nprocs=nprocs )
    # Create loc array on each core
    Gloc = np.zeros(
        (end_row-start_row , npix ), dtype=np.float_
    )
    offset_hpx = start_row
    for l in np.arange(lmax):
        Gloc += heat_kernel(
            scalprod[offset_hpx : offset_hpx + Gloc.shape[0], :], l, sigmabeam
        )
    comm.Barrier()
    G = comm.allgather(Gloc)
    G = np.concatenate(G).reshape((npix, npix))
    return G

def build_adjacency_from_KS_distance_gather(nside, comm, X, sigmaX, ntests=50, nresample=100):

    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    #npix = hp.nside2npix(nside)
    npix = X.size
    start_row, end_row = split_data_among_processors(size=npix, rank=rank, nprocs=nprocs )
    Qloc = np.zeros(
        (end_row-start_row , npix ), dtype=np.float_
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
    Q = comm.allgather(Qloc)
    Q = np.concatenate(Q).reshape((npix, npix))
    Q[np.diag_indices(npix)] = 0.0
    return minmaxrescale(Q, a=0, b=1)


def build_adjacency_from_heat_kernel_savedata(nside, comm, stopping_threshold=1e-7, KS_weighted=False,
                    Q=None, alpha=0.5, matrixdir="./", read_from_disc=True ):

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
    start_row, end_row = split_data_among_processors(size=npix, rank=rank, nprocs=nprocs )
    # Create loc array on each core
    Gloc = np.zeros(
        (end_row-start_row , npix ), dtype=np.float_
    )
    offset_hpx = start_row
    for l in np.arange(lmax):
        Gloc += heat_kernel(
            scalprod[offset_hpx : offset_hpx + Gloc.shape[0], :], l, sigmabeam
        )
    np.save(f"{matrixdir}/localGmatr_proc{rank}.npy",  Gloc)

    comm.Barrier()
    if read_from_disc :
        G= np.zeros((npix,npix ))
        for proc in range(nprocs):
            qmatr = np.load(f"{matrixdir}/localGmatr_proc{proc}.npy", allow_pickle=True)
            start ,end = split_data_among_processors(size=npix, rank=proc , nprocs=nprocs )
            G[ start:end ,:  ] = qmatr
        return G
    else:
        pass


def build_adjacency_from_KS_distance_savedata( nside, comm, X, sigmaX, ntests=50,
                        nresample=100, matrixdir="./", read_from_disc=True ):

    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    npix = hp.nside2npix(nside)

    start_row, end_row = split_data_among_processors(size=npix, rank=rank, nprocs=nprocs )
    Qloc = np.zeros(
        (end_row-start_row , npix ), dtype=np.float_
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

    np.save(f"{matrixdir}/localQmatr_proc{rank}.npy", Qloc)
    if read_from_disc:
        Q= np.zeros((npix,npix ))

        for proc in range(nprocs):
            qmatr = np.load(f"{matrixdir}/localQmatr_proc{proc}.npy", allow_pickle=True )
            start ,end = split_data_among_processors(size=npix, rank=proc , nprocs=nprocs )
            Q[ start:end ,:  ] = qmatr
        Q[np.diag_indices(npix)] = 0.0
        return minmaxrescale(Q, a=0, b=1)
    else:
        pass


def build_adjacency_from_nearest_neighbours(
    nside,
    comm,
    neighbour_order=1,
    KS_weighted=False,
    X=None,
    sigmaX=None,
    ntests=50,
    nresample=100,
):

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

    if not KS_weighted:
        return D

    else:
        ## Computation of KS weights, computation is distributed among processing elements

        # We also want to exploit the symmetry  properties of the distance,i.e. D(i,j)= D(j,i)
        # we therefore need only to consider  the indices of neighbour pixels in the upper diagonal part of D
        # D(i,i ) = 0 (we don't estimate that! )
        # this reduces   the computation time !
        uD = np.zeros_like(D)
        Indices = np.triu_indices(npix, 1)

        uD[Indices] = D[Indices]
        mask = np.array(np.where(uD == 1))
        start, stop = split_data_among_processors(
            size=mask[0].size, rank=rank, nprocs=nprocs
        )

        Dweighted_local = np.zeros_like(D)
        for i, j in zip(mask[0][start:stop], mask[1][start:stop]):

            X_i = (X[i], sigmaX[i])
            X_j = (X[j], sigmaX[j])
            q = kolmogorov_smirnov_distance(
                x=X_i, y=X_j, ntests=ntests, nsamp=nresample
            )
            Dweighted_local[i, j] = q
            Dweighted_local[j, i] = Dweighted_local[i, j]
        Dweighted_local = comm.allreduce(Dweighted_local, op=MPI.SUM)
        ## we define  the KS distance   as a sin (Qij *pi/2 ) , with Qij the quantile of KS test
        return np.sin(Dweighted_local * np.pi / 2)


def estimate_Laplacian_matrix(W, kind="unnormalized"):

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
    eval, evec = sparse.linalg.eigsh(
        L, k=n_eig, return_eigenvectors=True, which="SM", tol=tolerance
    )
    return eval, evec


def get_under_over_partition_measures(K, labels, W):
    """
    See definitions in sect. 7.2 of
    http://scholar.google.com/scholar?hl=en&btnG=Search&q=intitle:Chapter+15+-+Clustering+Methods#4
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


def build_distance_matrix_from_eigenvectors(W, comm):
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    npix = W.shape[0]
    Indices = np.array(np.triu_indices(npix, 1))
    # assigning a symmetric matrix values in mpi
    globalsize = Indices.shape[1]
    start, stop = split_data_among_processors(size=globalsize, rank=rank, nprocs=nprocs)
    Dloc = np.zeros((npix, npix))
    for i, j in Indices[:, start:stop].T:
        # the euclidean distance is estimated considering npix samples of m features
        # (each row of W  is an  m-dimensional (m being the number  of columns of W)
        Dloc[i, j] = np.linalg.norm(W[i, :] - W[j, :])
        Dloc[j, i] = Dloc[i, j]
    Dloc = comm.allreduce(Dloc, op=MPI.SUM)
    return Dloc


def get_scatter_info( N, nprocs):
    # Split input array by the number of   nprocs
    datatoscatter = np.zeros((N,N))
    split = np.array_split(datatoscatter, nprocs, axis=0)
    split_sizes =  np.array ([  split[i].shape[0 ] for i in range(nprocs )], dtype='int' )

    split_sizes_in = split_sizes * N
    offset_in = np.insert(np.cumsum(split_sizes_in), 0, 0)[0:-1]

    return offset_in, split_sizes


def extend_matrix(mask ,compressed_matr , fill_value=0   ):
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
    Npix = hp.nside2npix(hp.get_nside(mask) )
    mask= np.expand_dims(mask,axis=1)
    mask2d = np.bool_(mask.dot(mask.T))


    expanded_matr =np.zeros((Npix,Npix ))
    expanded_matr [mask2d ]= compressed_matr.flatten()
    expanded_matr [~mask2d ]=  fill_value
    return expanded_matr



def build_adjacency_from_KS_distance_nn(nside, comm, X, sigmaX,
                        order_nn, ntests=50, nresample=100):
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    npix = hp.nside2npix(nside)
    Indices = (np.triu_indices(npix, 1))
    Qloc = np.zeros(npix * npix)
    start, stop = split_data_among_processors(
            size=Indices[0].size, rank=rank, nprocs=nprocs
        )

    for i in range(start,stop ):
        i_indx =np.ma.masked_equal(Indices[0],i ) .mask
        j_indx =Indices[1][i_indx]
        listpix = get_neighbours(ipix=i,nside=nside,order=order_nn)
        intersect = np.intersect1d(j_indx, listpix )
        for j in intersect :

            X_i = (X[i], sigmaX[i])
            X_j = (X[j], sigmaX[j])
            q = kolmogorov_smirnov_distance(x=X_i, y=X_j, ntests=ntests, nsamp=nresample)
            Qloc[i * npix + j] = q
            Qloc[j * npix + i] = q

    Qloc = comm.allreduce(Qloc, op=MPI.SUM)
    Qloc = Qloc.reshape((npix, npix))
    return minmaxrescale(Qloc, a=0, b=1)

def statistical_compatibility (x, y  ):

    mu1 = x[0]
    sigma1 = x[1]
    mu2 = y[0]
    sigma2 = y[1]
    significance = pl.fabs( mu1-mu2)/pl.sqrt(sigma1**2 +sigma2**2)
    return significance

def build_adjacency_from_compatibility(nside, comm, X, sigmaX):
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    npix = hp.nside2npix(nside)
    Indices = (np.triu_indices(npix, 1))
    Qloc = np.zeros( (npix,npix ))
    start, stop = split_data_among_processors(
            size=Indices[0].size, rank=rank, nprocs=nprocs
        )

    for i  in Indices[0][start:stop ]:
        i_indx =np.ma.masked_equal(Indices[0],i ) .mask
        j_indx =Indices[1][i_indx]
        X_i = (X[i], sigmaX[i])
        X_j = (X[j_indx], sigmaX[j_indx ])
        q = statistical_compatibility(x=X_i, y=X_j)

        Qloc[i , intersect ] =  1-  q
        Qloc[intersect  , i] =   1-   q
    mask =Qloc <0
    Qloc[mask]= 0
    Qloc = comm.allreduce(Qloc, op=MPI.SUM)
    return Qloc

def build_adjacency_from_compatibility_nn(nside, comm, X, sigmaX, order_nn):
    if comm is None:
        rank = 0
        nprocs = 1
    else:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    npix = hp.nside2npix(nside)
    Indices = (np.triu_indices(npix, 1))
    Qloc = np.zeros((npix , npix))
    start, stop = split_data_among_processors(
            size=Indices[0].size, rank=rank, nprocs=nprocs
        )

    for i in range(start, stop  ):

        i_indx =np.ma.masked_equal(Indices[0],i ) .mask
        j_indx =Indices[1][i_indx]
        listpix = get_neighbours(ipix=i,nside=nside,order=order_nn)
        intersect = np.intersect1d(j_indx, listpix )
        X_i = (X[i], sigmaX[i])
        X_j = (X[intersect ], sigmaX[intersect])

        q = statistical_compatibility(x=X_i, y=X_j)

        Qloc[i , intersect ] =  1-  q
        Qloc[intersect  , i] =   1-   q
    mask =Qloc <0
    Qloc[mask]= 0
    Qloc = comm.allreduce(Qloc, op=MPI.SUM)
    return Qloc
