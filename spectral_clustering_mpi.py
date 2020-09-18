import healpy as hp
import pylab as pl
import numpy as np

from scipy.stats import norm , ks_2samp
from mpi4py import MPI
from scipy import sparse
from scipy.special import legendre

import pysm
import pysm.units as u

from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import fgbuster.separation_recipes as sr
from fgbuster.visualization import corner_norm
from fgbuster.observation_helpers import get_instrument, get_sky
# Imports needed for component separation
from fgbuster import (CMB, Dust, Synchrotron,  # sky-fitting model
                      MixingMatrix)  # separation routine
import warnings
warnings.filterwarnings("ignore")



def split_data_among_processors(size ,rank , nprocs ):
    localsize = np.int_(size   /nprocs)
    remainder = size % nprocs

    if (rank < remainder) :
        start  = np.int_(rank * (localsize  + 1))
        stop =np.int_(  start + localsize +1  )

    else:
        start = np.int_(rank * localsize + remainder  )
        stop =np.int_(  start + (localsize  )   )
    return start, stop



def get_neighbours(ipix,nside,  order ):
    if order==0:
        return np.unique(hp.get_all_neighbours(theta=ipix,nside= nside  ))
    else:
        ipix =   np.unique(hp.get_all_neighbours(theta=ipix,nside= nside ) )
        return get_neighbours(ipix,nside,order-1 )


def KS_distance(x,y) :
    ## Source of the tabulated values  from
    ## http://www.matf.bg.ac.rs/p/files/69-[Michael_J_Panik]_Advanced_Statistics_from_an_Elem(b-ok.xyz)-776-779.pdf
    # We consider 95th quantile (alpha=0.05 )

    mu1= x[0]; sigma1=x[1];
    mu2= y[0]; sigma2=y[1] ;
    ntests =50
    n=100
    D=[]
    # We can repeat the KS test ntest times (it takes time) on two samples with same size n=m

    for test in range(ntests):
        rvs1 = norm.rvs(size=n, loc=mu1, scale=sigma1)
        rvs2 = norm.rvs(size=n , loc=mu2, scale=sigma2 )
        #D,pval =ks_2samp(rvs1, rvs2)
        #D.append(ks_2samp(rvs1, rvs2)[0] )
        _, pval =ks_2samp(rvs1, rvs2)
        D.append(pval )
    alpha= .05

    med = np.median(D)
    #return med
    dicmed ={}
    if med < alpha :
        return 1e-2
    elif alpha<med<0.1:
        return 0.1
    else:
        return 1.

def get_lmax(nside , stop_threshold ):
    pixscale =hp.nside2resol(nside=nside  )
    sigmabeam  =  ( pixscale  )/ pl.sqrt(8*np.log(2))
    ell =np.arange(  3*nside -1  )
    arr =np.array( [ (2*l  +1 )/4./np.pi* np.exp(- sigmabeam  *l *(l+1) ) for l in ell  ])
    lmax  = np.argmin(np.fabs(arr -stop_threshold))

    return lmax, sigmabeam

def heat_kernel (Theta, l,sigma ):
    return (2*l+1 )/4./np.pi *np.exp(- sigma   *l *(l+1) ) *legendre(l )  (Theta)


def build_adjacency_from_heat_kernel(nside,comm, stopping_threshold=1e-7 ):
    if comm is None :
        rank =0
        nprocs = 1
    else :
        rank =comm.Get_rank()
        nprocs  = comm.Get_size()

    p = np.arange(hp.nside2npix(nside))
    V =np.array(hp.pix2vec(ipix =p, nside=hp.get_nside(p))).T
    scalprod=V.dot(V.T)

    lmax, sigmabeam = get_lmax (nside, stopping_threshold )

    Gloc = np.zeros_like(scalprod )

    start_ell,stop_ell =split_data_among_processors(lmax, rank, nprocs )
    for l in np.arange(start_ell,stop_ell  ):
        Gloc += heat_kernel (scalprod, l, sigmabeam)
    G=    np.zeros_like(Gloc )
    comm.Reduce(Gloc , G , op=MPI.SUM)
    return G

def build_adjacency_from_nearest_neighbours( nside,  comm,neighbour_order=1 ,KS_weighted=False, X=None ,sigmaX=None):

    if comm is None :
        rank =0
        nprocs = 1
    else :
        rank =comm.Get_rank()
        nprocs  = comm.Get_size()


    npix = hp.nside2npix(nside)
    Dloc=np.zeros((npix,npix ))
    #Identify for each pixels what are the nearest neighbours,
    #the order specifies how many nearest neighbours to include:
    # the higher the order, the more are the neighbours included
    start,stop= split_data_among_processors(size= npix ,rank= rank, nprocs=nprocs  )

    for i  in np.arange(start,stop ) :
        neighbours = get_neighbours(ipix=i ,nside= nside ,order=neighbour_order)
        Dloc[i,neighbours] = 1
    D  =  np.zeros_like(Dloc )
    comm.Allreduce(Dloc , D , op=MPI.SUM)

    if not KS_weighted :
        return D

    else :
    ## Computation of KS weights, computation is distributed among processing elements

    # We also want to exploit the symmetry  properties of the distance,i.e. D(i,j)= D(j,i)
    # we therefore need only to consider  the indices of neighbour pixels in the upper diagonal part of D
    # D(i,i ) = 0 (we don't estimate that! )
    # this reduces   the computation time !
        uD =np.zeros_like(D)
        Indices = np.triu_indices(npix ,1)

        uD[Indices] = D[Indices]
        mask = np.array( np.where(uD==1) )
        start,stop= split_data_among_processors(size= mask[0].size,rank= rank, nprocs=nprocs  )

        Dweighted_local  = np.zeros_like(D )
        for i,j   in  zip(mask[0][start:stop],mask[1][start:stop] ) :

            X_i= (X[i], sigmaX[i])
            X_j= (X[j], sigmaX[j])
            q=KS_distance(x=X_i,y=X_j )
            Dweighted_local[i,j]= q
            Dweighted_local[j,i]=Dweighted_local[i,j]
        Dweighted =  np.zeros_like(Dweighted_local)
        comm.Reduce(Dweighted_local , Dweighted , op=MPI.SUM)
        return Dweighted

def estimate_Laplacian_matrix (W , kind ='unnormalized'):

    ### Matrices that we are dealing with are sparse, it's more efficient to defined them as
    ### Compressed Sparse Row  (CSR ) matrices with scipy .

    Ws  = sparse.csr_matrix(W)
    D = W.dot(np.ones_like(W[0,:]))

    Dinv = 1./ (D)
    Dinv2 = 1./ pl.sqrt( D)
    Ds =sparse.spdiags(D, diags=0, m= W.shape[0],n=W.shape[1])
    L = sparse.csr_matrix( Ds- Ws )

    if kind=='unnormalized':
        return L
    elif kind=='normalized':
        Dinvs   = sparse.spdiags(Dinv, diags=0, m= W.shape[0],n=W.shape[1])
        Lnorm= Dinvs.dot(L)
        return Lnorm
    elif kind=='symmetric':
        Dinv2s   = sparse.spdiags(Dinv2, diags=0, m= W.shape[0],n=W.shape[1])
        Lsym= Dinv2s .dot(L.dot(Dinv2s))
        return Lsym

def estimate_Ritz_eigenpairs (L,n_eig = 600, tolerance=1e-12 ):
    eval,evec = sparse.linalg.eigsh(L, k=n_eig ,   return_eigenvectors=True, which="SM",tol=tolerance    )
    return eval,evec

def build_distance_matrix_from_eigenvectors(W ,comm  ):
    rank =comm.Get_rank()
    nprocs  = comm.Get_size()
    comm.Bcast(W, root=0)

    npix = W.shape [0]
    Indices = np.array (np.triu_indices(npix ,1) )
    #assigning a symmetric matrix values in mpi
    globalsize= Indices.shape[1]
    start,stop =split_data_among_processors(size= globalsize, rank=rank,nprocs=nprocs )
    Dloc = np.zeros( (npix,npix ))
    for i,j  in   (Indices[:,start:stop] .T) :
        # the euclidean distance is estimated considering npix samples of m features
        # (each row of W  is an  m-dimensional (m being the number  of columns of W)
        Dloc[i,j] =  np.linalg.norm(W[i,:] - W[j,:]  )
        Dloc[j,i] =Dloc[i,j]
    D=  np.zeros_like(Dloc )
    comm.Reduce(Dloc , D, op=MPI.SUM)
    return D


def main( ):
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    nside=8
    X=np.ones( hp.nside2npix(nside )) #np.random.uniform(size= hp.nside2npix(nside ))
    sigmaX=np.ones_like(X)*.2
    import time
    s =time.perf_counter ()

    A  = build_adjacency_from_nearest_neighbours(  nside=nside , neighbour_order=0,
                                comm=comm,  )
    e = time.perf_counter ()
    if rank==0 :
        #pl.imshow(A  )
        #pl.show()
        print(f"build_adjacency_from_nearest_neighbours, execution time {e-s }sec")

    s =time.perf_counter ()

    A  = build_adjacency_from_nearest_neighbours(  nside=nside , neighbour_order=0,
                                    comm=comm,
                                    KS_weighted=True, X=X, sigmaX=X  )
    e = time.perf_counter ()
    if rank==0 :
        print(f"build_adjacency_from_nearest_neighbours (weighted), execution time {e-s }sec")
    s =time.perf_counter ()
    A = build_adjacency_from_heat_kernel (nside, comm  )
    e = time.perf_counter ()

    if rank==0 :
        #pl.imshow(A  )
        #pl.show()
        print(f"build_adjacency_from_heat_kernel, execution time {e-s }sec")


    L = estimate_Laplacian_matrix(A )

    #if rank==0 :
        #pl.imshow(L.todense()  )
        #hp.mollview(L.toarray()[100,:],  title="L", cmap='coolwarm', min=-17,max=17,);pl.show()
    s =time.perf_counter ()
    l, W = estimate_Ritz_eigenpairs (L, n_eig = 6 )
    e = time.perf_counter ()
    if rank==0 :
        pl.plot(l, 'o');pl.show()
        #hp.mollview(W[:,0]);pl.show()
        print(f"estimate_Ritz_eigenpairs, execution time {e-s }sec")

    #we don't consider the smallest eigenvectors since it's the constant vector
    s=time.perf_counter()
    E = build_distance_matrix_from_eigenvectors(W[:,1:] ,comm=comm )
    e = time.perf_counter ()
    if rank==0 :
        pl.imshow(E);pl.colorbar();pl.show()
        print(f"build_distance_matrix_from_eigenvectors, execution time {e-s }sec")

    comm.Disconnect

    return
main ()
