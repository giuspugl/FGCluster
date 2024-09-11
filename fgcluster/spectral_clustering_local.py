from scipy.stats import wasserstein_distance 
from scipy import sparse 
from scipy.sparse import  coo_array as coo 
import healpy as hp
import pylab as pl
import numpy as np

import fgcluster as fgc
from fgcluster import (
     utils as cu  )

def build_adjacency_from_heat_kernel_local (
   pixs,nside, stopping_threshold=1e-7, KS_weighted=False, Q=None, alpha=0.5
):
    """
    Estimate the Adjacency matrix in each sub-map .

    **Parameters**

    - `nside`:{int}
    resolution parameter of the healpix map
    - `pixs`:{list of int}
    list of pixels considered in the sub-map 
    -`stopping_threshold` :{float}
    the threshold to stop the sum over `l` of the heat kernel
    - `KS_weighted`:{bool}
    boolean to distort the adjacency with the KS distance
    - `Q`:{np.matrix}
    matrix encoding the KS distance
    - `alpha`:{float}
    the weight to set the amplitude of the KS distortion
    """
    
    V = np.array(hp.pix2vec(ipix=pixs, nside= nside )).T
    
    scalprod = cu.minmaxrescale(V.dot(V.T), a=-1, b=1)
    if KS_weighted:
        Theta = np.arccos(scalprod)
        Psi = Theta + alpha *Q 
        scalprod = np.cos(Psi)

    lmax, sigmabeam = cu.get_lmax(nside, stopping_threshold)
    Gloc=np.zeros_like(scalprod) 
    for l in np.arange(lmax):
        Gloc += fgc. heat_kernel(
            scalprod, l, sigmabeam
        )
    return Gloc



def build_adjacency_from_wasserstein_distance(
     X, sigmaX, nresample=256, nsigma=  2 ):
    
    """
    Build the adjacency matrix  accounting for the uncertainties. 
    We  bootstrap resample with a normal distribution   each X, deltaX pair. 
    We perform on the bootstrapped sample  the Wasserstein distance  (WD)
    and we exclude all the points whose WD > WD (nsigma ). Where the latter is estimated by considering 
    the a sample whose value is centered around the same X value but with nsigma*deltaX uncertainty. 


    - `X`:{np.array}
    features to cluster
    - `sigmaX`:
    {np.array}
    uncertainties of the  features to cluster
    - `nresample`:{int}
    size of the bootstrap resampling of a measurement  (X1,sigmaX1)
    -  `nsigma` {int}
    confidence level distance to exclude distant points. (default= 2_

    """
    npix = X.size
    values = np.random.normal (loc=X, scale=sigmaX  , size=(nresample , npix ) ) 
    Qloc = np.zeros((npix, npix), dtype=np.float_)  # Create Qloc array on each core
    for i in range(npix-1):
        v2 = np.random.normal(loc=X[i]+nsigma *sigmaX[i], scale=nsigma *sigmaX[i] , size=nresample ) 
        
        wdist = lambda x: wasserstein_distance(values[:,i],x  ) 
        dist_nsigma = wdist(v2) 
        
        q= np.hstack(list(map(wdist, values[:,i:].T ) ) )
        mask= q>dist_nsigma 
        q[mask ]  = np.pi/2
        q[~mask]  = 0
        Qloc[i, i:] = q
        Qloc[ i: ,i ] = q
        
    return Qloc 


def sparsify_matrix(A, threshold=1e-3):
    """
    Once the affinity matrix is built, is full of zero values, 
    we build a sparse COO matrix (_ a sparse matrix in COOrdinate format _ ) 
    considering only the non-zero values (threshold value need to be specified).

    this will be used to estimate the Laplacian matrix.
    """
    rows, cols  =np.where( np.log10(A) >threshold ) 
    mask =  np.log10(A) >threshold 
    sparseA = coo  ( (A[mask] , (rows, cols))  , shape = A.shape ) 
    return sparseA



def estimate_Laplacian_matrix_sparse(W, kind="unnormalized"):
    """
    Estimate the Laplacian matrix from the sparse matrix of weights`W`.
    3 implementation of the Laplacian are available :
    - unnormalized
    - normalized
    - symmetric


    """

    ### Matrices that we are dealing with are sparse, it's more efficient to defined them as
    ### Compressed Sparse Row  (CSR ) matrices with scipy .
    
    D = sparse.spdiags(W.sum(axis=0)   , diags=0, m=W.shape[0], n=W.shape[1])
    L =  (D - W)
    #pl.imshow(W.toarray()  ) 
    if kind == "unnormalized":
        return L
    elif kind == "normalized":
        Dinvs = sparse.spdiags(1/W.diagonal () , diags=0, m=W.shape[0], n=W.shape[1])
        Lnorm = Dinvs.dot(L)
        return Lnorm
    elif kind == "symmetric":
        Dinv2s = sparse.spdiags(1/np.sqrt(W.diagonal () ) , diags=0, m=W.shape[0], n=W.shape[1])
        Lsym = Dinv2s.dot(L.dot(Dinv2s))
        return Lsym
    else:
        raise SyntaxError(
            ' "kind" can   be one among :  "unnormalized" ,  "normalized" , "symmetric"  '
        )



def build_eigenvector_distance (X ):
    
    """
    Estimate the Euclidean distance from the matrix built from the
    Laplacian eigenvectors  `W`. 
    """
    npix = X.shape [0] 

    Devec = np.zeros((npix, npix), dtype=np.float_)  # Create Qloc array on each core
    for i in range(npix-1  ):
        # the euclidean distance is estimated considering npix samples of m features
        # (each row of W  is an  m-dimensional (m being the number  of columns of W)
        norm2= lambda v: np.linalg.norm(X[i] -  v)
        q = np.array (list(map(norm2 ,X[i:] )))
        Devec[i, i:] =  q
        
        Devec[i:, i] =  q
        #print(q) 
        #break 
    return Devec 





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