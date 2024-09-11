import healpy as hp
import pylab as pl
import numpy as np
import time
from mpi4py import MPI
import argparse
import os
from os import path
import fgcluster as fgc
from fgcluster.spectral_clustering_mpi import ( 
                                    estimate_Ritz_eigenpairs,
                                    from_ell_to_index)
from fgcluster.spectral_clustering_local import (
                                    build_adjacency_from_wasserstein_distance,
                                    build_adjacency_from_heat_kernel_local,
                                    sparsify_matrix,
                                    estimate_Laplacian_matrix_sparse, 
                                    build_eigenvector_distance )


from fgcluster import (
     utils as cu  )
from sklearn.cluster import AgglomerativeClustering



def main(args) :
    comm    = MPI.COMM_WORLD
    workdir =os. getcwd()
    rank =comm.Get_rank()
    nprocs =comm.Get_size()
    nside=args.nside
    nside_superpixel=8

    string =  args.spectral_parameter

    lmax= nside*3 
    cl= np.zeros(lmax-1)
    ell = np.arange(1,lmax ) 
    cl[:nside] = 1./ell[:nside] **2 

    param = hp. synfast (cl, nside=nside )
    param= cu.minmaxrescale(param, a=1.4, b=2 )

    sigmaparam = abs(param)*1e-2 
    sigmaparam = hp. synfast (cl, nside=nside )
    sigmaparam= cu.minmaxrescale(sigmaparam, a=param.min()/100, b=param.max()/100  )
    
    firstpix,lastpix  = cu . split_data_among_processors(size=hp.nside2npix(nside_superpixel), rank=rank , nprocs=nprocs )
    
    print(rank, firstpix,lastpix)
    
    local_patches = np.zeros_like( param  , dtype=pl.int_)
    c=0 
    for ipixloc in np.arange(firstpix,lastpix): 
        m_super = np.zeros(hp.nside2npix(nside_superpixel ))
        m_super[ipixloc]=1
        local_map =hp.ud_grade( m_super, nside_out=args.nside) 
        listp =np.where(local_map!=0)[0]
        X = param[listp] .copy() 
        sigmaX= sigmaparam[listp] .copy() 

        s= time.perf_counter()
        
        if np.bool_(args.KS_weight)  :
            Q =build_adjacency_from_wasserstein_distance(X = X, sigmaX=sigmaX ,nresample=256 , nsigma=2  ) 
        else:
            Q=None



        A = build_adjacency_from_heat_kernel_local(pixs =listp, nside=hp.get_nside(param ), 
                                                     KS_weighted= np.bool_(args.KS_weight) ,
                                                    Q=Q  ,alpha=args.KS_weight )
        """
        if rank ==0:
            pl.subplot(121)
            pl.title('Heat Kernel matrix  ')
            pl.imshow(np.log(A))
            pl.subplot(122)
            pl.title('KS distance matrix ')
            pl.imshow(np.log(Q)) 
            pl.show()
        """
        sparseA = sparsify_matrix(A)

        L = estimate_Laplacian_matrix_sparse(sparseA, kind='normalized'  ) 
        
        
        
        #TODO: n_eig value has to be further explored 

        l, W = estimate_Ritz_eigenpairs (L, n_eig = np.int_(cu.from_index_to_ell(A.shape[1] )  ))

        E = build_eigenvector_distance(W[:,1:] )

        clusters  = AgglomerativeClustering(distance_threshold=args.distance_threshold,connectivity= None, metric = 'precomputed',
                                    linkage='average', compute_full_tree=True, n_clusters=None   ).fit(E  )
        mapout_labels  =cu.mappify(arr=  clusters.labels_+1 +rank*100,  nside=hp.get_nside(param) , pixs=listp) 

        local_patches+= np.int_( mapout_labels)

        if c==1:break 
        c+=1
    ## reduce 
    
    patches=np.zeros_like(local_patches)
    comm.Allreduce(local_patches, patches, op=MPI.SUM )

    if rank ==0 :
        hp.mollview( patches,cmap=pl.cm.tab20 ); pl.show()
        fmap= f'{workdir}/clusterpatches/{string}_clusters_spectralclus_{args.KS_weight:.2f}_{args.nside}.fits'
        #hp.write_map(fmap,  patches, overwrite=True )

    
    comm.Disconnect 




    pass





if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )

	parser.add_argument("--spectral-parameter" , help='spectral  parameter (Bd, Td, Bs) to perform clustering' )
	parser.add_argument("--parameter-uncertainties", help = 'path to the  healpix  map of uncertainties',   )

	parser.add_argument("--nside", help="nside of output maps" ,required=True ,  type=np.int_)
	parser.add_argument("--KS-weight", help="weight to the KS distance to be combined with heat kernel" ,
                            required=True ,  type=np.float_)
	parser.add_argument("--distance-threshold", help = 'threshold to define connectivity in AgglomerativeClustering',
                                    default=0.3,  type=np.float_)

	args = parser.parse_args()


	main( args)