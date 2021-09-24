# ForeGroundCluster

FGCluster is a python package aimed at running spectral clustering onto Healpix maps.
The inputs are a map encoding the feature to cluster. In this case the pixel similarity is given by the geometrical affinity of each pixel in the sphere.
However, if uncertainty map is  provided as an input,   the adjacency is modified in such a way that the pixel similarity accounts *also* for the statistical significance given by the pixel values in a map  and the uncertainties.  For further details on the methodology please see Puglisi et al. 2021 .


## Requirements
```
numpy>=1.14
matplotlib>=2.0.0
scipy>=0.19.1
astropy>=1.2
scikit-learn>=0.2.4.1
healpy>=1.10.3
mpi4py>=3.0.3
```

## Install

```
git clone https://github.com/giuspugl/fgcluster.git
cd fgcluster
python setup.py install
```

## Test the installation
```
mpirun -np 4 python test_spectral_mpi.py

```

Notice that the whole package is implemented to work in parallel with `mpi4py`.  For debugging runs please use `-np 1  `.

## Tutorial Script

To learn how to use `FGCluster` we provided a tutorial notebook, available at [this link](https://github.com/giuspugl/fgcluster/blob/master/Tutorial.ipynb)
