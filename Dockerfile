FROM ubuntu:latest
WORKDIR /tmp
MAINTAINER Giuseppe Puglisi <gpuglisi@berkeley.edu>

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        python3-dev          \
        git                  \
        python3-pip          \
        wget              && \
    apt-get clean all

ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

ARG mpi4py=3.0.0
ARG mpi4py_prefix=mpi4py-$mpi4py

RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    python3 setup.py build                                                   && \
    python3 setup.py install                                                 && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix

RUN /sbin/ldconfig

RUN \
     git clone    https://github.com/giuspugl/fgcluster.git && \
     cd fgcluster                                           && \
     git checkout spectral                                  && \
     pip3 install -r requirements.txt                       && \
     cd ..


RUN \
    git clone    https://github.com/fgbuster/fgbuster.git && \
    cd fgbuster                                           && \
    git checkout clusters                                 && \
    pip3 install -e .                                     && \
    cd ..




RUN \
    cd fgcluster                     && \
    python3 setup.py install         && \
    cd ..

 
