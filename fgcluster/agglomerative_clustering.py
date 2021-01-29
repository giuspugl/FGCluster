import warnings
import pylab as pl
import healpy as hp
from sklearn import preprocessing
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import astropy.units as u
import os
from scipy.interpolate import interp1d


from .estimate_component_separation_residuals import (
    estimate_Stat_and_Sys_residuals,
    estimate_spectra,
)
from .utils import hellinger_distance
import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)


class ClusterData:
    def estimate_haversine(self):
        if self.verbose:
            print("Estimating Spatial locality with Haversine distance .")
        self.haversine = DistanceMetric.get_metric("haversine")
        # prepare features  Coordinates
        longi, lat = pl.deg2rad(
            hp.pix2ang(
                nside=self._nside,
                ipix=pl.arange(hp.nside2npix(self._nside)),
                lonlat=True,
            )
        )
        mask = pl.ma.masked_inside(longi, 0, pl.pi).mask
        longi[mask] = -longi[mask]
        longi[pl.logical_not(mask)] = 2 * pl.pi - longi[pl.logical_not(mask)]
        Theta = pl.array([lat[self.galactic_mask], longi[self.galactic_mask]]).T

        angdist_matr = self.haversine.pairwise(Theta)
        angdist_matr = pl.ma.fix_invalid(angdist_matr, fill_value=pl.pi).data

        # all the distances are equally weighted and so far range in 0,1

        weight_eucl = 1.0 / self._distance_matr.max()
        weight_hav = 1.0 / angdist_matr.max()

        self._distance_matr = pl.multiply(
            weight_eucl * self._distance_matr, weight_hav * angdist_matr
        )

        self._X = pl.concatenate([self._X, Theta], axis=1)
        pass

    def estimate_affinity(self, affinity, file_affinity):

        if affinity == "euclidean":
            self._metric = DistanceMetric.get_metric("euclidean")
        elif affinity == "hellinger":
            self._metric = DistanceMetric.get_metric(hellinger_distance)
            assert (
                self._nfeat > 1
            ), f"can't estimate hellinger distance with {self._nfeat} \
                                        features, needed >1 features"
            try:
                self._distance_matr = pl.load(file_affinity)
                if self.verbose:
                    print(f"Loading affinity from {file_affinity}")
                return
            except FileNotFoundError:
                warnings.warn(
                    "Warning: Recomputing the Hellinger distance, this might take a while... "
                )
                self._metric = DistanceMetric.get_metric(hellinger_distance)
        else:
            raise ValueError(
                f" Affinity  '{affinity}' not recognized,"
                "  please use either  'euclidean' or 'hellinger' "
            )

        self._distance_matr = self._metric.pairwise(self._X)
        self._distance_matr = pl.ma.fix_invalid(
            self._distance_matr, fill_value=1.0
        ).data

    pass

    def __init__(
        self,
        features,
        nfeatures,
        nside=16,
        include_haversine=False,
        galactic_mask=None,
        affinity="euclidean",
        scaler=preprocessing.StandardScaler(),
        file_affinity="",
        verbose=False,
        save_affinity=False,
        feature_weights=None,
    ):
        """
        -features: list of features to cluster
        -nfeatures

        """
        self._nside = nside
        self._nfeat = nfeatures
        if galactic_mask is None:
            self.galactic_mask = np.bool_(pl.ones_like(features[0]))
        else:
            self.galactic_mask = galactic_mask
        features[0] = features[0][galactic_mask]
        features[1] = features[1][galactic_mask]

        if self._nfeat > 1:
            assert features[0].shape[0] == features[1].shape[0]
            self._npix = features[0].shape[0]  # hp.nside2npix(nside)
        else:
            self._npix = features.shape[0]
        if feature_weights is None:
            feature_weights = pl.ones(self._nfeat)
        self.verbose = verbose
        self._X = pl.zeros((self._npix, self._nfeat))

        if self._nfeat == 1:
            features = [features]
        for i, x in zip(range(self._nfeat), features):
            self._X[:, i] = x
        # Standard rescaling of all the features
        if scaler is not None:
            self._X = scaler.fit_transform(self._X)

        for i in range(self._nfeat):
            self._X[:, i] *= feature_weights[i]

        self.estimate_affinity(affinity, file_affinity)

        self._has_angles = False
        if include_haversine:
            self._has_angles = True
            self.estimate_haversine()

        self._Affinity = pairwise_kernels(self._distance_matr, metric="precomputed")

        if save_affinity:
            pl.save(file_affinity, self._Affinity)

    def intracluster_distance(self, K, labels):

        mu = pl.zeros(K * self._X.shape[1]).reshape(K, self._X.shape[1])
        MD = pl.zeros(K)
        for k in range(K):
            ck = pl.where(labels == k)[0]
            Xk = self._X[ck, :]
            mu[k] = Xk.mean(axis=0)

            Nk = len(ck)
            E = self._metric.pairwise(
                X=Xk[:, : self._nfeat], Y=mu[k, : self._nfeat].reshape(-1, self._nfeat)
            )

            E = pl.ma.fix_invalid(E, fill_value=1.0).data
            if self._has_angles and Nk > 1:
                H = self.haversine.pairwise(
                    X=Xk[:, self._nfeat :], Y=mu[k, self._nfeat :].reshape(-1, 2)
                )
                # H =pl.ma.fix_invalid(H,  fill_value=pl.pi).data
                E = pl.multiply(E, H)
            MD[k] = E.sum() / Nk
        return mu, MD

    def minimize_partition_measures(self):
        self.Vu = pl.zeros_like(self.Kvals).reshape(-1, 1) * 1.0
        self.Vo = pl.zeros_like(self.Kvals).reshape(-1, 1) * 1.0
        nvals = len(self.Kvals)
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")
            clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            clusters.fit_predict(self._Affinity)
            mu, MD = self.intracluster_distance(K, clusters.labels_)
            dmu = self._metric.pairwise(mu[:, : self._nfeat])
            dmu = pl.ma.fix_invalid(dmu, fill_value=1.0).data
            if self._has_angles:
                dmuang = self.haversine.pairwise(mu[:, self._nfeat :])
                # dmuang =pl.ma.fix_invalid(dmuang,  fill_value=pl.pi).data

                dmu = pl.multiply(dmu, dmuang)

            pl.fill_diagonal(dmu, pl.inf)
            self.Vo[j] = K / dmu.min()  # overpartition meas.
            self.Vu[j] = MD.sum() / K  # underpartition meas.

        # We have to match  Vo and Vu, we rescale Vo so that it ranges as Vu
        min_max_scaler = preprocessing.MinMaxScaler(
            feature_range=(self.Vu.min(), self.Vu.max())
        )

        self.Vu = min_max_scaler.fit_transform((self.Vu)).reshape(nvals)
        self.Vo = min_max_scaler.fit_transform((self.Vo)).reshape(nvals)
        # minimizing the squared sum

        Vsv = interp1d(
            self.Kvals, pl.sqrt(self.Vu ** 2 + self.Vo ** 2).T, kind="slinear"
        )

        Krange = pl.arange(self.Kvals.min(), self.Kvals.max())
        minval = pl.argmin(Vsv(Krange) - Vsv(Krange).min())

        Kopt = Krange[minval]
        return Kopt

    def minimize_residual_variances(self, **kwargs):
        """
        Syst. residuals behave as an underpartition measures, i.e. the larger is K the lower is  the residual( the more are patches  the better )
        stat. residuals behave as an overpartition measures, i.e. the larger is K the higher is the residuals (the lesser are the patches the worse, because you have less signal-to-noise in each patch)

        """
        self.Vu = pl.zeros_like(self.Kvals).reshape(-1, 1) * 1.0
        self.Vo = pl.zeros_like(self.Kvals).reshape(-1, 1) * 1.0
        nvals = len(self.Kvals)
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")
            clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            clusters.fit_predict(self._Affinity)
            msys, mstat = estimate_Stat_and_Sys_residuals(
                clusters.labels_, galactic_binmask=self.galactic_mask, **kwargs
            )
            m1 = pl.ma.masked_equal(msys[1], hp.UNSEEN).mask
            m2 = pl.ma.masked_equal(mstat[1], hp.UNSEEN).mask

            _, _, var_sys, var_stat = estimate_spectra(msys, mstat)

            self.Vo[j], self.Vu[j] = var_stat, var_sys

        # We have to match  Vo and Vu, we rescale Vo so that it ranges as Vu
        min_max_scaler = preprocessing.MinMaxScaler(
            feature_range=(self.Vu.min(), self.Vu.max())
        )
        self.Vu = min_max_scaler.fit_transform((self.Vu)).reshape(nvals)
        self.Vo = min_max_scaler.fit_transform((self.Vo)).reshape(nvals)

        Vsv = interp1d(
            self.Kvals, pl.sqrt(self.Vu ** 2 + self.Vo ** 2).T, kind="slinear"
        )
        Krange = pl.arange(self.Kvals.min(), self.Kvals.max())
        minval = pl.argmin(Vsv(Krange) - Vsv(Krange).min())
        Kopt = Krange[minval]

        return Kopt

    def get_WCSS(self, K, labels, distance_matr):
        MD = pl.zeros(K)
        for k in range(K):
            ck = pl.ma.masked_equal(labels, k).mask
            Xk = self._X[ck, :]
            Nk = self._X[ck, :].shape[0]
            E = distance_matr[pl.outer(ck, ck)]
            MD[k] = E.sum() / (2 * Nk)

        return MD.sum()

    def estimate_Gap_statistics(self, nrefs):
        masknans = pl.ma.masked_not_equal(self._X[:, 0], 0).mask
        minvals = self._X[masknans, :].min(axis=0)
        maxvals = self._X[masknans, :].max(axis=0)
        meanvals = self._X[masknans, :].mean(axis=0)
        stdvals = self._X[masknans, :].std(axis=0)
        ref_Affinity = []
        Dref = []

        # Compute a random uniform reference distribution of features
        # precompute Distances and affinities.
        for i in range(nrefs):

            random_X = pl.ones_like(self._X)
            # random_X [:,0 ] =np.random.uniform (low = minvals[0] , high=maxvals[0], size=pl.int_( self._X.shape[0]/10 ) )
            random_X[:, 1] = np.random.uniform(
                low=pl.quantile(q=0.16, a=self._X[masknans, 1]),
                high=pl.quantile(q=0.16, a=self._X[masknans, 1]),
                size=pl.int_(self._X.shape[0]),
            )
            random_X[:, 0] = np.random.normal(
                loc=meanvals[0], scale=stdvals[0], size=pl.int_(self._X.shape[0])
            )
            ref_D = self._metric.pairwise(random_X)
            ref_D = pl.ma.fix_invalid(ref_D, fill_value=1.0).data

            Dref.append(ref_D)

            ref_Affinity.append(pairwise_kernels(ref_D, metric="precomputed"))

        self.Gaps = pl.zeros(len(self.Kvals))
        self.sd = self.Gaps * 0.0
        self.W = self.Gaps * 0.0  # KL index
        p = self._nfeat
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")
            self.clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            self.clusters.fit_predict(self._Affinity)
            # estimate WCSS for the samples
            W = self.get_WCSS(K, self.clusters.labels_, self._distance_matr)
            self.W[j] = W
            # estimate WCSS for random samples
            ref_W = pl.zeros(nrefs)

            for i in range(nrefs):
                ref_clusters = AgglomerativeClustering(
                    n_clusters=K,
                    affinity="precomputed",
                    linkage="average",
                    connectivity=self.connectivity,
                )
                ref_clusters.fit_predict(ref_Affinity[i])
                ref_W[i] = self.get_WCSS(K, ref_clusters.labels_, Dref[i])

            self.sd[j] = np.std(np.log(ref_W)) * np.sqrt(1 + 1.0 / nrefs)
            self.Gaps[j] = np.mean(np.log(ref_W)) - np.log(W)

        ## see section 4 of Tibishrani et al. http://web.stanford.edu/~hastie/Papers/gap.pdf

        gaps_criterion = pl.array(
            [self.Kvals[:-1], self.Gaps[:-1] - self.Gaps[1:] + self.sd[1:]]
        )
        mask = pl.array(gaps_criterion[1, :] >= 0)
        return pl.int_(gaps_criterion[0, mask][0])

    def maximize_Krzanowski_Lai_index(self):
        # Krzanowski -Lai    index
        self.W = pl.zeros(len(self.Kvals))

        p = self._nfeat
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")

            self.clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            self.clusters.fit_predict(self._Affinity)
            # estimate WCSS for the samples
            self.W[j] = self.get_WCSS(K, self.clusters.labels_, self._distance_matr)
        # see eq. 3.1 of Krzanowski and Lai 1988 Biometrics
        DIFF = pl.array(
            [
                self.Kvals[1:],
                (self.W[:-1] * self.Kvals[:-1] ** (2 / p))
                - (self.W[1:] * self.Kvals[1:] ** (2 / p)),
            ]
        )

        # for k=1, KL index is undefined

        self.KL = pl.array(
            [self.Kvals[1:-1], pl.fabs(DIFF[1, :-1] / DIFF[1, 1:])]
        )  # see eq. 3.2
        maxindex = self.KL[1, :].argmax()

        return pl.int_(self.KL[0, maxindex])

    def __call__(
        self,
        K=None,
        nvals=10,
        Kmax=50,
        Kmin=2,
        minimize="partition",
        connectivity=None,
        **kwargs,
    ):

        self.connectivity = connectivity

        if K is not None:
            if self.verbose:
                print(f"Running Hierarchical clustering to find K={K} clusters.")
            self.clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            self.clusters.fit_predict(self._Affinity)
            return

        if self.verbose:
            print(
                "Running Hierarchical clustering "
                f"with several choices of K from {Kmin} to {Kmax} to find K_opt"
            )

        self.Kvals = pl.linspace(Kmin, Kmax, num=nvals, dtype=pl.int16)
        if minimize == "partition":
            self.K_optim = self.minimize_partition_measures()
        elif minimize == "residuals":
            self.K_optim = self.minimize_residual_variances(**kwargs)

        elif minimize == "gap":
            if not "nrefs" in kwargs.keys():
                kwargs["nrefs"] = 3

            self.K_optim = self.estimate_Gap_statistics(nrefs=kwargs["nrefs"])
        elif minimize == "KL":
            self.K_optim = self.maximize_Krzanowski_Lai_index()
        elif minimize == "silhouette":
            self.K_optim = self.maximize_silhouette_score()
        elif minimize == "DB":
            self.K_optim = self.minimize_davies_bouldin_score()
        elif minimize == "CH":
            self.K_optim = self.maximize_calinski_harabasz_score()

        if self.verbose:
            print(f"Optimal number of clusters: K_opt ={self.K_optim}")
        self.clusters = AgglomerativeClustering(
            n_clusters=self.K_optim,
            affinity="precomputed",
            linkage="average",
            connectivity=self.connectivity,
        )
        self.clusters.fit_predict(self._Affinity)

    def maximize_silhouette_score(self):
        self.silhouette = pl.zeros(len(self.Kvals))
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")
            clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            clusters.fit_predict(self._Affinity)
            self.silhouette[j] = silhouette_score(
                X=self._distance_matr, labels=clusters.labels_, metric="precomputed"
            )
        maxindex = self.silhouette.argmax()
        return pl.int_(self.Kvals[maxindex])

    def maximize_calinski_harabasz_score(self):
        self.CH = pl.zeros(len(self.Kvals))
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")
            clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            clusters.fit_predict(self._Affinity)
            self.CH[j] = calinski_harabasz_score(X=self._X, labels=clusters.labels_)
        return pl.int_(self.Kvals[self.CH.argmax()])

    def minimize_davies_bouldin_score(self):
        self.DB = pl.zeros(len(self.Kvals))
        for j, K in enumerate(self.Kvals):
            if self.verbose:
                print(f"Running with K={K} clusters")
            clusters = AgglomerativeClustering(
                n_clusters=K,
                affinity="precomputed",
                linkage="average",
                connectivity=self.connectivity,
            )
            clusters.fit_predict(self._Affinity)
            self.DB[j] = davies_bouldin_score(X=self._X, labels=clusters.labels_)
        return pl.int_(self.Kvals[self.DB.argmin()])
