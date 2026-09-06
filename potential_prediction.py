import os
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class PotentialTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pca_n=45, n_radial_bins=72):
        self.pca_n = pca_n
        self.n_radial_bins = n_radial_bins
        self.pca = PCA(n_components=pca_n, random_state=42)

    def _center_potential(self, matrix):
        mask = matrix < 10
        if mask.sum() == 0:
            return matrix
        coords = np.argwhere(mask)
        cr = coords[:, 0].mean()
        cc = coords[:, 1].mean()
        shift_r = int(128 - cr)
        shift_c = int(128 - cc)
        return np.roll(np.roll(matrix, shift_r, axis=0), shift_c, axis=1)

    def fit(self, x, y=None):
        centered = np.array([self._center_potential(m.astype(np.float64)) for m in x])
        X_flat = centered.reshape(len(centered), -1)
        self.pca.fit(X_flat)
        return self

    def transform(self, x):
        centered = np.array([self._center_potential(m.astype(np.float64)) for m in x])
        X_flat = centered.reshape(len(centered), -1)
        X_pca = self.pca.transform(X_flat)

        mean = centered.mean(axis=(1, 2), keepdims=True)
        std = centered.std(axis=(1, 2), keepdims=True) + 1e-8
        x_norm = (centered - mean) / std
        fft_abs = np.abs(np.fft.fft2(x_norm))
        low_fft = fft_abs[:, :26, :26].reshape(len(x), -1)

        rows, cols = np.indices((256, 256))
        r = np.sqrt((rows - 128) ** 2 + (cols - 128) ** 2)
        max_r = 128 * np.sqrt(2)

        result = []
        for i in range(len(x)):
            c = centered[i]
            data = x[i].astype(np.float64)
            feats = [X_pca[i], low_fft[i]]

            for bs in [8, 16, 32, 64]:
                nb = 256 // bs
                blocked = c.reshape(nb, bs, nb, bs).mean(axis=(1, 3)).flatten()
                feats.append(blocked)

            feats.append(c.mean(axis=1))
            feats.append(c.mean(axis=0))
            feats.append(c.std(axis=1))
            feats.append(c.std(axis=0))

            n_bins = self.n_radial_bins
            r_mean = np.zeros(n_bins)
            r_std = np.zeros(n_bins)
            r_max = np.zeros(n_bins)
            for b in range(n_bins):
                lo = b * max_r / n_bins
                hi = (b + 1) * max_r / n_bins
                ring = (r >= lo) & (r < hi)
                if ring.sum() > 5:
                    vals = c[ring]
                    r_mean[b] = vals.mean()
                    r_std[b] = vals.std()
                    r_max[b] = vals.max()
            feats.extend([r_mean, r_std, r_max])

            flat = c.flatten()
            mask = data < 10
            if mask.sum() > 0:
                pot_vals = data[mask]
                gx = np.abs(np.diff(c, axis=1, append=c[:, -1:]))
                gy = np.abs(np.diff(c, axis=0, append=c[-1:, :]))
                grad = np.sqrt(gx**2 + gy**2)
                stats = [
                    data.mean(),
                    data.std(),
                    data.min(),
                    data.max(),
                    data.sum(),
                    pot_vals.mean(),
                    pot_vals.std(),
                    pot_vals.min(),
                    ((flat - flat.mean()) ** 3).mean() / (flat.std() ** 3 + 1e-8),
                    ((flat - flat.mean()) ** 4).mean() / (flat.std() ** 4 + 1e-8) - 3,
                    grad.mean(),
                    grad.std(),
                    grad.max(),
                    np.sum(c * r**2) / np.sum(c) if np.sum(c) != 0 else 0,
                ]
            else:
                stats = [20.0] * 14
            feats.append(np.array(stats))

            result.append(np.concatenate(feats))

        return np.array(result, dtype=np.float32)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)


def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)

    regressor = Pipeline(
        [
            ("vectorizer", PotentialTransformer(pca_n=45, n_radial_bins=72)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=680,
                    max_features=0.23,
                    min_samples_leaf=1,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: float(value) for file, value in zip(test_files, predictions)}
