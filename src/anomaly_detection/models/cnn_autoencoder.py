"""CPU convolutional autoencoder with normal-only sequence training.

Scores combine group means of scaled continuous MSE and binary cross entropy.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from ..data import CONTINUOUS_COLUMNS, SENSOR_COLUMNS


class ConvolutionalAutoencoder(nn.Module):
    """Compress in time with strided convolutions; reconstruct the exact length."""

    def __init__(self, n_features: int = 15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, n_features, kernel_size=3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(self.encoder(values))
        if reconstruction.shape[-1] != values.shape[-1]:
            reconstruction = functional.interpolate(
                reconstruction,
                size=values.shape[-1],
                mode="linear",
                align_corners=False,
            )
        return reconstruction


class SensorWindowDataset(Dataset):
    """Materialize only a minibatch of channel-first windows at a time."""

    def __init__(self, values: np.ndarray, endpoints: np.ndarray, length: int):
        self.values, self.endpoints, self.length = values, endpoints, length

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, index: int) -> torch.Tensor:
        end = self.endpoints[index]
        return torch.from_numpy(self.values[end - self.length + 1 : end + 1].T.copy())


class SequenceAutoencoder:
    """Fit normal sequences; early-stop on a separate normal development period."""

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        sequence_length: int = 60,
        seed: int = 42,
    ):
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 4
        ):
            raise ValueError("sequence_length must be an integer of at least four")
        self.parameters = dict(parameters or {})
        self.sequence_length, self.seed = sequence_length, seed
        self.feature_names = list(SENSOR_COLUMNS)
        self.scaler = StandardScaler()
        self.network: ConvolutionalAutoencoder | None = None
        self.history: list[dict[str, float | int]] = []
        self.binary_loss_weight = float(self.parameters.get("binary_loss_weight", 1.0))
        if not np.isfinite(self.binary_loss_weight) or self.binary_loss_weight <= 0:
            raise ValueError("binary_loss_weight must be finite and positive")

    def _values(self, features: pd.DataFrame, *, fitting: bool = False) -> np.ndarray:
        if (
            not isinstance(features, pd.DataFrame)
            or features.empty
            or features.columns.tolist() != self.feature_names
        ):
            raise ValueError("Supply sensor columns in their frozen training order")
        values = features.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("All sensor readings must be finite")
        if not np.isin(values[:, len(CONTINUOUS_COLUMNS) :], [0, 1]).all():
            raise ValueError("Digital channels must be binary")
        transformed = values.copy()
        numeric = values[:, : len(CONTINUOUS_COLUMNS)]
        transformed[:, : len(CONTINUOUS_COLUMNS)] = (
            self.scaler.fit_transform(numeric)
            if fitting
            else self.scaler.transform(numeric)
        )
        return transformed

    def _endpoints(self, endpoints: np.ndarray, row_count: int) -> np.ndarray:
        endpoints = np.asarray(endpoints)
        if (
            endpoints.ndim != 1
            or not np.issubdtype(endpoints.dtype, np.integer)
            or not len(endpoints)
        ):
            raise ValueError("At least one integer sequence endpoint is required")
        if (
            np.any(endpoints < self.sequence_length - 1)
            or np.any(endpoints >= row_count)
            or np.any(np.diff(endpoints) <= 0)
        ):
            raise ValueError(
                "Endpoints must be sorted, unique and contain complete windows"
            )
        return endpoints.astype(np.int64, copy=False)

    def _loader(
        self, values: np.ndarray, endpoints: np.ndarray, *, shuffle: bool = False
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            SensorWindowDataset(values, endpoints, self.sequence_length),
            batch_size=int(self.parameters.get("batch_size", 128)),
            shuffle=shuffle,
            generator=generator,
            num_workers=0,
        )

    def _feature_errors(
        self, reconstruction: torch.Tensor, actual: torch.Tensor
    ) -> torch.Tensor:
        continuous_count = len(CONTINUOUS_COLUMNS)
        numeric = (
            (reconstruction[:, :continuous_count] - actual[:, :continuous_count])
            .square()
            .mean(dim=2)
        )
        binary = functional.binary_cross_entropy_with_logits(
            reconstruction[:, continuous_count:],
            actual[:, continuous_count:],
            reduction="none",
        ).mean(dim=2)
        return torch.cat([numeric, binary], dim=1)

    def _scores(self, errors: torch.Tensor) -> torch.Tensor:
        count = len(CONTINUOUS_COLUMNS)
        return (
            errors[:, :count].mean(dim=1)
            + self.binary_loss_weight * errors[:, count:].mean(dim=1)
        ) / (1 + self.binary_loss_weight)

    @staticmethod
    def _sample_endpoints(endpoints: np.ndarray, maximum: int) -> np.ndarray:
        if maximum < 1:
            raise ValueError("Window limits must be positive")
        return (
            endpoints
            if len(endpoints) <= maximum
            else endpoints[np.linspace(0, len(endpoints) - 1, maximum, dtype=int)]
        )

    def fit(
        self,
        train_features: pd.DataFrame,
        train_endpoints: np.ndarray,
        development_features: pd.DataFrame,
        development_endpoints: np.ndarray,
    ) -> "SequenceAutoencoder":
        threads = int(self.parameters.get("threads", 2))
        epochs, patience = (
            int(self.parameters.get("epochs", 15)),
            int(self.parameters.get("patience", 3)),
        )
        stride = int(self.parameters.get("training_stride", 5))
        if (
            min(
                threads,
                epochs,
                patience,
                stride,
                int(self.parameters.get("batch_size", 128)),
            )
            < 1
        ):
            raise ValueError("Training counts and batch size must be positive")
        torch.set_num_threads(threads)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        training_values = self._values(train_features, fitting=True)
        development_values = self._values(development_features)
        training_endpoints = self._endpoints(train_endpoints, len(training_values))[
            ::stride
        ]
        training_endpoints = self._sample_endpoints(
            training_endpoints, int(self.parameters.get("max_training_windows", 6000))
        )
        development_endpoints = self._endpoints(
            development_endpoints, len(development_values)
        )[::stride]
        development_endpoints = self._sample_endpoints(
            development_endpoints,
            int(self.parameters.get("max_development_windows", 1500)),
        )
        self.training_windows, self.development_windows = (
            len(training_endpoints),
            len(development_endpoints),
        )
        training_loader = self._loader(
            training_values, training_endpoints, shuffle=True
        )
        development_loader = self._loader(development_values, development_endpoints)
        self.network = ConvolutionalAutoencoder(len(self.feature_names))
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.parameters.get("learning_rate", 0.001)),
        )
        best_loss, best_state, epochs_without_improvement = np.inf, None, 0
        self.history = []
        for epoch in range(1, epochs + 1):
            self.network.train()
            training_loss_sum, training_count = 0.0, 0
            for batch in training_loader:
                optimizer.zero_grad()
                loss = self._scores(
                    self._feature_errors(self.network(batch), batch)
                ).mean()
                loss.backward()
                optimizer.step()
                training_loss_sum += float(loss.detach()) * len(batch)
                training_count += len(batch)
            self.network.eval()
            development_loss_sum, development_count = 0.0, 0
            with torch.inference_mode():
                for batch in development_loader:
                    loss = self._scores(
                        self._feature_errors(self.network(batch), batch)
                    ).mean()
                    development_loss_sum += float(loss) * len(batch)
                    development_count += len(batch)
            development_loss = development_loss_sum / development_count
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": training_loss_sum / training_count,
                    "development_loss": development_loss,
                }
            )
            if not np.isfinite(development_loss):
                raise ValueError("Autoencoder loss became non-finite")
            if development_loss < best_loss - 1e-7:
                best_loss, best_state, epochs_without_improvement = (
                    development_loss,
                    deepcopy(self.network.state_dict()),
                    0,
                )
                self.best_epoch = epoch
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Training did not produce a valid model")
        self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def reconstruction_errors(
        self, features: pd.DataFrame, endpoints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score every supplied endpoint; return weighted scores and channel errors."""
        if self.network is None:
            raise RuntimeError("Fit the autoencoder before inference")
        values = self._values(features)
        endpoints = self._endpoints(endpoints, len(values))
        scores, per_feature = [], []
        self.network.eval()
        with torch.inference_mode():
            for batch in self._loader(values, endpoints):
                errors = self._feature_errors(self.network(batch), batch)
                scores.append(self._scores(errors).numpy())
                per_feature.append(errors.numpy())
        return np.concatenate(scores), np.concatenate(per_feature)

    def score(self, features: pd.DataFrame, endpoints: np.ndarray) -> np.ndarray:
        return self.reconstruction_errors(features, endpoints)[0]

    def reconstruct(self, window: np.ndarray) -> np.ndarray:
        """Reconstruct one [time,feature] sensor window in original sensor units."""
        if self.network is None:
            raise RuntimeError("Fit the autoencoder before inference")
        if np.asarray(window).shape != (self.sequence_length, len(self.feature_names)):
            raise ValueError("Window must have [sequence_length,n_sensors] shape")
        frame = pd.DataFrame(window, columns=self.feature_names)
        values = torch.from_numpy(self._values(frame).T.copy()).unsqueeze(0)
        with torch.inference_mode():
            prediction = self.network(values)[0].T.numpy()
        count = len(CONTINUOUS_COLUMNS)
        prediction[:, :count] = self.scaler.inverse_transform(prediction[:, :count])
        prediction[:, count:] = torch.sigmoid(
            torch.from_numpy(prediction[:, count:])
        ).numpy()
        return prediction
