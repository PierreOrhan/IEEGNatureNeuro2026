import numpy as np
import typing as tp
from pydantic import BaseModel, Field
import warnings
from typing import List, Optional
import torch
from .api import AbstractBuffer
import tqdm

# ---------------------------------------------------------------------------
# Lazy shifted subspace  (compatible with BandedRidgeOperatorLazy)
# ---------------------------------------------------------------------------

class LazyTimeshiftSubspace:
    """A single time-shifted view of a feature matrix.

    No shifted array is ever allocated.  During ``load()`` only the rows that
    are actually needed are gathered and transferred to the compute device.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        The **original** (unshifted) feature array.  Kept on CPU as a numpy
        array so it is shared (zero-copy) across all lag objects.
    shift : int
        Signed sample offset.  Positive = future tap, negative = past tap.
        Row ``t`` of the output corresponds to row ``t - shift`` of features
        (with NaN padding where that index is out of range).
    total_T : int
        Number of time-steps in the full recording (= output length).
    device : torch.device
        Compute device the tensor is moved to during ``load()``.
    dtype : torch.dtype
        Target dtype (default float32).
    """

    def __init__(
        self,
        features: np.ndarray,
        shift: int,
        total_T: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        row_indices: Optional[np.ndarray] = None,
    ):
        self._features  = features          # shared, never copied
        self._shift     = shift
        self._total_T   = total_T
        self.device     = device
        self._dtype     = dtype
        # row_indices: if not None, we only expose a subset of rows (train/val
        # split).  This is set by __getitem__.
        self._row_indices = row_indices

        p = features.shape[1]
        n_rows = len(row_indices) if row_indices is not None else total_T
        self.shape = (n_rows, p)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Slicing support — returns a new lazy view, no data copied
    # ------------------------------------------------------------------
    def __getitem__(self, idx) -> "LazyTimeshiftSubspace":
        """Return a lazy view restricted to the given row indices."""
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        elif isinstance(idx, slice):
            idx = np.arange(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            idx = np.asarray(idx)
        elif isinstance(idx, int):
            idx = np.asarray([idx])
        # Compose with existing row selection
        if self._row_indices is not None:
            idx = self._row_indices[idx]
        return LazyTimeshiftSubspace(
            self._features, self._shift, self._total_T,
            self.device, self._dtype, row_indices=idx,
        )

    # ------------------------------------------------------------------
    # Core: materialise exactly the rows we need, then transfer to device
    # ------------------------------------------------------------------
    def load(self) -> torch.Tensor:
        """Return a (n_rows, p) float tensor on ``self.device``.

        Only the required source rows are read from the numpy array;
        out-of-range positions are filled with 0.0 (NaNs in features are
        replaced with 0 to avoid polluting gradients).
        """
        n_src, p  = self._features.shape
        shift     = self._shift

        if self._row_indices is not None:
            out_rows   = self._row_indices                        # which output rows
        else:
            out_rows   = np.arange(self._total_T)

        n_out = len(out_rows)
        buf   = np.zeros((n_out, p), dtype=np.float32)

        # Source row for output row i:  src = out_rows[i] - shift
        src_rows = out_rows - shift                               # signed
        valid    = (src_rows >= 0) & (src_rows < n_src)

        if valid.any():
            buf[valid] = self._features[src_rows[valid]]
            # Replace NaNs inherited from the original feature array
            np.nan_to_num(buf, nan=0.0, copy=False)

        t = torch.from_numpy(buf)                                 # zero-copy when possible
        t = t.pin_memory()                                        # fast CPU->GPU transfer
        return t.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# Factory: replaces generate_timeshift_features / generate_reverse_timeshift
# ---------------------------------------------------------------------------

def make_lazy_timeshifts(
    features: np.ndarray,
    nb_buffer: int,
    period: float,
    prediction_start: float,
    frequency: int,
    duration: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    reverse: bool = False,
) -> List[LazyTimeshiftSubspace]:
    """Return ``nb_buffer`` lazy subspaces, one per lag.

    Parameters mirror ``generate_timeshift_features`` /
    ``generate_reverse_timeshift_features`` but **no shifted array is ever
    allocated**.

    Parameters
    ----------
    features        : (n_samples, n_features) float32 numpy array
    nb_buffer       : number of lags
    period          : lag step in seconds
    prediction_start: first lag offset in seconds
    frequency       : samples per second
    duration        : total recording duration in seconds
    device          : torch device for ``load()``
    reverse         : if True, shifts are negative (past taps)
    """
    total_T = int(np.ceil(duration * frequency))
    subspaces = []
    for i in range(nb_buffer):
        offset_s = prediction_start + i * period
        shift    = int(offset_s * frequency)
        if reverse:
            shift = -shift
        subspaces.append(
            LazyTimeshiftSubspace(features, shift, total_T, device, dtype)
        )
    return subspaces


# ---------------------------------------------------------------------------
# Existing helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def generate_timeshift_features(
    features: np.ndarray,
    t_current: tp.List[float],
    period: float,
    nb_buffer: int,
    prediction_start: float,
    prediction_end: float,
    frequency: int,
    duration: float,
) -> tp.List[np.ndarray]:
    """
        Timeshift a features that is allready a timecoursed features.

        features : np.ndarray (n_samples, n_features)
        t_current : not used
        period : not used
        prediction_start: phase
        prediction_end: phase + duration = phase + period*nb_buffer
        nb_buffer: number of subspace in the buffer
        duration: total duration of the recording, in s
    """
    
    shifted_features = [
        np.zeros((int(np.ceil(duration * frequency)), features.shape[1]), dtype=np.float32) + np.nan
        for _ in range(nb_buffer)
    ]
    for i in tqdm.tqdm(range(nb_buffer)):
        start_wbuffer = int((prediction_start + i * period) * frequency) + np.arange(features.shape[0])
        ok_shift = start_wbuffer[(start_wbuffer >= 0) & (start_wbuffer < shifted_features[i].shape[0])]
        shifted_features[i][ok_shift, :] = features[: len(ok_shift), :]
    return shifted_features


def generate_reverse_timeshift_features(
    features: np.ndarray,
    t_current: tp.List[float],
    period: float,
    nb_buffer: int,
    prediction_start: float,
    prediction_end: float,
    frequency: int,
    duration: float,
) -> tp.List[np.ndarray]:
    """
        Timeshift a features that is allready a timecoursed features.
        features : np.ndarray (n_samples, n_features)
        t_current : not used
        period : not used
        prediction_start: phase
        prediction_end: phase + duration = phase + period*nb_buffer
        nb_buffer: number of subspace in the buffer
        duration: total duration of the recording, in s
    """
    shifted_features = [
        np.zeros((int(np.ceil(duration * frequency)), features.shape[1]), dtype=np.float32) + np.nan
        for _ in range(nb_buffer)
    ]
    for i in range(nb_buffer):
        start_wbuffer = int((prediction_start - i * period) * frequency) + np.arange(features.shape[0])
        ok_shift = start_wbuffer[(start_wbuffer >= 0) & (start_wbuffer < shifted_features[i].shape[0])]
        shifted_features[i][ok_shift, :] = features[len(features) - len(ok_shift) :, :]
    return shifted_features


# ---------------------------------------------------------------------------
# Buffer class — lazy variant
# ---------------------------------------------------------------------------

class SymTimeShift_sharedreg_lazy(AbstractBuffer):
    """Symmetric Time Shift — lazy version.

    Returns a list of ``LazyTimeshiftSubspace`` objects instead of a single
    concatenated array.  Each subspace is one lag and is moved to the GPU
    only when ``load()`` is called during the LSMR matvec/rmatvec passes.

    Memory cost: O(n_samples × n_features) for the shared base array instead
    of O(nb_buffer × total_T × n_features) for the pre-shifted version.
    """

    nb_buffer: int  = Field(..., description="Number of subspaces per direction")
    period: float   = Field(..., description="Lag step in seconds")
    device: str     = Field("cuda", description="Torch device string, e.g. 'cuda' or 'cpu'")

    def __call__(
        self,
        features: np.ndarray,
        t_current: tp.List[float],
        context: tp.List[tp.Any],
    ) -> List[LazyTimeshiftSubspace]:
        dev = torch.device(self.device)

        past_subspaces = make_lazy_timeshifts(
            features,
            nb_buffer        = self.nb_buffer,
            period           = self.period,
            prediction_start = self.prediction_start,
            frequency        = self.frequency,
            duration         = self.duration,
            device           = dev,
            reverse          = True,
        )
        future_subspaces = make_lazy_timeshifts(
            features,
            nb_buffer        = self.nb_buffer,
            period           = self.period,
            prediction_start = self.prediction_start,
            frequency        = self.frequency,
            duration         = self.duration,
            device           = dev,
            reverse          = False,
        )
        # One subspace per lag — the ridge solver consumes them one at a time
        return past_subspaces + future_subspaces


    @property
    def number_of_subspaces(self):
        return 2*self.nb_buffer

class SymTimeShift_sharedreg(AbstractBuffer):
    """
        Symmetric Time Shift features.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    period: float = Field(..., description="Duration of the period in s; " \
    "this is the time relative to the word during which the subspace is supposed to predict " \
    "the neural activity before switching to the next subspace.")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        past_tap_features = generate_reverse_timeshift_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)  
        future_tap_features = generate_timeshift_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)
        out_features = np.concatenate(past_tap_features + future_tap_features, axis=1)
        return [out_features]
    
    @property
    def number_of_subspaces(self):
        return 1