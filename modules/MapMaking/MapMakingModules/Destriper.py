from __future__ import annotations
import numpy as np
from mpi4py import MPI
import logging
from scipy.sparse.linalg import LinearOperator
from modules.utils import bin_funcs

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------- small MPI helpers ----------

def _dot_global32(x_f32: np.ndarray, y_f32: np.ndarray) -> float:
    """Global dot in float64 while inputs stay float32."""
    loc = float(np.dot(x_f32.astype(np.float64, copy=False),
                       y_f32.astype(np.float64, copy=False)))
    return comm.allreduce(loc, op=MPI.SUM)

def _norm_global32(x_f32: np.ndarray) -> float:
    return np.sqrt(_dot_global32(x_f32, x_f32))

def sum_map_all_inplace(m: np.ndarray) -> np.ndarray:
    """Allreduce sum over processes, in-place. Handles float32/64."""
    if m.dtype == np.float64:
        mpi_type = MPI.DOUBLE
    else:
        mpi_type = MPI.FLOAT
    comm.Allreduce(MPI.IN_PLACE, [m, mpi_type], op=MPI.SUM)
    return m

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------- preconditioner ----------

def _build_diag_precond_f32(data_obj, lambda_ridge=0.0) -> np.ndarray:
    """
    Very cheap diagonal (Jacobi) preconditioner ~ per-offset weight sum.
    Returns M_inv in float32 with safe floor.
    """
    w = data_obj.data.weights.astype(np.float32, copy=False)
    L = int(data_obj.data.offset_length)
    ns = w.size
    n_off = (ns + L - 1) // L  # ceil

    # Sum weights per offset chunk (no extra allocations beyond one small array)
    wsum = np.zeros(n_off, dtype=np.float32)
    # fast chunked accumulate
    # (np.add.reduceat is concise, but explicit loop keeps peak memory low)
    off_idx = np.arange(ns, dtype=np.int64) // L
    np.add.at(wsum, off_idx, w)

    # Make invertible
    eps = np.float32(1e-12)
    M_inv = 1.0 / np.maximum(wsum + np.float32(lambda_ridge), eps)
    return M_inv

# ---------- offsets -> offset map (float32) ----------

def _offsets_to_map_f32(offsets_f32: np.ndarray, data_obj) -> np.ndarray:
    """
    Project 1D offsets back to a per-pixel offset map (float32).
    """
    sum_map = np.zeros_like(data_obj.data.sum_map, dtype=np.float32)
    wei_map = np.zeros_like(data_obj.data.weight_map, dtype=np.float32)
    off_map = np.zeros_like(data_obj.data.sum_map, dtype=np.float32)

    bin_funcs.bin_offset_to_map(offsets_f32, off_map, sum_map, wei_map,
                                data_obj.data.pixels, data_obj.data.weights, data_obj.data.offset_length)

    # global reductions
    sum_map = sum_map_all_inplace(sum_map)
    wei_map = sum_map_all_inplace(wei_map)

    mask = wei_map > 0
    out = np.full_like(off_map, np.nan, dtype=np.float32)
    out[mask] = sum_map[mask] / wei_map[mask]
    return out

# ---------- LinearOperator wrapper ----------

def _make_linear_operator(ax) -> LinearOperator:
    """
    Wrap AxCOMAP into a SciPy LinearOperator with float32 I/O.
    """
    n = ax.n_offsets
    def mv(v):
        # v is float32 vector of offsets; Ax returns float32 residuals per offset
        return ax(v)
    return LinearOperator((n, n), matvec=mv, dtype=np.float32)

# ---------- the data class ------------------

def _laplacian_first(v: np.ndarray, out: np.ndarray):
    """out = D^T D v  (first-difference smoothness, penalises jumps)."""
    out.fill(0.0)
    n = v.size
    if n == 0: return
    if n == 1: 
        out[0] = 0.0
        return
    out[0]      = -v[0] + v[1]
    out[1:-1]   = -2.0*v[1:-1] + v[:-2] + v[2:]
    out[-1]     = -v[-1] + v[-2]

def _laplacian_second(v: np.ndarray, out: np.ndarray):
    """out = (D^2)^T (D^2) v  (second-difference smoothness, penalises ramps)."""
    _laplacian_first(v, out)             # out = L1 v
    tmp = np.empty_like(v)
    _laplacian_first(out, tmp)           # tmp = L1(out) = L1(L1 v)
    out[:] = tmp


class AxCOMAP:
    """
    Linear operator wrapper for destriping:
      v (offsets) -> A v  where A ≈ F^T N^{-1} Z F

    Assumes `data_object` exposes:
      - pixels:   (nsamples,) int64/int32 pixel indices (-1 for invalid)
      - weights:  (nsamples,) float32/64 inverse-noise weights per sample
      - offset_length: int (samples per offset chunk)
      - rhs:      (n_offsets,) float32 right-hand side b = F^T N^{-1} Z d
      - sum_map, weight_map: (npix,) float32 working map shapes (for sizing)
    """
    def __init__(self, data_object,
                 lambda_ridge: float = 0.0,
                 alpha_grad:   float = 0.0,
                 alpha_curv:   float = 0.0):
        self.data_object  = data_object
        self.lambda_ridge = float(lambda_ridge)
        self.alpha_grad   = float(alpha_grad)
        self.alpha_curv   = float(alpha_curv)

        # Shapes / derived sizes
        self.nsamples = int(self.data_object.data.pixels.size)
        self.npix     = int(self.data_object.data.sum_map.size)
        self.L        = int(self.data_object.data.offset_length)
        self._n_offsets = _ceil_div(self.nsamples, self.L)

        # Validate rhs length if present
        if hasattr(self.data_object.data, "rhs"):
            if self.data_object.data.rhs.size != self._n_offsets:
                # Be explicit — a silent mismatch here wrecks CG
                raise ValueError(
                    f"rhs length ({self.data_object.data.rhs.size}) "
                    f"!= n_offsets ({self._n_offsets}); "
                    "ensure rhs was built with ceil(nsamples/offset_length)."
                )

        # Work buffers (float32 to honor memory budget)
        self._sum_map = np.zeros(self.npix, dtype=np.float32)
        self._wei_map = np.zeros(self.npix, dtype=np.float32)
        self._sky_map = np.zeros(self.npix, dtype=np.float32)
        self._Ax      = np.zeros(self._n_offsets, dtype=np.float32)
        self._reg_buf = np.zeros(self._n_offsets, dtype=np.float32)  

        # Handy direct refs (avoid attribute lookups in hot path)
        self._pix = self.data_object.data.pixels          # (nsamples,)
        self._wts = self.data_object.data.weights         # (nsamples,)

        # Sum weights to scale priors 
        w = self.data_object.data.weights.astype(np.float32, copy=False)
        ns = w.size
        wsum = np.zeros(self.n_offsets, dtype=np.float32)
        off_idx = np.arange(ns, dtype=np.int64) // self.L
        np.add.at(wsum, off_idx, w)

        self.lambda_ridge *= np.median(wsum)
        self.alpha_grad   *= np.median(wsum)
        self.alpha_curv   *= np.median(wsum)


    @property
    def n_offsets(self) -> int:
        return self._n_offsets

    def matvec(self, offsets: np.ndarray) -> np.ndarray:
        """
        Compute A * offsets (per-offset residuals), float32 in/out.
        Offsets length must be n_offsets (ceil division).
        """
        if offsets.dtype != np.float32:
            offsets = offsets.astype(np.float32, copy=False)
        if offsets.size != self._n_offsets:
            raise ValueError(f"offsets length {offsets.size} != n_offsets {self._n_offsets}")

        # zero work buffers
        self._sum_map.fill(0.0)
        self._wei_map.fill(0.0)
        self._sky_map.fill(0.0)
        self._Ax.fill(0.0)

        # 1) Bin offsets -> sky_map (sum/weight), local
        bin_funcs.bin_offset_to_map(offsets,
                                    self._sky_map,   # filled after division inside kernel
                                    self._sum_map,
                                    self._wei_map,
                                    self._pix,
                                    self._wts,
                                    self.L)

        # 2) Globalize map sums/weights, then recompute sky_map = sum/wei
        sum_map_all_inplace(self._sum_map)
        sum_map_all_inplace(self._wei_map)
        mask = self._wei_map > 0.0
        # keep defined everywhere (zeros where no weight)
        self._sky_map[~mask] = 0.0
        self._sky_map[mask]  = self._sum_map[mask] / self._wei_map[mask]

        # 3) Bin back to offsets -> residuals (A * offsets)
        bin_funcs.bin_offset_to_rhs(self._Ax,
                                    offsets,
                                    self._sky_map,
                                    self._pix,
                                    self._wts,
                                    self.L)

        if self.alpha_grad > 0.0:
            _laplacian_first(offsets, self._reg_buf)
            self._Ax += np.float32(self.alpha_grad) * self._reg_buf

        if self.alpha_curv > 0.0:
            _laplacian_second(offsets, self._reg_buf)
            self._Ax += np.float32(self.alpha_curv) * self._reg_buf

        if self.lambda_ridge > 0.0:
            self._Ax += np.float32(self.lambda_ridge) * offsets

        return self._Ax

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.matvec(v)

# ---------- the solver (PCG in float32 with f64 scalars) ----------

def cgm(Ax,
        threshold: float = 1e-6,
        niter: int = 1000,
        verbose: bool = True,
        x0: np.ndarray | None = None,
        return_offset_map: bool = True):
    """
    Memory-lean PCG for destriping: solve (F^T N^-1 Z F) a = b.
    - Big arrays remain float32 (fits your memory constraints).
    - All MPI dot products run in float64 to stabilize convergence.

    Returns:
        offset_map (float32) if return_offset_map else 1D offsets (float32)
    """
    # Build operator
    Aop = _make_linear_operator(Ax)

    # Right-hand side in float32
    b = Ax.data_object.data.rhs.astype(np.float32, copy=False)
    b_norm = max(1.0, _norm_global32(b))

    # Preconditioner (diag) in float32
    M_inv = _build_diag_precond_f32(Ax.data_object, Ax.lambda_ridge)

    # Initial guess
    if x0 is None:
        x = np.zeros(Ax.n_offsets, dtype=np.float32)
    else:
        x = x0.astype(np.float32, copy=True)

    # r = b - A x
    r = b - Aop.matvec(x)
    z = M_inv * r
    p = z.copy()

    rz_old = _dot_global32(r, z)
    r_norm = _norm_global32(r)
    if rank == 0:
        logging.info(f"[PCG32] start relres={r_norm/b_norm:.3e} tol={threshold:.1e} maxiter={niter}")

    for k in range(niter):
        Ap = Aop.matvec(p)

        denom = _dot_global32(p, Ap)
        if denom == 0.0 or not np.isfinite(denom):
            if rank == 0:
                logging.warning("[PCG32] breakdown: denom=0 or NaN")
            break

        alpha = rz_old / denom  # float64 scalar
        # x += alpha * p ; r -= alpha * Ap   (keep in float32)
        x += np.float32(alpha) * p
        r -= np.float32(alpha) * Ap

        # convergence check
        r_norm = _norm_global32(r)
        relres = r_norm / b_norm
        if verbose and (k % 10 == 0) and rank == 0:
            logging.info(f"[PCG32] iter={k:4d} relres={relres:.3e}")
        if relres < threshold:
            if rank == 0:
                logging.info(f"[PCG32] converged in {k+1} iters, relres={relres:.3e}")
            break

        z = M_inv * r
        rz_new = _dot_global32(r, z)
        if rz_old == 0.0 or not np.isfinite(rz_new):
            if rank == 0:
                logging.warning("[PCG32] breakdown: rz_old=0 or rz_new NaN")
            break

        beta = rz_new / rz_old
        p = z + np.float32(beta) * p
        rz_old = rz_new

    # Either return the offset vector, or the offset map 
    if return_offset_map:
        return _offsets_to_map_f32(x, Ax.data_object)
    else:
        return x

