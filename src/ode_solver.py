"""Autonomous full-spectrum ODE dynamics for the shallow autoencoder.

The equations are adapted from the full-rank analytical updates in
https://github.com/mariaref/NonLinearShallowAE.  A PyTorch model is used only
to construct the initial spectral factors; all subsequent evolution is
independent of the model and of SGD.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Callable

import numpy as np
from scipy.integrate import RK45

from src.autoencoder import AutoEncoder
from src.integrals.integrals import Integral


MIN_ODE_ALPHA = 1.0
MAX_RK45_ALPHA_STEP = 1000.0
FACTOR_NORM_FLOOR = 0


@dataclass(frozen=True)
class ODEState:
    q: np.ndarray
    r: np.ndarray
    t: np.ndarray
    T0: np.ndarray


@dataclass(frozen=True)
class ODESample:
    alpha: float
    generalization_error: float


@dataclass(frozen=True)
class ODEResult:
    final_state: ODEState
    final_alpha: float
    nfev: int
    accepted_steps: int
    emitted_samples: int
    max_step: float
    sample_step: float
    rtol: float
    atol: float


SampleCallback = Callable[[ODESample], None]


class _CachedIntegral:
    """Cache moment evaluations within one state evaluation."""

    def __init__(self, integral: Integral):
        self.integral = integral
        self.cache: dict[tuple[str, tuple[int, ...], bytes], float] = {}

    def _evaluate(self, name: str, covariance: np.ndarray) -> float:
        covariance = np.ascontiguousarray(covariance, dtype=np.float64)
        key = (name, covariance.shape, covariance.tobytes())
        if key not in self.cache:
            self.cache[key] = float(getattr(self.integral, name)(covariance))
        return self.cache[key]

    def I2(self, covariance: np.ndarray) -> float:
        return self._evaluate("I2", covariance)

    def J2(self, covariance: np.ndarray) -> float:
        return self._evaluate("J2", covariance)

    def I21(self, covariance: np.ndarray) -> float:
        return self._evaluate("I21", covariance)

    def I22(self, covariance: np.ndarray) -> float:
        return self._evaluate("I22", covariance)

    def I3(self, covariance: np.ndarray) -> float:
        return self._evaluate("I3", covariance)


def factor_vector_from_model(
    model: AutoEncoder,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Convert one initialized untied model into spectral factors."""

    if model.tied:
        raise NotImplementedError("ODE mode currently requires tied: false")

    W = model.encoder.weight.detach().cpu().numpy().astype(np.float64, copy=True)
    V = (
        model.decoder.weight.detach()
        .cpu()
        .numpy()
        .T.astype(np.float64, copy=True)
    )
    K, D = W.shape
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    if eigenvectors.shape != (D, D):
        raise ValueError(
            f"Expected eigenvectors with shape {(D, D)}, got {eigenvectors.shape}"
        )
    if V.shape != (K, D):
        raise ValueError(f"Expected decoder weights with shape {(K, D)}, got {V.shape}")

    encoder_factors = (W @ eigenvectors).T
    decoder_factors = (V @ eigenvectors).T / np.sqrt(D)
    factors = np.concatenate((encoder_factors, decoder_factors), axis=1)
    if not np.all(np.isfinite(factors)):
        raise FloatingPointError("Initial factor vector contains a non-finite value")
    return np.ascontiguousarray(factors.reshape(-1))


def _factor_matrix(vector: np.ndarray, K: int, D: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    expected_size = 2 * K * D
    if vector.ndim != 1 or vector.size != expected_size:
        raise ValueError(
            f"Expected a flat factor state of size {expected_size}, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError("Factor state contains a non-finite value")
    return vector.reshape(D, 2 * K)


def _state_from_factor_matrix(factors: np.ndarray, K: int) -> ODEState:
    encoder_factors = factors[:, :K]
    decoder_factors = factors[:, K:]
    D = factors.shape[0]

    # G_m = z_m z_m^T with z_m = [a_m; b_m].
    q = np.empty((K, K, D), dtype=np.float64)
    r = np.empty((K, K, D), dtype=np.float64)
    t = np.empty((K, K, D), dtype=np.float64)
    np.einsum("dk,dl->kld", encoder_factors, encoder_factors, out=q)
    np.einsum("dk,dl->kld", decoder_factors, encoder_factors, out=r)
    np.einsum("dk,dl->kld", decoder_factors, decoder_factors, out=t)
    return ODEState(q=q, r=r, t=t, T0=np.sum(t, axis=2))


def state_from_factor_vector(vector: np.ndarray, K: int, D: int) -> ODEState:
    """Reconstruct all density blocks from a flat factor state."""

    return _state_from_factor_matrix(_factor_matrix(vector, K, D), K)


def initial_state_from_model(
    model: AutoEncoder,
    eigenvectors: np.ndarray,
) -> ODEState:
    """Convert one initialized untied model into full-spectrum densities."""

    K = model.hidden_size
    D = model.input_size
    return state_from_factor_vector(
        factor_vector_from_model(model, eigenvectors), K, D
    )


def pack_state(state: ODEState) -> np.ndarray:
    """Flatten an ODE state in the order q, r, t, T0."""

    return np.concatenate(
        (
            np.asarray(state.q, dtype=np.float64).ravel(),
            np.asarray(state.r, dtype=np.float64).ravel(),
            np.asarray(state.t, dtype=np.float64).ravel(),
            np.asarray(state.T0, dtype=np.float64).ravel(),
        )
    )


def unpack_state(
    vector: np.ndarray,
    K: int,
    D: int,
    *,
    copy: bool = False,
) -> ODEState:
    """Restore an ODE state from ``pack_state`` output."""

    vector = np.asarray(vector, dtype=np.float64)
    density_size = K * K * D
    matrix_size = K * K
    expected_size = 3 * density_size + matrix_size
    if vector.ndim != 1 or vector.size != expected_size:
        raise ValueError(
            f"Expected a flat ODE state of size {expected_size}, got {vector.shape}"
        )

    q = vector[:density_size].reshape(K, K, D)
    r = vector[density_size : 2 * density_size].reshape(K, K, D)
    t = vector[2 * density_size : 3 * density_size].reshape(K, K, D)
    T0 = vector[3 * density_size :].reshape(K, K)
    if copy:
        q, r, t, T0 = q.copy(), r.copy(), t.copy(), T0.copy()
    return ODEState(q=q, r=r, t=t, T0=T0)


def order_parameters_from_state(
    state: ODEState,
    eigenvalues: np.ndarray,
    D: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q1, R1 and T1 from the spectral density state."""

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.shape != (D,):
        raise ValueError(f"Expected {D} eigenvalues, got {eigenvalues.shape}")
    Q1 = np.einsum("kld,d->kl", state.q, eigenvalues) / D
    R1 = np.einsum("kld,d->kl", state.r, eigenvalues) / D
    T1 = np.einsum("kld,d->kl", state.t, eigenvalues) / D
    return Q1, R1, T1


def generalization_error_from_state(
    state: ODEState,
    eigenvalues: np.ndarray,
    integral: Integral,
    D: int,
) -> float:
    """Evaluate the population reconstruction error represented by a state."""

    Q1, R1, T1 = order_parameters_from_state(state, eigenvalues, D)
    return _generalization_error_from_order_parameters(
        state.T0, Q1, R1, T1, eigenvalues, integral, D
    )


def _generalization_error_from_order_parameters(
    T0: np.ndarray,
    Q1: np.ndarray,
    R1: np.ndarray,
    T1: np.ndarray,
    eigenvalues: np.ndarray,
    integral: Integral,
    D: int,
) -> float:
    K = Q1.shape[0]
    integral = _CachedIntegral(integral)

    error = 0.5 * float(np.sum(eigenvalues)) / D
    for k in range(K):
        covariance = np.array(
            [[T1[k, k], R1[k, k]], [R1[k, k], Q1[k, k]]],
            dtype=np.float64,
        )
        error -= float(integral.J2(covariance))

    decoder_term = 0.0
    for k in range(K):
        for l in range(K):
            covariance = np.array(
                [[Q1[k, k], Q1[k, l]], [Q1[k, l], Q1[l, l]]],
                dtype=np.float64,
            )
            decoder_term += float(integral.I2(covariance)) * T0[k, l]
    error += 0.5 * decoder_term
    return float(error)


def generalization_error_from_factor_vector(
    vector: np.ndarray,
    K: int,
    D: int,
    eigenvalues: np.ndarray,
    integral: Integral,
) -> float:
    """Evaluate population error without reconstructing full densities."""

    factors = _factor_matrix(vector, K, D)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.shape != (D,):
        raise ValueError(f"Expected {D} eigenvalues, got {eigenvalues.shape}")
    encoder_factors = factors[:, :K]
    decoder_factors = factors[:, K:]
    weighted_encoder = eigenvalues[:, None] * encoder_factors
    weighted_decoder = eigenvalues[:, None] * decoder_factors
    Q1 = encoder_factors.T @ weighted_encoder / D
    R1 = decoder_factors.T @ weighted_encoder / D
    T1 = decoder_factors.T @ weighted_decoder / D
    T0 = decoder_factors.T @ decoder_factors
    return _generalization_error_from_order_parameters(
        T0, Q1, R1, T1, eigenvalues, integral, D
    )


def _positive_denominator(value: float, label: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"ODE left its covariance domain: {label}={value}")
    return float(value)


def _covariance2(c11: float, c12: float, c22: float) -> np.ndarray:
    return np.array([[c11, c12], [c12, c22]], dtype=np.float64)


def _dt_component(
    k: int,
    l: int,
    state: ODEState,
    eigenvalues: np.ndarray,
    Q1: np.ndarray,
    integral: Integral,
    D: int,
) -> np.ndarray:
    qkk = _positive_denominator(Q1[k, k], f"Q1[{k},{k}]")
    covariance = np.full((2, 2), qkk, dtype=np.float64)
    derivative = (
        float(integral.J2(covariance))
        / qkk
        * eigenvalues
        * state.r[l, k]
    )

    for a in range(Q1.shape[0]):
        covariance = _covariance2(qkk, Q1[k, a], Q1[a, a])
        derivative -= float(integral.I2(covariance)) * state.t[a, l] * D
    return derivative


def _dt0_component(
    k: int,
    l: int,
    state: ODEState,
    Q1: np.ndarray,
    R1: np.ndarray,
    integral: Integral,
) -> float:
    qkk = _positive_denominator(Q1[k, k], f"Q1[{k},{k}]")
    covariance = np.full((2, 2), qkk, dtype=np.float64)
    derivative = float(integral.J2(covariance)) / qkk * R1[l, k]

    for a in range(Q1.shape[0]):
        covariance = _covariance2(qkk, Q1[k, a], Q1[a, a])
        derivative -= float(integral.I2(covariance)) * state.T0[a, l]
    return float(derivative)


def _dq_component(
    k: int,
    l: int,
    state: ODEState,
    eigenvalues: np.ndarray,
    Q1: np.ndarray,
    R1: np.ndarray,
    T1: np.ndarray,
    integral: Integral,
) -> np.ndarray:
    qkk = _positive_denominator(Q1[k, k], f"Q1[{k},{k}]")
    det_qt = _positive_denominator(
        qkk * T1[k, k] - R1[k, k] ** 2,
        f"Q1[{k},{k}]*T1[{k},{k}]-R1[{k},{k}]^2",
    )
    covariance = _covariance2(qkk, R1[k, k], T1[k, k])
    direct = float(integral.I21(covariance)) * (
        T1[k, k] * state.q[k, l] - R1[k, k] * state.r[k, l]
    )
    direct += float(integral.I22(covariance)) * (
        qkk * state.r[k, l] - R1[k, k] * state.q[k, l]
    )
    direct /= det_qt

    interaction = np.zeros_like(state.q[k, l])
    for a in range(Q1.shape[0]):
        if a == k:
            continue
        qaa = Q1[a, a]
        qka = Q1[k, a]
        det_qq = _positive_denominator(
            qkk * qaa - qka**2,
            f"Q1 pair determinant ({k},{a})",
        )
        covariance3 = np.array(
            [[qkk, qkk, qka], [qkk, qkk, qka], [qka, qka, qaa]],
            dtype=np.float64,
        )
        component = float(integral.I3(covariance3)) * (
            qaa * state.q[k, l] - qka * state.q[a, l]
        )
        covariance3 = np.array(
            [[qkk, qka, qka], [qka, qaa, qaa], [qka, qaa, qaa]],
            dtype=np.float64,
        )
        component += float(integral.I3(covariance3)) * (
            qkk * state.q[a, l] - qka * state.q[k, l]
        )
        interaction -= component * state.T0[k, a] / det_qq

    covariance3 = np.full((3, 3), qkk, dtype=np.float64)
    self_term = (
        -state.T0[k, k]
        / qkk
        * float(integral.I3(covariance3))
        * state.q[k, l]
    )
    return (interaction + self_term + direct) * eigenvalues


def _dr_decoder_component(
    k: int,
    l: int,
    state: ODEState,
    eigenvalues: np.ndarray,
    Q1: np.ndarray,
    integral: Integral,
    D: int,
) -> np.ndarray:
    qkk = _positive_denominator(Q1[k, k], f"Q1[{k},{k}]")
    covariance = np.full((2, 2), qkk, dtype=np.float64)
    derivative = (
        float(integral.J2(covariance))
        / qkk
        * state.q[k, l]
        * eigenvalues
        / D
    )
    for a in range(Q1.shape[0]):
        covariance = _covariance2(qkk, Q1[k, a], Q1[a, a])
        derivative -= float(integral.I2(covariance)) * state.r[a, l]
    return derivative


def _dr_encoder_component(
    k: int,
    l: int,
    state: ODEState,
    eigenvalues: np.ndarray,
    Q1: np.ndarray,
    R1: np.ndarray,
    T1: np.ndarray,
    integral: Integral,
    D: int,
) -> np.ndarray:
    qll = _positive_denominator(Q1[l, l], f"Q1[{l},{l}]")
    det_qt = _positive_denominator(
        qll * T1[l, l] - R1[l, l] ** 2,
        f"Q1[{l},{l}]*T1[{l},{l}]-R1[{l},{l}]^2",
    )
    covariance = _covariance2(qll, R1[l, l], T1[l, l])
    direct = float(integral.I21(covariance)) * (
        T1[l, l] * state.r[k, l] - R1[l, l] * state.t[k, l]
    )
    direct += float(integral.I22(covariance)) * (
        qll * state.t[k, l] - R1[l, l] * state.r[k, l]
    )
    direct /= det_qt

    interaction = np.zeros_like(state.r[k, l])
    for a in range(Q1.shape[0]):
        if a == l:
            continue
        qaa = Q1[a, a]
        qla = Q1[l, a]
        det_qq = _positive_denominator(
            qll * qaa - qla**2,
            f"Q1 pair determinant ({l},{a})",
        )
        covariance3 = np.array(
            [[qll, qll, qla], [qll, qll, qla], [qla, qla, qaa]],
            dtype=np.float64,
        )
        component = float(integral.I3(covariance3)) * (
            qaa * state.r[k, l] - qla * state.r[k, a]
        )
        covariance3 = np.array(
            [[qll, qla, qla], [qla, qaa, qaa], [qla, qaa, qaa]],
            dtype=np.float64,
        )
        component += float(integral.I3(covariance3)) * (
            qll * state.r[k, a] - qla * state.r[k, l]
        )
        interaction -= component * state.T0[l, a] / det_qq

    covariance3 = np.full((3, 3), qll, dtype=np.float64)
    self_term = (
        -state.T0[l, l]
        / qll
        * float(integral.I3(covariance3))
        * state.r[k, l]
    )
    return (interaction + self_term + direct) * eigenvalues / D


def ode_state_derivative(
    state: ODEState,
    eigenvalues: np.ndarray,
    learning_rate: float,
    integral: Integral,
    D: int,
    *,
    derive_t0_from_t: bool = False,
) -> ODEState:
    """Evaluate all four full-spectrum derivatives from one common state."""

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    K = state.T0.shape[0]
    expected_density_shape = (K, K, D)
    if any(array.shape != expected_density_shape for array in (state.q, state.r, state.t)):
        raise ValueError(
            f"Expected q, r and t with shape {expected_density_shape}"
        )
    if eigenvalues.shape != (D,):
        raise ValueError(f"Expected {D} eigenvalues, got {eigenvalues.shape}")
    if not all(
        np.all(np.isfinite(array))
        for array in (state.q, state.r, state.t, state.T0)
    ):
        raise FloatingPointError("ODE state contains a non-finite value")

    Q1, R1, T1 = order_parameters_from_state(state, eigenvalues, D)
    integral = _CachedIntegral(integral)
    dq = np.zeros_like(state.q)
    dr = np.zeros_like(state.r)
    dt = np.zeros_like(state.t)
    dT0 = None if derive_t0_from_t else np.zeros_like(state.T0)

    for k, l in combinations_with_replacement(range(K), 2):
        dq_kl = _dq_component(k, l, state, eigenvalues, Q1, R1, T1, integral)
        dq_lk = (
            dq_kl
            if k == l
            else _dq_component(l, k, state, eigenvalues, Q1, R1, T1, integral)
        )
        dq[k, l] = learning_rate * (dq_kl + dq_lk) / D
        dq[l, k] = dq[k, l]

        dt_kl = _dt_component(k, l, state, eigenvalues, Q1, integral, D)
        dt_lk = (
            dt_kl
            if k == l
            else _dt_component(l, k, state, eigenvalues, Q1, integral, D)
        )
        dt[k, l] = learning_rate * (dt_kl + dt_lk) / D
        dt[l, k] = dt[k, l]

        if dT0 is not None:
            dT0_kl = _dt0_component(k, l, state, Q1, R1, integral)
            dT0_lk = (
                dT0_kl
                if k == l
                else _dt0_component(l, k, state, Q1, R1, integral)
            )
            dT0[k, l] = learning_rate * (dT0_kl + dT0_lk)
            dT0[l, k] = dT0[k, l]

    for k in range(K):
        for l in range(K):
            decoder_part = _dr_decoder_component(
                k, l, state, eigenvalues, Q1, integral, D
            )
            encoder_part = _dr_encoder_component(
                k, l, state, eigenvalues, Q1, R1, T1, integral, D
            )
            dr[k, l] = learning_rate * (decoder_part + encoder_part)

    if dT0 is None:
        dT0 = np.sum(dt, axis=2)
    derivative = ODEState(q=dq, r=dr, t=dt, T0=dT0)
    if not all(
        np.all(np.isfinite(array))
        for array in (derivative.q, derivative.r, derivative.t, derivative.T0)
    ):
        raise FloatingPointError("ODE derivative contains a non-finite value")
    return derivative


def factor_state_derivative(
    vector: np.ndarray,
    K: int,
    D: int,
    eigenvalues: np.ndarray,
    learning_rate: float,
    integral: Integral,
    *,
    norm_floor: float = FACTOR_NORM_FLOOR,
) -> np.ndarray:
    """Evaluate the full-spectrum ODE in factor coordinates."""

    norm_floor = float(norm_floor)
    if not np.isfinite(norm_floor) or norm_floor < 0.0:
        raise ValueError("norm_floor must be finite and >= 0")

    factors = _factor_matrix(vector, K, D)
    encoder_factors = factors[:, :K]
    decoder_factors = factors[:, K:]
    squared_norm = np.einsum("di,di->d", factors, factors, optimize=True)
    if np.any(squared_norm <= norm_floor):
        mode = int(np.flatnonzero(squared_norm <= norm_floor)[0])
        raise FloatingPointError(
            f"Factor mode {mode} has squared norm {squared_norm[mode]:.6e}, "
            f"below floor {norm_floor:.6e}"
        )

    state = _state_from_factor_matrix(factors, K)
    density_derivative = ode_state_derivative(
        state,
        eigenvalues,
        learning_rate,
        integral,
        D,
        derive_t0_from_t=True,
    )

    # Apply H_m z_m without materializing each 2K x 2K block.
    hz = np.empty_like(factors)
    temporary = np.empty((D, K), dtype=np.float64)
    np.einsum(
        "kld,dl->dk", density_derivative.q, encoder_factors, out=hz[:, :K]
    )
    np.einsum(
        "lkd,dl->dk", density_derivative.r, decoder_factors, out=temporary
    )
    hz[:, :K] += temporary
    np.einsum(
        "kld,dl->dk", density_derivative.r, encoder_factors, out=hz[:, K:]
    )
    np.einsum(
        "kld,dl->dk", density_derivative.t, decoder_factors, out=temporary
    )
    hz[:, K:] += temporary

    z_hz = np.einsum("di,di->d", factors, hz, optimize=True)

    # dz = Hz/s - (z^T H z)z/(2s^2), where s = z^T z.
    derivative = hz / squared_norm[:, None]
    derivative -= (
        z_hz / (2.0 * squared_norm**2)
    )[:, None] * factors
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Factor derivative contains a non-finite value")
    return np.ascontiguousarray(derivative.reshape(-1))


def solve_ode_from_model(
    model: AutoEncoder,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    integral: Integral,
    learning_rate: float,
    alpha_end: float,
    sample_step: float,
    on_sample: SampleCallback | None = None,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float = MAX_RK45_ALPHA_STEP,
    factor_norm_floor: float = FACTOR_NORM_FLOOR,
) -> ODEResult:
    """Evolve an initialized model autonomously with adaptive RK45.

    ``sample_step`` controls only streamed output interpolation. It is not an
    integration step: RK45 chooses its own accepted steps from ``rtol`` and
    ``atol``, capped by ``max_step``. A non-negative
    ``alpha_end`` below one is promoted to one so every ODE call performs at
    least one full alpha of evolution.
    """

    alpha_end = float(alpha_end)
    if not np.isfinite(alpha_end):
        raise ValueError("alpha_end must be finite")
    if alpha_end < 0.0:
        raise ValueError("alpha_end must be >= 0")
    alpha_end = max(MIN_ODE_ALPHA, alpha_end)
    sample_step = float(sample_step)
    if not np.isfinite(sample_step) or sample_step <= 0.0:
        raise ValueError("sample_step must be finite and > 0")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be > 0")
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be finite and > 0")

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    D = model.input_size
    K = model.hidden_size
    if eigenvalues.shape != (D,):
        raise ValueError(f"Expected {D} eigenvalues, got {eigenvalues.shape}")

    initial_vector = factor_vector_from_model(model, eigenvectors)
    emitted_samples = 0

    def emit(alpha: float, vector: np.ndarray) -> None:
        nonlocal emitted_samples
        error = generalization_error_from_factor_vector(
            vector, K, D, eigenvalues, integral
        )
        if not np.isfinite(error):
            raise FloatingPointError(
                f"Generalization error became non-finite at alpha={alpha}"
            )
        if on_sample is not None:
            on_sample(ODESample(alpha=float(alpha), generalization_error=error))
        emitted_samples += 1

    emit(0.0, initial_vector)

    def rhs(_alpha: float, vector: np.ndarray) -> np.ndarray:
        return factor_state_derivative(
            vector,
            K,
            D,
            eigenvalues,
            learning_rate,
            integral,
            norm_floor=factor_norm_floor,
        )

    solver = RK45(
        fun=rhs,
        t0=0.0,
        y0=initial_vector,
        t_bound=float(alpha_end),
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    accepted_steps = 0
    sample_index = 1
    last_emitted_alpha = 0.0
    time_tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, alpha_end)

    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(
                f"RK45 failed at alpha={solver.t}: {message or 'unknown reason'}"
            )
        accepted_steps += 1
        interpolator = solver.dense_output()

        while True:
            sample_alpha = sample_index * sample_step
            if sample_alpha > alpha_end + time_tolerance:
                break
            if sample_alpha > solver.t + time_tolerance:
                break
            sample_alpha = min(sample_alpha, alpha_end)
            emit(sample_alpha, interpolator(sample_alpha))
            last_emitted_alpha = sample_alpha
            sample_index += 1

    if last_emitted_alpha != alpha_end:
        emit(alpha_end, solver.y)

    return ODEResult(
        final_state=state_from_factor_vector(solver.y, K, D),
        final_alpha=float(solver.t),
        nfev=int(solver.nfev),
        accepted_steps=accepted_steps,
        emitted_samples=emitted_samples,
        max_step=max_step,
        sample_step=sample_step,
        rtol=rtol,
        atol=atol,
    )
