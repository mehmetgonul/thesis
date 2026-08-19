"""Autonomous full-spectrum ODE dynamics for the shallow autoencoder.

The equations are adapted from the full-rank analytical updates in
https://github.com/mariaref/NonLinearShallowAE.  A PyTorch model is used only
to construct the initial order-parameter densities; all subsequent evolution
is independent of the model and of SGD.
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
MAX_RK45_ALPHA_STEP = 1.0


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


def initial_state_from_model(
    model: AutoEncoder,
    eigenvectors: np.ndarray,
) -> ODEState:
    """Convert one initialized untied model into full-spectrum densities."""

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

    W_eig = W @ eigenvectors
    V_eig = V @ eigenvectors

    q = np.einsum("kd,ld->kld", W_eig, W_eig)
    r = np.einsum("kd,ld->kld", V_eig, W_eig) / np.sqrt(D)
    t = np.einsum("kd,ld->kld", V_eig, V_eig) / D
    T0 = V @ V.T / D
    return ODEState(q=q, r=r, t=t, T0=T0)


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
    K = Q1.shape[0]

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
            decoder_term += float(integral.I2(covariance)) * state.T0[k, l]
    error += 0.5 * decoder_term
    return float(error)


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
    dq = np.zeros_like(state.q)
    dr = np.zeros_like(state.r)
    dt = np.zeros_like(state.t)
    dT0 = np.zeros_like(state.T0)

    for k, l in combinations_with_replacement(range(K), 2):
        dq_kl = _dq_component(k, l, state, eigenvalues, Q1, R1, T1, integral)
        dq_lk = _dq_component(l, k, state, eigenvalues, Q1, R1, T1, integral)
        dq[k, l] = learning_rate * (dq_kl + dq_lk) / D
        dq[l, k] = dq[k, l]

        dt_kl = _dt_component(k, l, state, eigenvalues, Q1, integral, D)
        dt_lk = _dt_component(l, k, state, eigenvalues, Q1, integral, D)
        dt[k, l] = learning_rate * (dt_kl + dt_lk) / D
        dt[l, k] = dt[k, l]

        dT0_kl = _dt0_component(k, l, state, Q1, R1, integral)
        dT0_lk = _dt0_component(l, k, state, Q1, R1, integral)
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

    derivative = ODEState(q=dq, r=dr, t=dt, T0=dT0)
    if not all(
        np.all(np.isfinite(array))
        for array in (derivative.q, derivative.r, derivative.t, derivative.T0)
    ):
        raise FloatingPointError("ODE derivative contains a non-finite value")
    return derivative


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
) -> ODEResult:
    """Evolve an initialized model autonomously with adaptive RK45.

    ``sample_step`` controls only streamed output interpolation. It is not an
    integration step: RK45 chooses its own accepted steps from ``rtol`` and
    ``atol``, capped at one alpha per accepted step. A non-negative
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

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    D = model.input_size
    K = model.hidden_size
    if eigenvalues.shape != (D,):
        raise ValueError(f"Expected {D} eigenvalues, got {eigenvalues.shape}")

    initial_state = initial_state_from_model(model, eigenvectors)
    initial_vector = pack_state(initial_state)
    emitted_samples = 0

    def emit(alpha: float, vector: np.ndarray) -> None:
        nonlocal emitted_samples
        state = unpack_state(vector, K, D)
        error = generalization_error_from_state(state, eigenvalues, integral, D)
        if not np.isfinite(error):
            raise FloatingPointError(
                f"Generalization error became non-finite at alpha={alpha}"
            )
        if on_sample is not None:
            on_sample(ODESample(alpha=float(alpha), generalization_error=error))
        emitted_samples += 1

    emit(0.0, initial_vector)

    def rhs(_alpha: float, vector: np.ndarray) -> np.ndarray:
        state = unpack_state(vector, K, D)
        return pack_state(
            ode_state_derivative(state, eigenvalues, learning_rate, integral, D)
        )

    solver = RK45(
        fun=rhs,
        t0=0.0,
        y0=initial_vector,
        t_bound=float(alpha_end),
        max_step=MAX_RK45_ALPHA_STEP,
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

    if last_emitted_alpha < alpha_end - time_tolerance:
        emit(alpha_end, solver.y)

    return ODEResult(
        final_state=unpack_state(solver.y, K, D, copy=True),
        final_alpha=float(solver.t),
        nfev=int(solver.nfev),
        accepted_steps=accepted_steps,
        emitted_samples=emitted_samples,
        max_step=MAX_RK45_ALPHA_STEP,
        sample_step=sample_step,
        rtol=rtol,
        atol=atol,
    )
