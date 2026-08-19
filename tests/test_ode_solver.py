from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.autoencoder import AutoEncoder
from src.config import RunConfig
from src.hash import check_hash, make_hash
from src.integrals.analytical_integrals import LinearAnalyticIntegral
from src.ode_solver import (
    MAX_RK45_ALPHA_STEP,
    MIN_ODE_ALPHA,
    initial_state_from_model,
    ode_state_derivative,
    order_parameters_from_state,
    pack_state,
    solve_ode_from_model,
    unpack_state,
)


class ODESolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.D = 5
        self.K = 2
        self.config = RunConfig(
            dataset="synthetic",
            activation_function="linear",
            tied=False,
            bias=False,
            alpha=1,
            learning_rate=0.1,
            D=self.D,
            K=self.K,
            init_std=0.2,
        )
        torch.manual_seed(7)
        self.model = AutoEncoder(self.config)
        rng = np.random.default_rng(11)
        self.eigenvectors, _ = np.linalg.qr(rng.normal(size=(self.D, self.D)))
        self.eigenvalues = np.linspace(0.4, 1.6, self.D)

    def test_initial_densities_reproduce_weight_order_parameters(self) -> None:
        state = initial_state_from_model(self.model, self.eigenvectors)
        Q1, R1, T1 = order_parameters_from_state(
            state, self.eigenvalues, self.D
        )

        W = self.model.encoder.weight.detach().numpy().astype(np.float64)
        V = self.model.decoder.weight.detach().numpy().T.astype(np.float64)
        omega = self.eigenvectors @ np.diag(self.eigenvalues) @ self.eigenvectors.T

        np.testing.assert_allclose(Q1, W @ omega @ W.T / self.D)
        np.testing.assert_allclose(R1, V @ omega @ W.T / self.D**1.5)
        np.testing.assert_allclose(T1, V @ omega @ V.T / self.D**2)
        np.testing.assert_allclose(state.T0, V @ V.T / self.D)
        np.testing.assert_allclose(state.T0, np.sum(state.t, axis=2))

    def test_pack_round_trip_and_derivative_are_finite(self) -> None:
        state = initial_state_from_model(self.model, self.eigenvectors)
        restored = unpack_state(
            pack_state(state), self.K, self.D, copy=True
        )
        for actual, expected in (
            (restored.q, state.q),
            (restored.r, state.r),
            (restored.t, state.t),
            (restored.T0, state.T0),
        ):
            np.testing.assert_array_equal(actual, expected)

        derivative = ode_state_derivative(
            state,
            self.eigenvalues,
            self.config.learning_rate,
            LinearAnalyticIntegral,
            self.D,
        )
        for array in (
            derivative.q,
            derivative.r,
            derivative.t,
            derivative.T0,
        ):
            self.assertTrue(np.all(np.isfinite(array)))
        np.testing.assert_allclose(derivative.q, derivative.q.transpose(1, 0, 2))
        np.testing.assert_allclose(derivative.t, derivative.t.transpose(1, 0, 2))
        np.testing.assert_allclose(derivative.T0, derivative.T0.T)

    def test_rk45_streams_requested_samples(self) -> None:
        samples = []
        result = solve_ode_from_model(
            model=self.model,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            integral=LinearAnalyticIntegral,
            learning_rate=0.01,
            alpha_end=1.0,
            sample_step=0.25,
            on_sample=samples.append,
        )

        np.testing.assert_allclose(
            [sample.alpha for sample in samples],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
        self.assertTrue(all(np.isfinite(sample.generalization_error) for sample in samples))
        self.assertEqual(result.emitted_samples, 5)
        self.assertGreater(result.nfev, 0)
        self.assertGreater(result.accepted_steps, 0)
        self.assertAlmostEqual(result.final_alpha, 1.0)

    def test_rk45_solves_at_least_one_alpha(self) -> None:
        result = solve_ode_from_model(
            model=self.model,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            integral=LinearAnalyticIntegral,
            learning_rate=0.01,
            alpha_end=0.0,
            sample_step=1.0,
        )

        self.assertEqual(result.final_alpha, MIN_ODE_ALPHA)
        self.assertGreater(result.nfev, 0)
        self.assertGreater(result.accepted_steps, 0)

    def test_rk45_takes_at_least_one_step_and_emits_error_per_alpha(self) -> None:
        samples = []
        result = solve_ode_from_model(
            model=self.model,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            integral=LinearAnalyticIntegral,
            learning_rate=0.01,
            alpha_end=3.0,
            sample_step=1.0,
            on_sample=samples.append,
        )

        self.assertGreaterEqual(result.accepted_steps, 3)
        self.assertEqual(result.max_step, MAX_RK45_ALPHA_STEP)
        np.testing.assert_allclose(
            [sample.alpha for sample in samples],
            [0.0, 1.0, 2.0, 3.0],
        )

    def test_ode_flag_never_changes_experiment_hash(self) -> None:
        legacy_payload = dataclasses.asdict(self.config)
        legacy_payload.pop("ode_sol")

        self.assertEqual(make_hash(self.config), make_hash(legacy_payload))
        ode_config = dataclasses.replace(self.config, ode_sol=True)
        self.assertEqual(make_hash(ode_config), make_hash(self.config))

    def test_online_and_ode_completion_share_hash_but_use_distinct_files(self) -> None:
        ode_config = dataclasses.replace(self.config, ode_sol=True)
        online_config = dataclasses.replace(self.config, ode_sol=False)
        run_hash = make_hash(ode_config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = Path(temporary_directory)
            base = artifacts / run_hash / "autoencoder" / "seed_40"
            base.mkdir(parents=True)

            for name in (
                "ode_state.npz",
                "ode_loss.jsonl",
                "ode_meta.json",
                "initial_model.pt",
            ):
                (base / name).touch()

            self.assertEqual(
                check_hash(ode_config, 40, artifacts),
                (str(base / "ode_state.npz"), str(base / "ode_loss.jsonl")),
            )
            self.assertIsNone(check_hash(online_config, 40, artifacts))

            (base / "model.pt").touch()
            (base / "loss.jsonl").touch()
            self.assertEqual(
                check_hash(online_config, 40, artifacts),
                (str(base / "model.pt"), str(base / "loss.jsonl")),
            )

            bias_config = dataclasses.replace(
                self.config, bias=True, ode_sol=True
            )
            bias_base = (
                artifacts
                / make_hash(bias_config)
                / "autoencoder"
                / "seed_40"
            )
            bias_base.mkdir(parents=True)
            (bias_base / "model.pt").touch()
            (bias_base / "loss.jsonl").touch()
            self.assertEqual(
                check_hash(bias_config, 40, artifacts),
                (str(bias_base / "model.pt"), str(bias_base / "loss.jsonl")),
            )


if __name__ == "__main__":
    unittest.main()
