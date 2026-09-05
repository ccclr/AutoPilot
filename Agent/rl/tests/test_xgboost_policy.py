from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

try:
    import xgboost  # noqa: F401
    from cmab.xgboost_policy import XGBoostPolicy
except ImportError:
    xgboost = None
    XGBoostPolicy = None  # type: ignore


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=2,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=4,"
    "fast_path_timeout=300,k=4",
)


@unittest.skipIf(xgboost is None, "xgboost is not installed")
class XGBoostPolicyTests(unittest.TestCase):
    def _policy(self, **kwargs) -> XGBoostPolicy:
        defaults = dict(
            arms=ARMS,
            feature_dim=5,
            policy_name="xgboost",
            n_estimators=4,
            max_depth=3,
            min_samples_to_fit=2,
            fit_every=1,
            random_state=0,
        )
        defaults.update(kwargs)
        return XGBoostPolicy(**defaults)

    def test_fits_and_selects_an_arm(self) -> None:
        policy = self._policy()
        context = [0.2, 0.4, 0.1, 0.3, 0.5]
        policy.update(
            [ARMS[0], ARMS[1]],
            [1.2, 0.4],
            contexts=[context, context],
            shared_seed_hex="abc123",
        )
        chosen = policy.select_arm(context, shared_seed_hex="abc123")
        self.assertIn(chosen, ARMS)

    def test_checkpoint_round_trip(self) -> None:
        source = self._policy()
        context = [0.1, 0.2, 0.3, 0.4, 0.5]
        source.update(
            [ARMS[0], ARMS[1]],
            [1.0, 0.3],
            contexts=[context, context],
            shared_seed_hex="s1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xgb.joblib"
            source.save(str(path))
            target = self._policy()
            target.load(str(path))
        self.assertTrue(target._is_fitted)
        self.assertEqual(target._update_count, source._update_count)
        self.assertEqual(
            source.select_arm(context, shared_seed_hex="s2"),
            target.select_arm(context, shared_seed_hex="s2"),
        )

    def test_numeric_encoding_matches_raw_values(self) -> None:
        policy = self._policy()
        encoded = policy._arm_to_vector(ARMS[0])
        np.testing.assert_allclose(encoded, [100000, 32, 2, 0, 1])


if __name__ == "__main__":
    unittest.main()
