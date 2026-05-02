import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Strategy import config
from Strategy.factor.factor_base import load_all_factors, load_factor
from Strategy.label.label_generator import load_label


class DtypeLoadingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.old_factor_dir = config.FACTOR_OUTPUT_DIR
        self.old_label_dir = config.LABEL_OUTPUT_DIR
        config.FACTOR_OUTPUT_DIR = self.tmp_path / "factors"
        config.LABEL_OUTPUT_DIR = self.tmp_path / "labels"
        config.FACTOR_OUTPUT_DIR.mkdir()
        config.LABEL_OUTPUT_DIR.mkdir()

    def tearDown(self):
        config.FACTOR_OUTPUT_DIR = self.old_factor_dir
        config.LABEL_OUTPUT_DIR = self.old_label_dir
        self._tmp.cleanup()

    def _write_wide_feather(self, path: Path) -> None:
        df = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "000001": np.array([1.25, 2.50], dtype=np.float64),
                "000002": np.array([3.75, 4.00], dtype=np.float64),
            }
        )
        df.to_feather(path)

    def test_load_factor_can_downcast_values_to_float32(self):
        self._write_wide_feather(config.FACTOR_OUTPUT_DIR / "alpha.fea")

        factor = load_factor("alpha", dtype="float32")

        self.assertEqual(str(factor["000001"].dtype), "float32")
        self.assertEqual(str(factor["000002"].dtype), "float32")
        self.assertIsInstance(factor.index, pd.DatetimeIndex)

    def test_load_all_factors_passes_dtype_to_each_factor(self):
        self._write_wide_feather(config.FACTOR_OUTPUT_DIR / "alpha.fea")
        self._write_wide_feather(config.FACTOR_OUTPUT_DIR / "beta.fea")

        factors = load_all_factors(dtype="float32")

        self.assertEqual(set(factors), {"alpha", "beta"})
        self.assertEqual(str(factors["alpha"]["000001"].dtype), "float32")
        self.assertEqual(str(factors["beta"]["000002"].dtype), "float32")

    def test_load_label_can_downcast_values_to_float32(self):
        self._write_wide_feather(config.LABEL_OUTPUT_DIR / "LABEL_OPEN930_1000.fea")

        label = load_label("OPEN930_1000", dtype="float32")

        self.assertEqual(str(label["000001"].dtype), "float32")
        self.assertEqual(str(label["000002"].dtype), "float32")
        self.assertIsInstance(label.index, pd.DatetimeIndex)

    def test_default_loaders_preserve_float64_values(self):
        self._write_wide_feather(config.FACTOR_OUTPUT_DIR / "alpha.fea")
        self._write_wide_feather(config.LABEL_OUTPUT_DIR / "LABEL_OPEN930_1000.fea")

        factor = load_factor("alpha")
        label = load_label("OPEN930_1000")

        self.assertEqual(str(factor["000001"].dtype), "float64")
        self.assertEqual(str(label["000001"].dtype), "float64")


if __name__ == "__main__":
    unittest.main()
