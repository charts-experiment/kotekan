"""Configure guards for the optional shared scientific dependency (no GPU needed)."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

CHECKOUT = Path(__file__).resolve().parents[2]
CUDA = os.environ.get("CHARTS_TEST_CUDA_COMPILER", shutil.which("nvcc"))
CUDA_ARCH = os.environ.get("CHARTS_TEST_CUDA_ARCHITECTURES", "75")
DISABLED = [f"-DUSE_{feature}=OFF" for feature in
            ("DPDK", "HDF5", "ASDF", "JULIA", "AIRSPY", "GDAL", "LAPACK_BLAZE", "FFTW")]


def configure(build, *options):
    return subprocess.run(["cmake", "-S", str(CHECKOUT), "-B", build, *DISABLED, *options],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class BuildGuards(unittest.TestCase):
    def test_disabled_feature_needs_no_scientific_checkout(self):
        with tempfile.TemporaryDirectory(prefix="charts-disabled-") as build:
            result = configure(build, "-DUSE_CUDA=OFF",
                               "-DCHARTS_BEAMFORMER_SOURCE_DIR=/nonexistent/scientific-source")
            self.assertEqual(result.returncode, 0, result.stdout)
            targets = subprocess.check_output(["cmake", "--build", build, "--target", "help"], text=True)
            self.assertNotIn("beamformer_core", targets)
            self.assertNotIn("charts_beamformer_link_check", targets)

    def test_enabled_feature_requires_cuda(self):
        with tempfile.TemporaryDirectory(prefix="charts-no-cuda-") as build:
            result = configure(build, "-DUSE_CUDA=OFF", "-DUSE_CHARTS_BEAMFORMER=ON")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires the CUDA backend", result.stdout)

    @unittest.skipUnless(CUDA, "Set CHARTS_TEST_CUDA_COMPILER or put nvcc on PATH")
    def test_enabled_feature_rejects_missing_source(self):
        with tempfile.TemporaryDirectory(prefix="charts-no-source-") as build:
            result = configure(build, "-DUSE_CUDA=ON", "-DUSE_CHARTS_BEAMFORMER=ON",
                               f"-DCMAKE_CUDA_COMPILER={CUDA}", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}",
                               "-DCHARTS_BEAMFORMER_SOURCE_DIR=/nonexistent/scientific-source")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Set CHARTS_BEAMFORMER_SOURCE_DIR", result.stdout)


if __name__ == "__main__":
    unittest.main()
