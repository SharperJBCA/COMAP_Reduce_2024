import pytest
import numpy as np
import h5py
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ProcessLevel2.NoiseStats.NoiseStats import NoiseStatsLevel1, NoiseStatsLevel2

@pytest.fixture
def noise_stats_l1():
    """Fixture to create a fresh NoiseStatsLevel1 instance for each test"""
    return NoiseStatsLevel1()

@pytest.fixture
def noise_stats_l2():
    """Fixture to create a fresh NoiseStatsLevel2 instance for each test"""
    return NoiseStatsLevel2()

@pytest.fixture
def mock_data():
    """Fixture to create mock time series data"""
    np.random.seed(42)  # For reproducibility
    # Create synthetic data with known noise properties
    N = 1000
    sigma_white = 1.0
    sigma_red = 0.1
    alpha = -2.0
    k_ref = 1.0
    # White noise component
    white_noise = np.random.normal(0, 1, N)
    # Red noise component (1/f noise)
    ft_white_noise = np.fft.fft(white_noise)
    ft_k_modes = np.fft.fftfreq(N, d=1)
    ft_k_modes[0] = 1
    P = sigma_white**2 + sigma_red**2 * np.abs(ft_k_modes/k_ref)**alpha
    ft_red_noise = ft_white_noise * np.sqrt(P)

    red_noise = np.real(np.fft.ifft(ft_red_noise))

    return red_noise

@pytest.fixture
def mock_file_info():
    """Fixture to create mock COMAPData instance"""
    mock = Mock()
    mock.level1_path = "mock_level1.h5"
    mock.level2_path = "mock_level2.h5"
    return mock

class TestNoiseStatsLevel1:
    def test_initialization(self, noise_stats_l1):
        """Test proper initialization of class attributes"""
        assert noise_stats_l1.NCHANNELS == 1024
        assert noise_stats_l1.NBANDS == 4
        assert noise_stats_l1.NFEEDS == 19

    def test_auto_rms(self, noise_stats_l1, mock_data):
        """Test auto_rms calculation"""
        rms = noise_stats_l1.auto_rms(mock_data)
        assert isinstance(rms, float)
        # rms should equal approximately 1.0 for the synthetic data
        assert np.abs(rms - 1.0) < 0.1 # Within 10% of expected value

    @patch('h5py.File')
    def test_save_statistics(self, mock_h5py, noise_stats_l1, mock_file_info):
        """Test statistics saving functionality"""
        mock_stats = {
            'auto_rms': np.zeros((19, 4, 1024))
        }

        # Mock the HDF5 file and group
        mock_file = MagicMock()
        mock_group = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.__getitem__.return_value = mock_group
        mock_h5py.return_value = mock_file

        noise_stats_l1.save_statistics(mock_file_info, mock_stats)

        # Verify that datasets were created
        assert mock_group.create_dataset.call_count == len(mock_stats)

    def test_handling_of_nan_data(self, noise_stats_l1):
        """Test proper handling of NaN values"""
        nan_data = np.array([np.nan] * 100)

        # auto_rms should handle NaN values
        rms = noise_stats_l1.auto_rms(nan_data)
        assert np.isnan(rms)


class TestNoiseStatsLevel2:
    def test_initialization(self, noise_stats_l2):
        assert noise_stats_l2.NFEEDS == 19
        assert noise_stats_l2.NBANDS == 4

    def test_auto_rms(self, noise_stats_l2, mock_data):
        rms = noise_stats_l2.auto_rms(mock_data)
        assert isinstance(rms, float)
        assert np.abs(rms - 1.0) < 0.1

    def test_power_spectrum(self, noise_stats_l2, mock_data):
        """Test power spectrum calculation (log-binned)"""
        k, Pk, weights = noise_stats_l2.power_spectrum(mock_data)

        assert len(k) == len(Pk)
        assert len(k) == len(weights)
        assert k[0] > 0
        assert np.all(Pk >= 0)
        assert np.all(weights > 0)

    def test_fnoise_fit(self, noise_stats_l2, mock_data):
        """Test 1/f noise fitting"""
        auto_rms_val = noise_stats_l2.auto_rms(mock_data)
        sigma_white, sigma_red, alpha = noise_stats_l2.fnoise_fit(mock_data, auto_rms_val)

        # Check reasonable bounds for parameters
        assert np.abs(sigma_white - 1.0) < 0.5
        assert sigma_red >= 0
        assert -4 <= alpha <= 4

    def test_fnoise_fit_returns_nan_on_empty(self, noise_stats_l2):
        """Test that fnoise_fit returns NaN for empty data"""
        data = np.array([1.0, 1.0])  # Too short for meaningful spectrum
        result = noise_stats_l2.fnoise_fit(data, 1.0)
        assert all(np.isnan(v) for v in result)

    def test_power_spectrum_handles_nan(self, noise_stats_l2):
        """Test that NaN values in input don't corrupt the FFT"""
        data = np.random.normal(0, 1, 500)
        data[10] = np.nan
        data[100] = np.nan
        k, Pk, weights = noise_stats_l2.power_spectrum(data)
        assert np.all(np.isfinite(Pk))

    def test_model(self, noise_stats_l2):
        """Test noise model evaluation"""
        k = np.array([0.01, 0.1, 1.0])
        result = noise_stats_l2.model(k, sigma_white=1.0, sigma_red=0.5, alpha=-2.0)
        # At k=k_ref=0.1: model = 0.5^2 * 1 + 1.0^2 = 1.25
        assert np.isclose(result[1], 1.25)
        # White noise floor should dominate at high k
        assert result[2] > result[0]  # High k has less red noise


if __name__ == '__main__':
    pytest.main([__file__])
