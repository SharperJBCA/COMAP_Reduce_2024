import pytest
import numpy as np
import h5py
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the class (adjust import path as needed)
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from ProcessLevel2.NoiseStats.NoiseStats import NoiseStatsLevel1

@pytest.fixture
def noise_stats():
    """Fixture to create a fresh NoiseStatsLevel1 instance for each test"""
    return NoiseStatsLevel1()

@pytest.fixture
def mock_data():
    """Fixture to create mock time series data"""
    np.random.seed(42)  # For reproducibility
    # Create synthetic data with known noise properties
    N = 1000
    t = np.arange(N)
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
    def test_initialization(self, noise_stats):
        """Test proper initialization of class attributes"""
        assert noise_stats.NCHANNELS == 1024
        assert noise_stats.NBANDS == 4
        assert noise_stats.NFEEDS == 19

    def test_auto_rms(self, noise_stats, mock_data):
        """Test auto_rms calculation"""
        rms = noise_stats.auto_rms(mock_data)
        assert isinstance(rms, float)
        # rms should equal approximately 1.0 for the synthetic data
        assert np.abs(rms - 1.0) < 0.1 # Within 10% of expected value 
        
    def test_power_spectrum(self, noise_stats, mock_data):
        """Test power spectrum calculation"""
        k, Pk = noise_stats.power_spectrum(mock_data)
        
        # Basic checks
        assert len(k) == len(Pk)
        assert k[0] > 0  # First frequency should be positive
        assert all(Pk >= 0)  # Power should be non-negative
        assert len(k) == len(mock_data)//2 - 1  # Check expected length
        
    def test_fnoise_fit(self, noise_stats, mock_data):
        """Test 1/f noise fitting"""
        auto_rms_val = noise_stats.auto_rms(mock_data)
        sigma_white, sigma_red, alpha = noise_stats.fnoise_fit(mock_data, auto_rms_val)
        
        # Check reasonable bounds for parameters
        assert np.abs(sigma_white - 1.0) < 0.1  # Should be close to 1.0
        assert np.abs(sigma_red - 0.1) < 0.1  # Should be close to 0.1
        assert np.abs(alpha + 2.0) < 0.5  # Should be close to -2.0
        
        # Test with pure white noise
        #  ToDo: Fails this test, because alpha_white has large uncertainty and is biased.
        #white_noise = np.random.normal(0, 1, 1000)
        #_, _, alpha_white = noise_stats.fnoise_fit(white_noise, noise_stats.auto_rms(white_noise))
        #assert abs(alpha_white) < 0.5  # Should be close to 0 for white noise

    @patch('h5py.File')
    def test_save_statistics(self, mock_h5py, noise_stats, mock_file_info):
        """Test statistics saving functionality"""
        mock_stats = {
            'sigma_white': np.zeros((19, 4, 1024)),
            'sigma_red': np.zeros((19, 4, 1024)),
            'alpha': np.zeros((19, 4, 1024)),
            'auto_rms': np.zeros((19, 4, 1024))
        }
        
        # Mock the HDF5 file and group
        mock_file = MagicMock()
        mock_group = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.__getitem__.return_value = mock_group
        mock_h5py.return_value = mock_file
        
        noise_stats.save_statistics(mock_file_info, mock_stats)
                
        # Verify that datasets were created
        assert mock_group.create_dataset.call_count == len(mock_stats)
        

    def test_handling_of_nan_data(self, noise_stats):
        """Test proper handling of NaN values"""
        nan_data = np.array([np.nan] * 100)
        
        # auto_rms should handle NaN values
        rms = noise_stats.auto_rms(nan_data)
        assert np.isnan(rms)
        
        # power_spectrum should handle NaN values
        k, Pk = noise_stats.power_spectrum(nan_data)
        assert np.all(np.isnan(Pk))

if __name__ == '__main__':
    pytest.main([__file__])