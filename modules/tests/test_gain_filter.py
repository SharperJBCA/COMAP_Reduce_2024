import pytest
import numpy as np
import h5py
from unittest.mock import Mock, patch, MagicMock
from scipy.sparse.linalg import LinearOperator
import matplotlib 
matplotlib.use('Agg')
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modules.ProcessLevel2.GainFilterAndBin.GainFilterAndBin import GainFilterAndBin

@pytest.fixture
def gain_filter():
    """Fixture to create a fresh GainFilter instance for each test"""
    return GainFilterAndBin()

@pytest.fixture
def mock_data():
    """Fixture to create mock time series data"""
    np.random.seed(42)
    # Create synthetic data with shape (n_bands, n_channels, n_tod)
    return np.random.normal(0, 1, size=(4, 1024, 100))


@pytest.fixture
def mock_realistic_data():
    """Fixture to create mock time series data"""
    np.random.seed(42)
    # Create synthetic data with shape (n_bands, n_channels, n_tod)
    
    tau = 1/50. 
    bw = 2./1024. * 1e9 # Hz 
    tsys = 60.0 # K 
    tsys = np.ones((4, 1024))*tsys 

    # add some spikes 
    for i in range(4):
        idx = np.random.uniform(0, 1024, size=10).astype(int)
        tsys[i, idx] = 1000.0

    rms = tsys/np.sqrt(bw * tau)

    t_receiver = np.random.normal(tsys[:,:,np.newaxis]*np.ones((1,1,100)), rms[:,:,np.newaxis]*np.ones((1,1,100)), size=(4, 1024, 100))
    gain = 5000.0  
    rms_gain = 0.01 * gain 
    gain_receiver = np.ones((4, 1024, 100))*gain 
    # add a sine wave with amplitude rms_gain 
    gain_receiver += np.sin(3*np.linspace(0, 2*np.pi, 100))[np.newaxis,np.newaxis,:]*rms_gain  # sqrt(2) so that the rms == rms_gain
    #np.random.normal(gain, rms_gain, size=(4, 1024, 100))

    d = gain_receiver * t_receiver

    auto_rms = np.nanstd((d[...,::2]-d[...,1::2]), axis=-1)/np.sqrt(2)
    d_mean = np.nanmean(d, axis=-1) 

    d_norm = (d - d_mean[..., np.newaxis])/auto_rms[..., np.newaxis]

    return (d, d_norm, gain_receiver, t_receiver, tsys)

@pytest.fixture
def mock_realistic_gain_drift_data():
    """Fixture to create mock time series data
    
    In this case we add structure to the mean gain per channel. 
    """
    np.random.seed(42)
    # Create synthetic data with shape (n_bands, n_channels, n_tod)
    
    tau = 1/50. 
    bw = 2./1024. * 1e9 # Hz 
    tsys = 60.0 # K 
    tsys = np.ones((4, 1024))*tsys 

    # add some spikes 
    for i in range(4):
        idx = np.random.uniform(0, 1024, size=10).astype(int)
        tsys[i, idx] = 1000.0

    rms = tsys/np.sqrt(bw * tau)

    t_receiver = np.random.normal(tsys[:,:,np.newaxis]*np.ones((1,1,100)), rms[:,:,np.newaxis]*np.ones((1,1,100)), size=(4, 1024, 100))
    gain = 5000.0  
    rms_gain = 0.01 * gain 
    pure_gain_receiver = np.random.uniform(low=4000, high=6000, size=(4, 1024, 1)) * np.ones((1,1,100))
    # add a sine wave with amplitude rms_gain 
    gain_receiver = pure_gain_receiver + np.sin(3*np.linspace(0, 2*np.pi, 100))[np.newaxis,np.newaxis,:]*rms_gain  

    d = gain_receiver * t_receiver

    auto_rms = np.nanstd((d[...,::2]-d[...,1::2]), axis=-1)/np.sqrt(2)
    d_mean = np.nanmean(d, axis=-1) 

    d_norm = (d - d_mean[..., np.newaxis])/auto_rms[..., np.newaxis]

    return (d, d_norm, gain_receiver, pure_gain_receiver, t_receiver, tsys)

@pytest.fixture
def mock_system_temperature():
    """Fixture to create mock system temperature data"""
    np.random.seed(42)
    # Create synthetic Tsys with shape (n_bands, n_channels)
    return np.ones((4, 1024))*60

@pytest.fixture
def mock_file_info():
    """Fixture to create mock COMAPData instance"""
    mock = Mock()
    mock.level1_path = "mock_level1.h5"
    mock.level2_path = "mock_level2.h5"
    return mock

class TestGainFilter:
    def test_initialization(self, gain_filter):
        """Test proper initialization of class attributes"""
        assert gain_filter.NCHANNELS == 1024
        assert gain_filter.NBANDS == 4
        assert gain_filter.NFEEDS == 19
        assert gain_filter.end_cut == 20

    def test_normalise_data(self, gain_filter):
        """Test data normalization"""
        # Create test data
        data = np.ones((4, 1024, 100))  # All ones
        auto_rms = np.ones((4, 1024))   # All ones
        
        # Test with simple data
        normalized = gain_filter.normalise_data(data, auto_rms)
        assert normalized.shape == data.shape
        np.testing.assert_array_almost_equal(normalized, np.zeros_like(data))
        
        # Test with more realistic data
        data = np.random.normal(10, 1, size=(4, 1024, 100))
        auto_rms = np.ones((4, 1024))
        normalized = gain_filter.normalise_data(data, auto_rms)
        assert normalized.shape == data.shape
        assert not np.any(np.isnan(normalized))


    def test_gain_subtraction_fit(self, gain_filter, mock_data, mock_system_temperature):
        """Test gain subtraction fitting"""
        gain = gain_filter.gain_subtraction_fit(mock_data, mock_system_temperature)
        assert isinstance(gain, np.ndarray)
        assert not np.any(np.isnan(gain))
        
        # Test with all bad values
        bad_tsys = np.full_like(mock_system_temperature, np.nan)
        gain = gain_filter.gain_subtraction_fit(mock_data, bad_tsys)
        assert np.all(gain == 0)

    def test_gain_subtraction_fit_realistic(self, gain_filter, mock_realistic_data):
        """Test gain subtraction fitting"""
        data, data_normed,gain_receiver, t_receiver, mock_system_temperature = mock_realistic_data

        gain = gain_filter.gain_subtraction_fit(data_normed, mock_system_temperature)

        tau = 1/50.
        bw = 2./1024. * 1e9
        scale = np.sqrt(bw * tau)
        from matplotlib import pyplot 
        pyplot.plot(gain[:])
        pyplot.plot(np.mean(data_normed,axis=(0,1)),label='Mean Normalised Data')
        pyplot.savefig(f'modules/tests/test_gain_filter_realistic1.png')
        pyplot.close()

        assert np.mean(np.mean(data_normed,axis=(0,1)) - gain[:]) < 0.01 

    def test_gain_subtraction_fit_realistic_gain_drift(self, gain_filter, mock_realistic_gain_drift_data):
        """Test gain subtraction fitting"""
        data, data_normed,gain_receiver, pure_gain_receiver, t_receiver, mock_system_temperature = mock_realistic_gain_drift_data
        gain = gain_filter.gain_subtraction_fit(data_normed, mock_system_temperature)

        tau = 1/50.
        bw = 2./1024. * 1e9
        scale = np.sqrt(bw * tau)
        from matplotlib import pyplot 
        pyplot.plot(gain[:])
        pyplot.plot(np.mean(data_normed,axis=(0,1)),label='Mean Normalised Data')
        pyplot.savefig(f'modules/tests/test_gain_filter_realistic2.png')
        pyplot.close()

        pyplot.plot(np.mean(data/pure_gain_receiver,axis=(0,1))- np.mean(data/pure_gain_receiver))
        pyplot.plot(gain[:]* np.mean(mock_system_temperature)/scale)
        pyplot.savefig(f'modules/tests/test_gain_filter_orig_tod_realistic2.png')
        pyplot.close()

        pyplot.imshow(gain_receiver[2,:,:], aspect='auto')
        pyplot.savefig(f'modules/tests/test_gain_filter_gain_drift_realistic2.png')
        pyplot.close() 

        assert np.mean(np.mean(data_normed,axis=(0,1)) - gain[:]) < 0.01 

if __name__ == '__main__':
    pytest.main([__file__])