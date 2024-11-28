import pytest
from unittest.mock import Mock, patch, call
import subprocess
from datetime import datetime
import logging
import numpy as np
import os
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from DataAcquisition.DataAcquisition import DataAcquisition
from DataAcquisition.DataAssess import DataAssess

@pytest.fixture
def mock_presto_info():
    return {
        'host': 'test.host.com'
    }

@pytest.fixture
def mock_sql_file_command():
    return 'mock_sql_command'

@pytest.fixture
def mock_local_info():
    return {
        'general_data': '/path/to/data'
    }

@pytest.fixture
def data_acquisition(mock_presto_info, mock_sql_file_command, mock_local_info):
    return DataAcquisition(
        presto_info=mock_presto_info,
        sql_file_command=mock_sql_file_command,
        local_info=mock_local_info
    )

@pytest.fixture 
def level1_data_paths():
    return  ['/scratch/nas_cbassarc/sharper/work/COMAP/COMAP_Reduce_2024/modules/tests/test_data/level1_data']

@pytest.fixture
def level2_data_paths():    
    return ['/scratch/nas_cbassarc/sharper/work/COMAP/COMAP_Reduce_2024/modules/tests/test_data/level2_data']


@pytest.fixture
def data_assess(level1_data_paths, level2_data_paths):
    return DataAssess(
        level1_data_paths=level1_data_paths,
        level2_data_paths=level2_data_paths
    )

def test_initialization(mock_presto_info, mock_sql_file_command, mock_local_info):
    """Test initialization with additional kwargs"""
    extra_param = {'extra_param': 'value'}
    da = DataAcquisition(
        presto_info=mock_presto_info,
        sql_file_command=mock_sql_file_command,
        local_info=mock_local_info,
        **extra_param
    )
    assert hasattr(da, 'extra_param')
    assert da.extra_param == 'value'
    assert da.presto_info == mock_presto_info
    assert da.sql_file_command == mock_sql_file_command
    assert da.local_info == mock_local_info

@patch('subprocess.Popen')
def test_query_sql_presto(mock_popen, data_acquisition):
    """Test SQL query to presto database"""
    # Mock the subprocess.Popen response
    mock_process = Mock()
    mock_process.communicate.return_value = (
        b'12345 _ L1 /path/to/data data L1 file.fits\n',
        None
    )
    mock_popen.return_value = mock_process

    result = data_acquisition.query_sql_presto()
    
    # Verify the result
    expected_output = {12345: '/comappath/to/data/data/file.fits'}
    assert result == expected_output

    # Verify SSH command construction
    expected_command = 'ssh -v test.host.com  "python /scr/comap_analysis/sharper/sql_query/sql_query_general.py  " '
    mock_popen.assert_called_once_with(expected_command, shell=True, stdout=subprocess.PIPE)

@patch('subprocess.Popen')
def test_query_sql_presto_invalid_data(mock_popen, data_acquisition):
    """Test SQL query with invalid data"""
    # Mock the subprocess.Popen response with invalid obsid
    mock_process = Mock()
    mock_process.communicate.return_value = (
        b'invalid _ L1 /path/to/data data L1 file.fits\n',
        None
    )
    mock_popen.return_value = mock_process

    result = data_acquisition.query_sql_presto()
    assert result == {}  # Should return empty dict for invalid data

def test_data_assess_level1_files(data_assess):
    """Test functions for finding data files"""
    level1_files = data_assess.get_file_paths_metadata(data_assess.level1_data_paths)
    test_obsid = 29828 
    assert test_obsid in level1_files
    assert level1_files[test_obsid]['file_path'].rsplit('/')[-1] == 'comap-0029828-2022-07-21-092019.hd5'

def test_data_assess_level2_files(data_assess):
    """Test functions for finding data files"""
    level2_files = data_assess.get_file_paths_metadata(data_assess.level2_data_paths, is_level2=True)
    test_obsid = 6329 
    assert test_obsid in level2_files
    assert level2_files[test_obsid]['file_path'].rsplit('/')[-1] == 'Level2_comap-0006329-2019-06-06-185641.hd5'
    assert len(level2_files.keys()) == 15 # This is how many test files we have in level2_data 


# # TODO: Add tests for remaining methods once implemented
# def test_query_sql_general(data_acquisition):
#     """Test general SQL query method"""
#     # Implementation pending
#     pass

# def test_get_existing_obsids(data_acquisition):
#     """Test retrieval of existing observation IDs"""
#     # Implementation pending
#     pass

# def test_check_obstype(data_acquisition):
#     """Test observation type checking"""
#     # Implementation pending
#     pass

# def test_run_rsyncs(data_acquisition):
#     """Test rsync operations"""
#     # Implementation pending
#     pass

# def test_update_known_obsids(data_acquisition):
#     """Test updating known observation IDs"""
#     # Implementation pending
#     pass

# def test_update_permissions(data_acquisition):
#     """Test permission updates"""
#     # Implementation pending
#     pass