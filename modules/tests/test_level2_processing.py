import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import os
from pathlib import Path
import multiprocessing as mp

# Import your pipeline module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from modules.ProcessLevel2.ProcessLevel2 import Level2Pipeline, setup_logging

@pytest.fixture
def mock_parameters():
    return {
        'Master': {
            'log_file': 'test_logs/pipeline.log',
            'sql_database': 'test.db',
            'nprocess': 2,
            '_pipeline': [
                {
                    'package': 'test_package',
                    'module': 'TestModule',
                    'args': {'arg1': 'value1'}
                }
            ]
        }
    }

@pytest.fixture
def mock_db():
    with patch('Pipeline.db') as mock:
        yield mock

@pytest.fixture
def mock_importlib():
    with patch('Pipeline.importlib') as mock:
        yield mock

def test_split_filelist():
    pipeline = Level2Pipeline("dummy_config.toml")
    filelist = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']
    nprocess = 2
    
    result = pipeline.split_filelist(filelist, nprocess)
    
    assert len(result) == nprocess
    assert result[0] == ['file1.txt', 'file3.txt']
    assert result[1] == ['file2.txt', 'file4.txt']

def test_parallel_processing():
    pipeline = Level2Pipeline("test_level2_processing.toml")
    # Test with actual multiprocessing
    filelist = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']
    nprocess = 2
    chunks = pipeline.split_filelist(filelist, nprocess)
    
    pipeline.execute_parallel(chunks, pipeline.parameters)

    assert len(chunks) == nprocess
    # Verify chunks are roughly equal size
    #assert max(len(chunk) for chunk in chunks) - min(len(chunk) for chunk in chunks) <= 1


@pytest.fixture
def cleanup_logging():
    yield
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
