import pytest
import os
from sqlalchemy.exc import IntegrityError
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from SQLModule.SQLModule import SQLModule, COMAPData, create_feed_table

@pytest.fixture
def sql_db():
    db = SQLModule()
    db.connect("test.db")
    yield db
    db.disconnect()
    os.remove("test.db")

@pytest.fixture
def sample_feed_data():
    return {'obsid':1, 'feed_number':1, 'bad_data':False}

@pytest.fixture
def sample_data():
    return {
        "obsid": 1,
        "level1_path": "/path/to/level1",
        "level2_path": "/path/to/level2",
        "bof": "test_bof",
        "bw": 1.0,
        "coeff_iq": 2,
        "features": 3,
        "fft_shift": 4,
        "instrument": "test_instrument",
        "iqcalid": 5,
        "level": 6,
        "nbit": 7,
        "nchan": 8,
        "nint": 9,
        "pixels": "test_pixels",
        "platform": "test_platform",
        "source": "test_source",
        "telescope": "test_telescope",
        "tsamp": 10.0,
        "utc_start": "2024-01-01",
        "version": "1.0"
    }

def test_database_connection(sql_db):
    assert sql_db.database is not None
    assert sql_db.session is not None

def test_insert_basic_data(sql_db, sample_data):
    sql_db.insert_or_update_data(sample_data)
    result = sql_db.session.query(COMAPData).filter_by(obsid=1).first()
    assert result is not None
    assert result.level1_path == "/path/to/level1"

def test_update_existing_data(sql_db, sample_data):
    # Initial insert
    sql_db.insert_or_update_data(sample_data)
    
    # Update
    updated_data = {"obsid": 1, "level1_path": "/new/path"}
    sql_db.insert_or_update_data(updated_data)
    
    result = sql_db.session.query(COMAPData).filter_by(obsid=1).first()
    assert result.level1_path == "/new/path"
    assert result.level2_path == "/path/to/level2"  # Original data preserved

def test_partial_insert(sql_db):
    minimal_data = {"obsid": 1, "level1_path": "/path/to/level1"}
    sql_db.insert_or_update_data(minimal_data)
    
    result = sql_db.session.query(COMAPData).filter_by(obsid=1).first()
    assert result.obsid == 1
    assert result.level1_path == "/path/to/level1"

def test_missing_obsid(sql_db):
    invalid_data = {"level1_path": "/path/to/level1"}
    with pytest.raises(ValueError, match="obsid is required"):
        sql_db.insert_or_update_data(invalid_data)

def test_multiple_records(sql_db, sample_data):
    # Insert first record
    sql_db.insert_or_update_data(sample_data)
    
    # Insert second record
    sample_data2 = sample_data.copy()
    sample_data2["obsid"] = 2
    sql_db.insert_or_update_data(sample_data2)
    
    results = sql_db.session.query(COMAPData).all()
    assert len(results) == 2

def test_update_float_values(sql_db):
    # Test handling of float values
    data = {"obsid": 1, "bw": 1.5, "tsamp": 2.5}
    sql_db.insert_or_update_data(data)
    
    result = sql_db.session.query(COMAPData).filter_by(obsid=1).first()
    assert result.bw == 1.5
    assert result.tsamp == 2.5

def test_insert_invalid_key(sql_db, sample_data):
    sample_data['invalid_key'] = 'invalid_value'
    sql_db.insert_or_update_data(sample_data)

def test_query_data(sql_db, sample_data):
    sql_db.insert_or_update_data(sample_data)
    result = sql_db.query_data(1)
    assert result["obsid"] == 1
    assert result["level1_path"] == "/path/to/level1"

def test_query_obsid_list(sql_db, sample_data):
    sql_db.insert_or_update_data(sample_data)
    extra_data = {'obsid': 2, 'level1_path': '/path/to/level1'}
    sql_db.insert_or_update_data(extra_data)
    result = sql_db.query_obsid_list([1,2])
    assert 1 in result
    assert result[1]["obsid"] == 1
    assert result[1]["level1_path"] == "/path/to/level1"
    assert 2 in result
    assert result[2]["obsid"] == 2

def test_query_missing_obsid(sql_db, sample_data):
    obsids = [1, 2, 3]
    sql_db.insert_or_update_data(sample_data)
    result = sql_db.query_obsid_list(obsids)
    assert result[1]['obsid'] == sample_data['obsid']
    assert result[1] == sample_data
    # Check for missing obsids
    assert not all([obsid in result for obsid in obsids])

def test_create_feed_table(sql_db, sample_feed_data):
    feed = 1 
    band = 0 
    comap_feed_class = create_feed_table(feed, band)

    # Insert data
    sql_db.session.add(comap_feed_class(**sample_feed_data))
    
def test_get_source_info():
    metadata = [{'source':'TauA,', 'comment':'test comment'},
                {'source':'GField10,CasA', 'comment':'test comment'},
                {'source':'GField10, ', 'comment':'Sky Nod'},
                {'source':'fg10, ,', 'comment':'test comment'}] 

    for md in metadata:
        source, source_group = COMAPData.get_source_info(md)
        md['source'] = source   
        md['source_group'] = source_group

    assert metadata[0]['source'] == 'TauA'
    assert metadata[0]['source_group'] == 'Calibrator'
    assert metadata[1]['source'] == 'GField10'
    assert metadata[1]['source_group'] == 'Galactic'
    assert metadata[2]['source'] == 'GField10'
    assert metadata[2]['source_group'] == 'SkyDip'
    assert metadata[3]['source'] == 'fg10'
    assert metadata[3]['source_group'] == 'Foreground'
