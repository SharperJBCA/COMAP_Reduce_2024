import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker, declarative_base, relationship
from typing import ClassVar 
import os 

Base = declarative_base() 
class COMAPData(Base):
    __tablename__ = 'comap_data'
    
    obsid: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    level1_path: Mapped[str | None] = mapped_column(nullable=True)
    level2_path: Mapped[str | None] = mapped_column(nullable=True)
    bof: Mapped[str | None] = mapped_column(nullable=True)
    bw: Mapped[float | None] = mapped_column(nullable=True)
    coeff_iq: Mapped[int | None] = mapped_column(nullable=True)
    features: Mapped[int | None] = mapped_column(nullable=True)
    fft_shift: Mapped[int | None] = mapped_column(nullable=True)
    instrument: Mapped[str | None] = mapped_column(nullable=True)
    iqcalid: Mapped[int | None] = mapped_column(nullable=True)
    level: Mapped[int | None] = mapped_column(nullable=True)
    nbit: Mapped[int | None] = mapped_column(nullable=True)
    nchan: Mapped[int | None] = mapped_column(nullable=True)
    nint: Mapped[int | None] = mapped_column(nullable=True)
    pixels: Mapped[str | None] = mapped_column(nullable=True)
    platform: Mapped[str | None] = mapped_column(nullable=True)
    source: Mapped[str | None] = mapped_column(nullable=True)
    source_group: Mapped[str | None] = mapped_column(nullable=True)
    telescope: Mapped[str | None] = mapped_column(nullable=True)
    tsamp: Mapped[float | None] = mapped_column(nullable=True)
    utc_start: Mapped[str | None] = mapped_column(nullable=True)
    version: Mapped[str | None] = mapped_column(nullable=True)

    @staticmethod
    def get_source_info(metadata: dict) -> tuple:
        """
        Get the source and source group from the metadata
        """
        source = metadata.get('source', 'Unknown').split(',')[0].strip()
        comment = metadata.get('comment', None)
        source_group = None
        if source:
            source = source.split(',')[0].strip() 
            if 'co' in source.lower():
                source_group = 'CO'
            elif 'field' in source.lower():
                source_group = 'Galactic'
            elif any([x.lower() in source.lower() for x in ['TauA','CasA','CygA','Jupiter','Venus','Mars']]):
                source_group = 'Calibrator'
            elif 'fg' in source.lower():
                source_group = 'Foreground'
            else:
                source_group = 'Other'
        if comment:
            if 'sky nod' in comment.lower():
                source_group = 'SkyDip' 
        return source, source_group

def create_feed_table(feed_number : int, band_number : int): 
    tablename = f'comap_feed_{feed_number:02d}_band{band_number:02d}'
    class COMAPFeed(Base): 
        __tablename__ = tablename

        obsid: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

        level1_path: Mapped[str] = mapped_column(nullable=True)
        level2_path: Mapped[str] = mapped_column(nullable=True) 

        feed_number: Mapped[int] = mapped_column(nullable=True)
        band_number: Mapped[int] = mapped_column(nullable=True)
        total_bands: Mapped[int] = mapped_column(nullable=True) 

        date_created: Mapped[str] = mapped_column(nullable=True) 

        bad_data: Mapped[bool] = mapped_column(nullable=True)
        stats_fitted_rms_white : Mapped[float] = mapped_column(nullable=True)
        stats_fitted_rms_red : Mapped[float] = mapped_column(nullable=True)
        stats_fitted_noise_alpha : Mapped[float] = mapped_column(nullable=True)
        stats_auto_rms : Mapped[float] = mapped_column(nullable=True) 

        sky_dip_median_amplitude : Mapped[float] = mapped_column(nullable=True)
        sky_dip_median_opacity : Mapped[float] = mapped_column(nullable=True)
        vane_median_tsys : Mapped[float] = mapped_column(nullable=True)
        vane_median_gain : Mapped[float] = mapped_column(nullable=True) 

        weather_median_air_temperature : Mapped[float] = mapped_column(nullable=True)
        weather_median_air_pressure : Mapped[float] = mapped_column(nullable=True)
        weather_median_relative_humidity : Mapped[float] = mapped_column(nullable=True)
        weather_median_rain_today : Mapped[float] = mapped_column(nullable=True)
        weather_median_dew_point : Mapped[float] = mapped_column(nullable=True) 

    return COMAPFeed 


class SQLModule: 

    def __init__(self) -> None:
        self.database = None 

    def connect(self, database_path : str) -> None:
        """
        Connect to a SQL database
        """
        self.database = sa.create_engine(f'sqlite:///{database_path}')
        self.session = sessionmaker(bind=self.database)() 
        Base.metadata.create_all(self.database)

    def disconnect(self) -> None:
        """
        Disconnect from the SQL database
        """
        self.session.close() 

    def insert_or_update_data(self, data: dict) -> None:
        """
        Insert data into the SQL database or update if entry exists
        """
        if 'obsid' not in data:
            raise ValueError("obsid is required for insert/update operations")
        
        valid_keys = COMAPData.__table__.columns.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            
        existing = self.session.query(COMAPData).filter_by(obsid=filtered_data['obsid']).first()
        
        if existing:
            # Update only the provided fields
            for key, value in filtered_data.items():
                setattr(existing, key, value)
        else:
            # Create new record
            existing = COMAPData(**filtered_data)
            self.session.add(existing)
            
        self.session.commit()

    def insert_or_update_data_comapdata(self, data: COMAPData) -> None:
        """
        Insert data into the SQL database or update if entry exists
        """
        existing = self.session.query(COMAPData).filter_by(obsid=data.obsid).first()
        
        if existing:
            # Update only the provided fields
            for key in COMAPData.__table__.columns.keys():
                setattr(existing, key, getattr(data, key))
        else:
            # Create new record
            existing = data
            self.session.add(existing)
            
        self.session.commit()

    def delete_level2_data(self, obsid: int) -> None:
        """
        Delete level 2 data from the SQL database and also delete the file
        """
        level2_path = self.session.query(COMAPData).filter_by(obsid=obsid).first().level2_path
        if level2_path:
            os.remove(level2_path)
        self.session.query(COMAPData).filter_by(obsid=obsid).update({'level2_path': None})
        self.session.commit()


    def delete_data(self, obsid: int) -> None:
        """
        Delete data from the SQL database
        """
        self.session.query(COMAPData).filter_by(obsid=obsid).delete()
        self.session.commit()

    def query_data(self, obsid: int) -> dict:
        """
        Query the SQL database for a specific observation ID
        """
        def remove_hidden(data):
            return {k: v for k, v in data.items() if not k.startswith('_')}
        data = self.session.query(COMAPData).filter_by(obsid=obsid).first()
        if data:
            return remove_hidden(data.__dict__)
        return {}
    
    def obsid_exists(self, obsid: int) -> bool:
        """
        Check if an observation ID exists in the SQL database
        """
        return self.session.query(COMAPData.obsid).filter_by(obsid=obsid).scalar() is not None
    
    def query_all_obsids(self):
        """
        Query the SQL database for all observation IDs
        """
        return [d.obsid for d in self.session.query(COMAPData.obsid).all()]
    
    def query_source_group_list(self, source_group: str, source: str = None, min_obsid=7000, max_obsid = 1000000, return_dict=True) -> dict:
        """
        Query the SQL database for a source group 
        """

        query = self.session.query(COMAPData).filter_by(source_group=source_group)
        if source:
            query = query.filter_by(source=source)

        query = query.filter(COMAPData.obsid >= min_obsid)
        query = query.filter(COMAPData.obsid <= max_obsid)

        data = query.all()

        if return_dict:
            return {d.obsid: d.__dict__ for d in data}
        else:
            return {d.obsid: d for d in data}
        
    def query_obsid_list(self, obsids: list, return_dict=True, source_group=None, source=None, min_obsid=7000) -> dict:
        """
        Query the SQL database for a list of observation IDs
        """
        def remove_hidden(data):
            return {k: v for k, v in data.items() if not k.startswith('_')}
        
        query = self.session.query(COMAPData).filter(COMAPData.obsid.in_(obsids))
        if source_group:
            query = query.filter_by(source_group=source_group)
        if source:
            query = query.filter_by(source=source)
        query = query.filter(COMAPData.obsid >= min_obsid)
        data = query.all()

        if return_dict:
            return {d.obsid: remove_hidden(d.__dict__) for d in data}
        else:
            return {d.obsid: d for d in data}
    
    def get_unprocessed_files(self, source_group=None, source=None, min_obsid=0, overwrite=False) -> list:
        """
        Get a list of unprocessed files
        """
        query = self.session.query(COMAPData)
        if source_group:
            query = query.filter_by(source_group=source_group)
        if source:
            query = query.filter_by(source=source)
        query = query.filter(COMAPData.obsid >= min_obsid)

        if overwrite:
            return query.all()
        return query.filter_by(level2_path=None).all()

db = SQLModule() 