import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker, declarative_base, relationship
from typing import ClassVar 
import os 
from sqlalchemy import and_, or_

# Create an enumerator class for GOOD/BAD FILES 
class FileFlag: 
    GOOD_FILE = True 
    BAD_FILE = False

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

    quality_flags: Mapped[list["QualityFlag"]] = relationship("QualityFlag", back_populates="observation")

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
    
class OldPathData(Base):
    __tablename__ = 'old_path_data'
    
    obsid: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    old_level1_path: Mapped[str | None] = mapped_column(nullable=True)

class QualityFlag(Base):
    __tablename__ = 'quality_flags'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    obsid: Mapped[int] = mapped_column(sa.ForeignKey('comap_data.obsid'), nullable=False)
    pixel: Mapped[int] = mapped_column(nullable=False)  # 1-19
    frequency_band: Mapped[int] = mapped_column(nullable=False)  # 0-7
    is_good: Mapped[bool] = mapped_column(nullable=False, default=True)
    comment: Mapped[str | None] = mapped_column(nullable=True, default=None)  # Adding comment field
    
    # Gain filtered noise statistics
    filtered_red_noise: Mapped[float | None] = mapped_column(nullable=True, default=None)
    filtered_white_noise: Mapped[float | None] = mapped_column(nullable=True, default=None)
    filtered_auto_rms: Mapped[float | None] = mapped_column(nullable=True, default=None)
    filtered_noise_index: Mapped[float | None] = mapped_column(nullable=True, default=None)
    
    # Raw noise statistics
    unfiltered_red_noise: Mapped[float | None] = mapped_column(nullable=True, default=None)
    unfiltered_white_noise: Mapped[float | None] = mapped_column(nullable=True, default=None)
    unfiltered_auto_rms: Mapped[float | None] = mapped_column(nullable=True, default=None)
    unfiltered_noise_index: Mapped[float | None] = mapped_column(nullable=True, default=None)
    
    # Additional statistics
    n_spikes: Mapped[int | None] = mapped_column(nullable=True, default=None)
    n_nan_values: Mapped[int | None] = mapped_column(nullable=True, default=None)
    mean_atm_temp: Mapped[float | None] = mapped_column(nullable=True, default=None)

    observation = relationship("COMAPData", back_populates="quality_flags")
    
    __table_args__ = (
        sa.UniqueConstraint('obsid', 'pixel', 'frequency_band'),
    )


# def create_feed_table(feed_number : int, band_number : int): 
#     tablename = f'comap_feed_{feed_number:02d}_band{band_number:02d}'
#     class COMAPFeed(Base): 
#         __tablename__ = tablename

#         obsid: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

#         level1_path: Mapped[str] = mapped_column(nullable=True)
#         level2_path: Mapped[str] = mapped_column(nullable=True) 

#         feed_number: Mapped[int] = mapped_column(nullable=True)
#         band_number: Mapped[int] = mapped_column(nullable=True)
#         total_bands: Mapped[int] = mapped_column(nullable=True) 

#         date_created: Mapped[str] = mapped_column(nullable=True) 

#         bad_data: Mapped[bool] = mapped_column(nullable=True)
#         stats_fitted_rms_white : Mapped[float] = mapped_column(nullable=True)
#         stats_fitted_rms_red : Mapped[float] = mapped_column(nullable=True)
#         stats_fitted_noise_alpha : Mapped[float] = mapped_column(nullable=True)
#         stats_auto_rms : Mapped[float] = mapped_column(nullable=True) 

#         sky_dip_median_amplitude : Mapped[float] = mapped_column(nullable=True)
#         sky_dip_median_opacity : Mapped[float] = mapped_column(nullable=True)
#         vane_median_tsys : Mapped[float] = mapped_column(nullable=True)
#         vane_median_gain : Mapped[float] = mapped_column(nullable=True) 

#         weather_median_air_temperature : Mapped[float] = mapped_column(nullable=True)
#         weather_median_air_pressure : Mapped[float] = mapped_column(nullable=True)
#         weather_median_relative_humidity : Mapped[float] = mapped_column(nullable=True)
#         weather_median_rain_today : Mapped[float] = mapped_column(nullable=True)
#         weather_median_dew_point : Mapped[float] = mapped_column(nullable=True) 

#     return COMAPFeed 
    

class SQLModule: 

    def __init__(self) -> None:
        self.database = None 

    def connect(self, database_path : str) -> None:
        """
        Set the database path, keeping this name for legacy reasons.

        Actual connection made using _connect routine.
        """
        #self.database = sa.create_engine(f'sqlite:///{database_path}')
        #self.session = sessionmaker(bind=self.database)() 
        #Base.metadata.create_all(self.database)
        self.database_path = database_path

    def _connect(self) -> None:
        """
        Connect to a SQL database
        """
        if self.database is None:
            self.database = sa.create_engine(f'sqlite:///{self.database_path}')
            Base.metadata.create_all(self.database)
        self.session = sessionmaker(bind=self.database)() 


    def disconnect(self) -> None:
        """
        Left for legacy reasons, using _disconnect instead. 
        Remember, db is connected/disconnected in each method. 
        """
        pass

    def _disconnect(self) -> None:
        """
        Disconnect from the SQL database
        """
        self.session.close() 

    def _connect_and_disconnect(self, func):
        """
        Decorator to connect and disconnect from the SQL database
        """
        def wrapper(*args, **kwargs):
            self._connect()
            result = func(*args, **kwargs)
            self._disconnect()
            return result
        return wrapper

    def update_single_value(self, obsid: int, column_name: str, new_value: any) -> None:
        """
        Update a single value for a specific observation ID
        
        Args:
            obsid (int): The observation ID to update
            column_name (str): The name of the column to update
            new_value (any): The new value to set
            
        Raises:
            ValueError: If the column name doesn't exist or obsid not found
        """
        self._connect()
        # Check if column exists
        if column_name not in COMAPData.__table__.columns:
            self._disconnect()
            raise ValueError(f"Column '{column_name}' does not exist in COMAPData table")
        
        # Check if obsid exists
        if not self.obsid_exists(obsid, new_connection=False):
            self._disconnect()
            raise ValueError(f"Observation {obsid} does not exist")
        
        # Update the value
        self.session.query(COMAPData).filter_by(obsid=obsid).update({column_name: new_value})
        self.session.commit()
        self._disconnect()

    def insert_or_update_data(self, data: dict) -> None:
        """
        Insert data into the SQL database or update if entry exists
        """
        if isinstance(data, COMAPData):
            data = data.__dict__
            
        if 'obsid' not in data:
            raise ValueError("obsid is required for insert/update operations")
        self._connect()

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
            
            for pixel in range(19):
                for freq in range(8):
                    flag = QualityFlag(
                        obsid=filtered_data['obsid'],
                        pixel=pixel,
                        frequency_band=freq,
                        is_good=True
                    )
                    self.session.add(flag)

        self.session.commit()
        self._disconnect()

    def initialize_quality_flags(self, obsid: int) -> None:
        """
        Initialize default quality flags for an observation if they don't exist
        """
        # Check if observation exists
        if not self.obsid_exists(obsid):
            raise ValueError(f"Observation {obsid} does not exist")
        self._connect()

        # Get existing flags
        existing_flags = (self.session.query(QualityFlag)
                        .filter_by(obsid=obsid)
                        .all())
        existing_pixels = {(f.pixel, f.frequency_band) for f in existing_flags}
        
        # Create missing flags
        new_flags = []
        for pixel in range(19):
            for freq in range(8):
                if (pixel, freq) not in existing_pixels:
                    new_flags.append(
                        QualityFlag(
                            obsid=obsid,
                            pixel=pixel,
                            frequency_band=freq,
                            is_good=True
                        )
                    )
        
        if new_flags:
            self.session.bulk_save_objects(new_flags)
            self.session.commit()
        self._disconnect()

    def delete_level2_data(self, obsid: int) -> None:
        """
        Delete level 2 data from the SQL database and also delete the file
        """
        self._connect()
        level2_path = self.session.query(COMAPData).filter_by(obsid=obsid).first().level2_path
        if level2_path:
            os.remove(level2_path)
        self.session.query(COMAPData).filter_by(obsid=obsid).update({'level2_path': None})
        self.session.commit()
        self._disconnect()

    def delete_data(self, obsid: int) -> None:
        """
        Delete data from the SQL database
        """
        self._connect()
        self.session.query(COMAPData).filter_by(obsid=obsid).delete()
        self.session.commit()
        self._disconnect()

    def query_data(self, obsid: int) -> dict:
        """
        Query the SQL database for a specific observation ID
        """
        def remove_hidden(data):
            return {k: v for k, v in data.items() if not k.startswith('_')}
        self._connect()
        data = self.session.query(COMAPData).filter_by(obsid=obsid).first()
        self._disconnect()
        if data:
            return remove_hidden(data.__dict__)
        return {}
    
    def obsid_exists(self, obsid: int, new_connection=True) -> bool:
        """
        Check if an observation ID exists in the SQL database
        """
        if new_connection:
            self._connect()
        a = self.session.query(COMAPData.obsid).filter_by(obsid=obsid).scalar() is not None
        if new_connection:
            self._disconnect()
        return a
    
    def query_all_obsids(self):
        """
        Query the SQL database for all observation IDs
        """
        self._connect()
        a = [d.obsid for d in self.session.query(COMAPData.obsid).all()]
        self._disconnect()
        return a
    
    def query_source_group_list(self, source_group: str, source: str = None, min_obsid=7000, max_obsid = 1000000, return_dict=True) -> dict:
        """
        Query the SQL database for a source group 
        """
        self._connect()
        if source_group == 'SkyDip':
            query = self.session.query(COMAPData).filter(COMAPData.source_group.like('SkyDip'))
        elif source_group == 'Calibrator':
            query = self.session.query(COMAPData).filter(COMAPData.source_group.like('Calibrator'))
        else:
            query = self.session.query(COMAPData).filter(or_(COMAPData.source_group.like('Galactic'),COMAPData.source_group.like('Foreground'))) #filter_by(source_group=source_group)
        if source:
            query = query.filter(and_(COMAPData.source.contains(source)))
        
        query = query.filter(COMAPData.obsid >= min_obsid)
        query = query.filter(COMAPData.obsid <= max_obsid)

        data = query.all()

        self._disconnect()

        if return_dict:
            return {d.obsid: d.__dict__ for d in data}
        else:
            return {d.obsid: d for d in data}
        
    def query_obsid_list(self, obsids: list, return_list=False, return_dict=True, source_group=None, source=None, min_obsid=7000) -> dict:
        """
        Query the SQL database for a list of observation IDs
        """
        def remove_hidden(data):
            return {k: v for k, v in data.items() if not k.startswith('_')}
        
        self._connect()
        query = self.session.query(COMAPData).filter(COMAPData.obsid.in_(obsids))
        if source_group:
            query = query.filter_by(source_group=source_group)
        if source:
            query = query.filter(and_(COMAPData.source.contains(source)))
        query = query.filter(COMAPData.obsid >= min_obsid)
        data = query.all()
        self._disconnect()
        if return_list:
            return data

        if return_dict:
            return {d.obsid: remove_hidden(d.__dict__) for d in data}
        else:
            return {d.obsid: d for d in data}
    
    def get_unprocessed_files(self, source_group=None, source=None, min_obsid=0, overwrite=False) -> list:
        """
        Get a list of unprocessed files
        """
        self._connect()
        query = self.session.query(COMAPData)
        if source_group:
            query = query.filter_by(source_group=source_group)
        if source:
            query = query.filter(and_(COMAPData.source.like(source)))

        query = query.filter(COMAPData.obsid >= min_obsid)

        a = query.all()
        self._disconnect()
        return a
        #return query.filter_by(level2_path=None).all()

    def add_quality_flags(self, obsid: int, flags: list[tuple[int, int, bool, str | None]]) -> None:
        """
        Add quality flags for an observation
        
        Args:
            obsid: The observation ID
            flags: List of tuples (pixel, frequency_band, is_good, comment)
        """
        self._connect()
        # Check if observation exists
        existing_flags = self.session.query(QualityFlag).filter_by(obsid=obsid).all()
        existing_flags_dict = {(flag.pixel, flag.frequency_band): flag for flag in existing_flags}

        for pixel, freq, is_good, comment in flags:
            if (pixel, freq) in existing_flags_dict:
                flag = existing_flags_dict[(pixel, freq)]
                flag.is_good = is_good
                flag.comment = comment
            else:
                flag = QualityFlag(
                    obsid=obsid,
                    pixel=pixel,
                    frequency_band=freq,
                    is_good=is_good,
                    comment=comment
                )
                self.session.add(flag)
        self.session.commit()
        self._disconnect()

    def get_quality_flags(self, obsid: int) -> dict[tuple[int, int], tuple[bool, str | None]]:
        """
        Get quality flags for an observation
        
        Returns:
            Dictionary with (pixel, frequency_band) tuple as key and (is_good, comment) as value
        """
        self._connect()
        flags = (self.session.query(QualityFlag)
                .filter_by(obsid=obsid)
                .all())
        self._disconnect()
        return {(flag.pixel, flag.frequency_band): flag for flag in flags}

    def get_bad_data_points(self, obsid: int) -> list[tuple[int, int, str | None]]:
        """
        Get all bad data points (pixel, frequency_band, comment) for an observation
        """
        self._connect()
        bad_flags = (self.session.query(QualityFlag)
                    .filter_by(obsid=obsid, is_good=False)
                    .all())
        self._disconnect()
        return [(flag.pixel, flag.frequency_band, flag.comment) for flag in bad_flags]
    
    def update_quality_flag_all(self, obsid: int, is_good: bool, comment: str = None) -> None:
        """
        Update all pixels/bands with the same quality flag
        """
        self._connect()
        self.session.query(QualityFlag).filter_by(obsid=obsid).update({'is_good': is_good, 'comment': comment})
        self.session.commit()
        self._disconnect()

    def update_quality_statistics(self, obsid: int, pixel: int, freq_band: int, stats: dict) -> None:
        """
        Update noise statistics for a specific obsid-pixel-frequency combination
        
        Args:
            obsid (int): Observation ID
            pixel (int): Pixel number (0-18)
            freq_band (int): Frequency band (0-7)
            stats (dict): Dictionary containing any of these keys:
                - filtered_red_noise (float)
                - filtered_white_noise (float)
                - filtered_auto_rms (float)
                - filtered_noise_index (float)
                - unfiltered_red_noise (float)
                - unfiltered_white_noise (float)
                - unfiltered_auto_rms (float)
                - unfiltered_noise_index (float)
                - n_spikes (int)
                - n_nan_values (int)
                - mean_atm_temp (float)
        """
        self._connect()
        # Get the existing flag
        flag = (self.session.query(QualityFlag)
            .filter_by(obsid=obsid, pixel=pixel, frequency_band=freq_band)
            .first())
        
        if flag is None:
            # If no flag exists, create one with default values
            flag = QualityFlag(
                obsid=obsid,
                pixel=pixel,
                frequency_band=freq_band,
                is_good=True  # default to good
            )
            self.session.add(flag)
        
        # Update the statistics
        for key, value in stats.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        self.session.commit()
        self._disconnect()

    def update_quality_statistics_bulk(self, obsid: int, stats_list: list[dict]) -> None:
        """
        Update noise statistics for multiple pixel-frequency combinations of an observation
        
        Args:
            obsid (int): Observation ID
            stats_list (list): List of dictionaries, each containing:
                - pixel (int): Pixel number
                - frequency_band (int): Frequency band
                - filtered_red_noise (float, optional)
                - filtered_white_noise (float, optional)
                ... etc for all statistics ...
        """
        
        for stats in stats_list:
            pixel = stats.pop('pixel')
            freq_band = stats.pop('frequency_band')
            self.update_quality_statistics(obsid, pixel, freq_band, stats)

db = SQLModule() 


class OldPathModule:
    def __init__(self) -> None:
        self.database = None

    def connect(self, database_path: str) -> None:
        self.database = sa.create_engine(f'sqlite:///{database_path}')
        self.session = sessionmaker(bind=self.database)()
        Base.metadata.create_all(self.database)

    def disconnect(self) -> None:
        self.session.close()

    def insert_or_update_path(self, obsid: int, old_path: str | None) -> None:
        data = self.session.query(OldPathData).filter_by(obsid=obsid).first()
        if data:
            data.old_level1_path = old_path
        else:
            data = OldPathData(obsid=obsid, old_level1_path=old_path)
            self.session.add(data)
        self.session.commit()

    def get_path(self, obsid: int) -> str | None:
        data = self.session.query(OldPathData).filter_by(obsid=obsid).first()
        return data.old_level1_path if data else None