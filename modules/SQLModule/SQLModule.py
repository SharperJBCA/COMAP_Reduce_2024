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
    summary: Mapped["ObservationSummary"] = relationship("ObservationSummary", back_populates="observation", uselist=False)

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

class ObservationSummary(Base):
    """Per-observation summary statistics, queryable from SQL without opening HDF5 files."""
    __tablename__ = 'observation_summary'

    obsid: Mapped[int] = mapped_column(sa.ForeignKey('comap_data.obsid'), primary_key=True)

    # Processing status
    processing_status: Mapped[str | None] = mapped_column(nullable=True, default=None)  # pending, complete, failed
    processing_error: Mapped[str | None] = mapped_column(nullable=True, default=None)

    # System temperature (median across feeds/bands/channels)
    median_tsys: Mapped[float | None] = mapped_column(nullable=True, default=None)

    # Atmosphere
    mean_tau: Mapped[float | None] = mapped_column(nullable=True, default=None)

    # Calibrator fitting
    calibrator_flux: Mapped[float | None] = mapped_column(nullable=True, default=None)
    calibrator_flux_error: Mapped[float | None] = mapped_column(nullable=True, default=None)
    calibrator_chi2: Mapped[float | None] = mapped_column(nullable=True, default=None)
    pointing_offset_az: Mapped[float | None] = mapped_column(nullable=True, default=None)
    pointing_offset_el: Mapped[float | None] = mapped_column(nullable=True, default=None)
    pointing_offset_ra: Mapped[float | None] = mapped_column(nullable=True, default=None)
    pointing_offset_dec: Mapped[float | None] = mapped_column(nullable=True, default=None)

    # Scan info
    n_scans: Mapped[int | None] = mapped_column(nullable=True, default=None)

    observation = relationship("COMAPData", back_populates="summary")

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
    

class CalibrationFlux(Base):
    """Per-observation, per-feed/band/channel measured flux from calibrator fitting."""
    __tablename__ = 'calibration_flux'

    id: Mapped[int] = mapped_column(primary_key=True)
    obsid: Mapped[int] = mapped_column(sa.ForeignKey('comap_data.obsid'), nullable=False)
    feed: Mapped[int] = mapped_column(nullable=False)       # 1-19
    band: Mapped[int] = mapped_column(nullable=False)       # 0-3
    channel: Mapped[int] = mapped_column(nullable=False)    # 0-1
    mjd: Mapped[float] = mapped_column(nullable=False)
    source: Mapped[str] = mapped_column(nullable=False)
    measured_flux: Mapped[float | None] = mapped_column(nullable=True)  # Jy
    flux_error: Mapped[float | None] = mapped_column(nullable=True)     # Jy
    amplitude: Mapped[float | None] = mapped_column(nullable=True)      # K (peak)
    chi2: Mapped[float | None] = mapped_column(nullable=True)

    __table_args__ = (
        sa.UniqueConstraint('obsid', 'feed', 'band', 'channel'),
    )


class CalibrationModelFit(Base):
    """Fitted temporal calibration model parameters per feed/band/channel."""
    __tablename__ = 'calibration_model_fit'

    id: Mapped[int] = mapped_column(primary_key=True)
    feed: Mapped[int] = mapped_column(nullable=False)       # 1-19
    band: Mapped[int] = mapped_column(nullable=False)       # 0-3
    channel: Mapped[int] = mapped_column(nullable=False)    # 0-1
    source: Mapped[str | None] = mapped_column(nullable=True)
    model_type: Mapped[str | None] = mapped_column(nullable=True)  # "polynomial", "nearest", "mean"
    # Polynomial coefficients as JSON: [c0, c1, ...], t = (mjd-59000)/365.25
    poly_coeffs: Mapped[str | None] = mapped_column(nullable=True)
    # For nearest-neighbor: JSON arrays of (mjd, gain) lookup
    nearest_mjds: Mapped[str | None] = mapped_column(nullable=True)
    nearest_gains: Mapped[str | None] = mapped_column(nullable=True)
    fit_rms: Mapped[float | None] = mapped_column(nullable=True)
    n_observations: Mapped[int | None] = mapped_column(nullable=True)
    updated_at: Mapped[str | None] = mapped_column(nullable=True)

    __table_args__ = (
        sa.UniqueConstraint('feed', 'band', 'channel'),
    )


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

    def clear_level2_info(self, obsid: int, clear_quality_stats: bool = False) -> None:
        """
        Clear Level-2 information for an obsid (without deleting any files).
        Optionally clears QualityFlag statistics but keeps the flags and is_good state.
        """
        self._connect()

        # Null-out the Level-2 path
        self.session.query(COMAPData).filter_by(obsid=obsid).update({'level2_path': None})

        if clear_quality_stats:
            # Set all stat fields on QualityFlag to NULL
            self.session.query(QualityFlag).filter_by(obsid=obsid).update({
                'filtered_red_noise': None,
                'filtered_white_noise': None,
                'filtered_auto_rms': None,
                'filtered_noise_index': None,
                'unfiltered_red_noise': None,
                'unfiltered_white_noise': None,
                'unfiltered_auto_rms': None,
                'unfiltered_noise_index': None,
                'n_spikes': None,
                'n_nan_values': None,
                'mean_atm_temp': None,
                'comment': None,  # optional; drop if you want to keep comments
            })

        self.session.commit()
        self._disconnect()


    def scrub_missing_level2_paths(self, clear_quality_stats: bool = False) -> list[int]:
        """
        For every row with a non-empty Level-2 path, check if the file exists.
        If it does not, clear Level-2 info (and optionally stats).
        Returns a list of obsids that were scrubbed.
        """
        import os

        self._connect()
        rows = (self.session.query(COMAPData.obsid, COMAPData.level2_path)
                .filter(COMAPData.level2_path.isnot(None))
                .filter(COMAPData.level2_path != "")
                .all())

        missing = []
        for obsid, path in rows:
            try:
                exists = os.path.exists(path)
            except Exception:
                exists = False
            if not exists:
                # Clear within the same transaction/session for efficiency
                self.session.query(COMAPData).filter_by(obsid=obsid).update({'level2_path': None})
                if clear_quality_stats:
                    self.session.query(QualityFlag).filter_by(obsid=obsid).update({
                        'filtered_red_noise': None,
                        'filtered_white_noise': None,
                        'filtered_auto_rms': None,
                        'filtered_noise_index': None,
                        'unfiltered_red_noise': None,
                        'unfiltered_white_noise': None,
                        'unfiltered_auto_rms': None,
                        'unfiltered_noise_index': None,
                        'n_spikes': None,
                        'n_nan_values': None,
                        'mean_atm_temp': None,
                        'comment': None,
                    })
                missing.append(obsid)

        if missing:
            self.session.commit()
        self._disconnect()
        return missing

    def remap_level2_paths(self, old_prefix: str, new_prefix: str,
                           dry_run: bool = True,
                           require_new_path_exists: bool = False) -> dict:
        """
        Bulk-update Level-2 paths when files have moved to a new disk.

        Args:
            old_prefix: Existing path prefix in the database.
            new_prefix: Replacement prefix.
            dry_run: If True, report changes without writing to the database.
            require_new_path_exists: If True, only update rows where the remapped
                path exists on disk.

        Returns:
            Summary dictionary with counters and a small sample of remapped entries.
        """
        old_prefix = old_prefix.rstrip('/')
        new_prefix = new_prefix.rstrip('/')

        self._connect()
        rows = (self.session.query(COMAPData.obsid, COMAPData.level2_path)
                .filter(COMAPData.level2_path.isnot(None))
                .filter(COMAPData.level2_path != "")
                .all())

        updated = 0
        skipped_missing = 0
        candidates = 0
        sample = []

        for obsid, old_path in rows:
            if not isinstance(old_path, str) or not old_path.startswith(old_prefix):
                continue

            candidates += 1
            suffix = old_path[len(old_prefix):]
            new_path = f"{new_prefix}{suffix}"

            if require_new_path_exists and (not os.path.exists(new_path)):
                skipped_missing += 1
                continue

            if not dry_run:
                self.session.query(COMAPData).filter_by(obsid=obsid).update({'level2_path': new_path})
            updated += 1

            if len(sample) < 10:
                sample.append({'obsid': obsid, 'old_path': old_path, 'new_path': new_path})

        if (not dry_run) and updated > 0:
            self.session.commit()

        self._disconnect()
        return {
            'dry_run': dry_run,
            'candidates': candidates,
            'updated': updated,
            'skipped_missing_new_path': skipped_missing,
            'sample': sample,
        }

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
    
    def get_files(self, source_group=None, source=None, min_obsid=0) -> list:
        """
        Get all files matching the given source/source_group/min_obsid criteria.
        """
        self._connect()
        query = self.session.query(COMAPData)
        if source_group:
            query = query.filter_by(source_group=source_group)
        if source:
            query = query.filter_by(source=source)

        query = query.filter(COMAPData.obsid >= min_obsid)

        a = query.all()
        self._disconnect()
        return a

    def get_unprocessed_files(self, source_group=None, source=None, min_obsid=0, overwrite=False) -> list:
        """Backward-compatible wrapper for get_files."""
        return self.get_files(source_group=source_group, source=source, min_obsid=min_obsid)

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

    def get_observation_snapshot(self, obsid: int) -> dict:
        """
        Return SQL-backed metadata for one observation, including quality flags.
        Useful for serializing into a Level-2 HDF5 file for portability.
        """
        self._connect()
        data = self.session.query(COMAPData).filter_by(obsid=obsid).first()
        if data is None:
            self._disconnect()
            return {}

        flags = (self.session.query(QualityFlag)
                .filter_by(obsid=obsid)
                .all())

        snapshot = {k: v for k, v in data.__dict__.items() if not k.startswith('_')}
        snapshot['quality_flags'] = [
            {
                'pixel': f.pixel,
                'frequency_band': f.frequency_band,
                'is_good': bool(f.is_good),
                'comment': f.comment,
                'filtered_red_noise': f.filtered_red_noise,
                'filtered_white_noise': f.filtered_white_noise,
                'filtered_auto_rms': f.filtered_auto_rms,
                'filtered_noise_index': f.filtered_noise_index,
                'unfiltered_red_noise': f.unfiltered_red_noise,
                'unfiltered_white_noise': f.unfiltered_white_noise,
                'unfiltered_auto_rms': f.unfiltered_auto_rms,
                'unfiltered_noise_index': f.unfiltered_noise_index,
                'n_spikes': f.n_spikes,
                'n_nan_values': f.n_nan_values,
                'mean_atm_temp': f.mean_atm_temp,
            }
            for f in flags
        ]

        summary = (self.session.query(ObservationSummary)
                   .filter_by(obsid=obsid).first())
        if summary:
            snapshot['summary'] = {
                'processing_status': summary.processing_status,
                'processing_error': summary.processing_error,
                'median_tsys': summary.median_tsys,
                'mean_tau': summary.mean_tau,
                'calibrator_flux': summary.calibrator_flux,
                'calibrator_flux_error': summary.calibrator_flux_error,
                'calibrator_chi2': summary.calibrator_chi2,
                'pointing_offset_az': summary.pointing_offset_az,
                'pointing_offset_el': summary.pointing_offset_el,
                'pointing_offset_ra': summary.pointing_offset_ra,
                'pointing_offset_dec': summary.pointing_offset_dec,
                'n_scans': summary.n_scans,
            }

        self._disconnect()
        return snapshot

    def query_scan_flags(self, obsid: int, pixel: int, frequency_band: int):
        """Return per-scan quality flags for a given obsid/pixel/band.

        Checks the observation-level is_good flag from the QualityFlag table.
        Per-scan noise statistics are stored in HDF5, not SQL, so fine-grained
        per-scan filtering is handled separately in ReadData._process_scan.

        Returns:
            None if the entry is good or does not exist (caller treats as all scans good).
            False if the entry exists and is_good=False (caller treats as all scans bad).
        """
        self._connect()
        flag = (self.session.query(QualityFlag)
                .filter_by(obsid=obsid, pixel=pixel, frequency_band=frequency_band)
                .first())
        self._disconnect()
        if flag is None or flag.is_good:
            return None
        return False

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

    def update_observation_summary(self, obsid: int, **kwargs) -> None:
        """
        Update the ObservationSummary for an observation (upsert).

        Accepts any ObservationSummary column as a keyword argument, e.g.:
            db.update_observation_summary(obsid, median_tsys=45.2, n_scans=12)
        """
        self._connect()
        summary = (self.session.query(ObservationSummary)
                   .filter_by(obsid=obsid).first())
        if summary is None:
            summary = ObservationSummary(obsid=obsid, **kwargs)
            self.session.add(summary)
        else:
            for key, value in kwargs.items():
                if hasattr(summary, key):
                    setattr(summary, key, value)
        self.session.commit()
        self._disconnect()

    def query_observation_summaries(self, min_obsid=0, max_obsid=1000000,
                                     source=None, source_group=None,
                                     processing_status=None) -> list[dict]:
        """
        Query ObservationSummary joined with COMAPData for filtering.

        Returns list of dicts with observation metadata + summary statistics.
        """
        self._connect()
        query = (self.session.query(COMAPData, ObservationSummary)
                 .outerjoin(ObservationSummary, COMAPData.obsid == ObservationSummary.obsid)
                 .filter(COMAPData.obsid >= min_obsid)
                 .filter(COMAPData.obsid <= max_obsid))
        if source:
            query = query.filter(COMAPData.source.contains(source))
        if source_group:
            query = query.filter(COMAPData.source_group == source_group)
        if processing_status:
            query = query.filter(ObservationSummary.processing_status == processing_status)

        results = []
        for obs, summary in query.all():
            row = {
                'obsid': obs.obsid,
                'source': obs.source,
                'source_group': obs.source_group,
                'level2_path': obs.level2_path,
                'utc_start': obs.utc_start,
            }
            if summary:
                row.update({
                    'processing_status': summary.processing_status,
                    'processing_error': summary.processing_error,
                    'median_tsys': summary.median_tsys,
                    'mean_tau': summary.mean_tau,
                    'calibrator_flux': summary.calibrator_flux,
                    'calibrator_flux_error': summary.calibrator_flux_error,
                    'calibrator_chi2': summary.calibrator_chi2,
                    'pointing_offset_az': summary.pointing_offset_az,
                    'pointing_offset_el': summary.pointing_offset_el,
                    'pointing_offset_ra': summary.pointing_offset_ra,
                    'pointing_offset_dec': summary.pointing_offset_dec,
                    'n_scans': summary.n_scans,
                })
            results.append(row)
        self._disconnect()
        return results

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

    # ---------- Calibration flux (per-element, per-observation) ----------

    def insert_calibration_flux(self, obsid: int, source: str, mjd: float,
                                flux, flux_errors, amplitudes, chi2) -> None:
        """Insert per-feed/band/channel flux measurements from calibrator fitting.

        Args:
            obsid: Observation ID
            source: Calibrator source name (e.g. 'TauA')
            mjd: Median MJD of the observation
            flux: ndarray [19, 4, 2] — flux density in Jy
            flux_errors: ndarray [19, 4, 2] — flux density errors in Jy
            amplitudes: ndarray [19, 4, 2] — peak amplitude in K
            chi2: ndarray [19, 4, 2] — reduced chi-squared
        """
        import numpy as np
        self._connect()
        nfeed, nband, nchan = flux.shape
        rows = []
        for ifeed in range(nfeed):
            for iband in range(nband):
                for ichan in range(nchan):
                    fval = float(flux[ifeed, iband, ichan])
                    if not np.isfinite(fval):
                        continue
                    rows.append(CalibrationFlux(
                        obsid=obsid,
                        feed=ifeed + 1,  # 1-indexed
                        band=iband,
                        channel=ichan,
                        mjd=mjd,
                        source=source,
                        measured_flux=fval,
                        flux_error=float(flux_errors[ifeed, iband, ichan]) if np.isfinite(flux_errors[ifeed, iband, ichan]) else None,
                        amplitude=float(amplitudes[ifeed, iband, ichan]) if np.isfinite(amplitudes[ifeed, iband, ichan]) else None,
                        chi2=float(chi2[ifeed, iband, ichan]) if np.isfinite(chi2[ifeed, iband, ichan]) else None,
                    ))

        # Upsert: delete existing rows for this obsid, then insert
        self.session.query(CalibrationFlux).filter_by(obsid=obsid).delete()
        self.session.add_all(rows)
        self.session.commit()
        self._disconnect()

    def query_calibration_flux(self, source: str,
                               min_obsid: int = 0,
                               max_obsid: int = 1000000) -> list:
        """Query per-element calibration flux measurements.

        Returns list of CalibrationFlux ORM objects.
        """
        self._connect()
        rows = (self.session.query(CalibrationFlux)
                .filter(CalibrationFlux.source == source,
                        CalibrationFlux.obsid >= min_obsid,
                        CalibrationFlux.obsid <= max_obsid)
                .order_by(CalibrationFlux.obsid)
                .all())
        # Detach from session before disconnect
        result = list(rows)
        self._disconnect()
        return result

    # ---------- Calibration model fit ----------

    def upsert_calibration_model(self, feed: int, band: int, channel: int,
                                 **kwargs) -> None:
        """Insert or update a calibration model fit for one feed/band/channel."""
        self._connect()
        row = (self.session.query(CalibrationModelFit)
               .filter_by(feed=feed, band=band, channel=channel)
               .first())
        if row is None:
            row = CalibrationModelFit(feed=feed, band=band, channel=channel, **kwargs)
            self.session.add(row)
        else:
            for key, value in kwargs.items():
                if hasattr(row, key):
                    setattr(row, key, value)
        self.session.commit()
        self._disconnect()

    def load_calibration_model(self) -> dict:
        """Load all calibration model fits from SQL and return a dict
        suitable for ReadData consumption.

        Returns dict with keys:
            "model_type": str
            "model_params": object array [19, 4, 2] of per-element param dicts
        """
        import numpy as np
        import json
        self._connect()
        rows = self.session.query(CalibrationModelFit).all()
        self._disconnect()

        if not rows:
            return None

        model_type = rows[0].model_type or "polynomial"
        model_params = np.empty((19, 4, 2), dtype=object)
        # Initialize all to None
        for i in range(19):
            for j in range(4):
                for k in range(2):
                    model_params[i, j, k] = None

        for row in rows:
            ifeed = row.feed - 1  # convert to 0-indexed
            params = {}
            if model_type in ("polynomial", "mean"):
                if row.poly_coeffs:
                    params["coeffs"] = json.loads(row.poly_coeffs)
                else:
                    params["coeffs"] = [1.0]  # fallback: unity gain
            elif model_type == "nearest":
                if row.nearest_mjds and row.nearest_gains:
                    params["mjds"] = np.array(json.loads(row.nearest_mjds))
                    params["gains"] = np.array(json.loads(row.nearest_gains))
            model_params[ifeed, row.band, row.channel] = params

        return {
            "model_type": model_type,
            "model_params": model_params,
        }

db = SQLModule()
