import numpy as np

from modules.SQLModule.SQLModule import COMAPData, db
from modules.pipeline_control.Pipeline import BaseCOMAPModule, RetryH5PY


class FinalizeLevel2Metadata(BaseCOMAPModule):
    """
    Optional post-processing step that embeds SQL metadata into the Level-2 file.

    This makes Level-2 files more self-contained by copying a snapshot of:
      - key COMAPData fields
      - per-feed / per-band quality flags
      - quality statistics and comments

    Add this module at the end of the ProcessLevel2 pipeline config.
    """

    def __init__(self, group_name: str = "level2/sql_snapshot", overwrite: bool = True) -> None:
        super().__init__()
        self.group_name = group_name
        self.overwrite = overwrite

    def run(self, file_info: COMAPData) -> None:
        if not file_info.level2_path:
            return

        snapshot = db.get_observation_snapshot(file_info.obsid)
        if not snapshot:
            return

        with RetryH5PY(file_info.level2_path, "a") as level2:
            if self.group_name in level2:
                if self.overwrite:
                    del level2[self.group_name]
                else:
                    return

            grp = level2.require_group(self.group_name)

            # Store scalar COMAPData metadata as attributes where possible.
            for key, value in snapshot.items():
                if key == "quality_flags":
                    continue
                if value is None:
                    continue
                try:
                    grp.attrs[key] = value
                except TypeError:
                    grp.attrs[key] = str(value)

            flags = snapshot.get("quality_flags", [])
            if not flags:
                return

            n = len(flags)
            pixels = np.zeros(n, dtype=np.int16)
            bands = np.zeros(n, dtype=np.int16)
            is_good = np.zeros(n, dtype=np.bool_)
            comments = np.array([f.get("comment") or "" for f in flags], dtype=h5_string_dtype())

            stats_keys = [
                "filtered_red_noise",
                "filtered_white_noise",
                "filtered_auto_rms",
                "filtered_noise_index",
                "unfiltered_red_noise",
                "unfiltered_white_noise",
                "unfiltered_auto_rms",
                "unfiltered_noise_index",
                "mean_atm_temp",
            ]
            int_keys = ["n_spikes", "n_nan_values"]

            stats = {k: np.full(n, np.nan, dtype=np.float64) for k in stats_keys}
            stats_int = {k: np.full(n, -1, dtype=np.int32) for k in int_keys}

            for i, f in enumerate(flags):
                pixels[i] = f["pixel"]
                bands[i] = f["frequency_band"]
                is_good[i] = bool(f.get("is_good", True))
                for k in stats_keys:
                    val = f.get(k, None)
                    if val is not None:
                        stats[k][i] = float(val)
                for k in int_keys:
                    val = f.get(k, None)
                    if val is not None:
                        stats_int[k][i] = int(val)

            grp.create_dataset("pixel", data=pixels)
            grp.create_dataset("frequency_band", data=bands)
            grp.create_dataset("is_good", data=is_good)
            grp.create_dataset("comment", data=comments)
            for k, arr in stats.items():
                grp.create_dataset(k, data=arr)
            for k, arr in stats_int.items():
                grp.create_dataset(k, data=arr)


def h5_string_dtype():
    import h5py

    return h5py.string_dtype(encoding="utf-8")
