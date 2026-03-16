# CreateMapDayNight.py
#
# Backwards-compatible wrapper around the unified CreateMap class.

from modules.MapMaking.MapMakingModules.CreateMap import CreateMap


class CreateMapDayNight(CreateMap):
    def __init__(self, *args, **kwargs):
        kwargs["split_mode"] = "day_night"
        super().__init__(*args, **kwargs)
