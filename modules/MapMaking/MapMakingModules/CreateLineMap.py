# CreateLineMap.py
#
# Backwards-compatible wrapper around the unified CreateMap class for line maps.

from modules.MapMaking.MapMakingModules.CreateMap import CreateMap


class CreateLineMap(CreateMap):
    def __init__(self, *args, **kwargs):
        kwargs["line_mode"] = True
        super().__init__(*args, **kwargs)
