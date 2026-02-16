# CreateMapPerFeed.py
#
# Backwards-compatible wrapper around the unified CreateMap class.

from modules.MapMaking.MapMakingModules.CreateMap import CreateMap


class CreateMapPerFeed(CreateMap):
    def __init__(self, *args, **kwargs):
        kwargs["split_mode"] = "per_feed"
        super().__init__(*args, **kwargs)
