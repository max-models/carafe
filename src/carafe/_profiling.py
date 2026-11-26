"""The scope-profiler tag of carafe's profiling regions.

The tag is registered as opt-in in `carafe`, so the regions record nothing
unless an application enables it (``scope_profiler.setup(enable_tags=["carafe"])``,
``SCOPE_PROFILER_ENABLE_TAGS=carafe`` or ``scope-profiler run --enable-tag
carafe``). Only coarse operations (solves, time steps, runs) have a region: a
disabled region still costs about 0.1 us per call.
"""

_PROFILE_TAGS = ("carafe",)
