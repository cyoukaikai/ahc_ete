import os
import smrc.utils

# this setting is fine
# TrackingBenchmarkDataRootDir = '/home/kai/tuat/release/AHC_ETE'

TrackingBenchmarkDataRootDir = smrc.utils.dir_name_up_n_levels(
    os.path.abspath(__file__), 4
)

MOT_DataRootDir = os.path.join(
    TrackingBenchmarkDataRootDir,
    'MOTChallenge'
)
TrackingBenchmarkDebugDir = os.path.join(
    TrackingBenchmarkDataRootDir,
    'cluster'
)