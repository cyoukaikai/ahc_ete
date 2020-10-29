import os

# ~/tuat/ did not work
TrackingBenchmarkDataRootDir = '/home/kai/tuat/data/tracking_benchmark/'
MOT_DataRootDir = os.path.join(
    TrackingBenchmarkDataRootDir,
    'MOT'
)
TrackingBenchmarkDebugDir = os.path.join(
    TrackingBenchmarkDataRootDir,
    'cluster'
)