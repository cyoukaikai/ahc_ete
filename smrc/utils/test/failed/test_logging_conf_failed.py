import logging
import logging.config
import os

# failed for the following two cases:
# kai@kai:~/tuat$ python smrc/line/tests/test_logging_conf.py
# kai@kai:~/tuat/smrc/line/tests$ python test_logging_conf.py

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'logging.conf'
)
print(log_file_path)
# print(f'{os.path.abspath("logging.conf")}')
# # /home/kai/Mask_RCNN/smrc/line/tests/logging.conf
# # /home/kai/tuat/smrc/line/tests/logging.conf

# log_file_path = "logging.conf"
assert os.path.isfile(log_file_path)
logging.config.fileConfig(log_file_path)

logger = logging.getLogger('simpleExample')

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')