import logging
from run import run

LOG_FORMAT = '%(asctime)s %(levelname)s - %(module)s.%(funcName)s() : %(message)s'
logging.basicConfig(filename='resources/results/info.log', level=logging.INFO, format=LOG_FORMAT)

run(10, 10, 1, 0.25, 0.25, 1)
