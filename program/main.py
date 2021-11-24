import logging
from run import run

LOG_FORMAT = '%(asctime)s %(levelname)s - %(module)s.%(funcName)s() : %(message)s'
logging.basicConfig(filename='resources/results/info.log', level=logging.INFO, format=LOG_FORMAT)

run(1500, 1500, 32, 0.25, 0, 0)
run(1500, 1500, 32, 0.25, 0.25, 1)
run(1500, 1500, 32, 0.25, 0.25, 2)
run(1500, 1500, 32, 0.25, 0.25, 3)
