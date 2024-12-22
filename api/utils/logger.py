# api/utils/logger.py
import logging
from datetime import datetime

logger = logging.getLogger("ppfl")
logger.setLevel(logging.INFO)

handler = logging.FileHandler("ppfl.log")
handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(handler)