# app/core/logging.py
import logging

logger = logging.getLogger("mmi")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | mmi | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
