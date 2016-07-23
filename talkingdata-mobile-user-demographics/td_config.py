# -*- coding: utf-8 -*-
"""
Common configurations.
"""

import logging
import yaml


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("td.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

with open("talkingdata.yml", 'r') as f:
    cfg = yaml.load(f)
