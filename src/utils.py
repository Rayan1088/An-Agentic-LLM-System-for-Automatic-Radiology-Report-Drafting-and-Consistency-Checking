import sys
import yaml
import torch
from typing import Any
from src.logger import logging
from torch.nn import functional as F
from src.exception import CustomException
logger = logging.getLogger("config_load.py")

def load_config(config_path):
    try:
        with open(config_path, 'r') as configfile:
            config = yaml.safe_load(configfile)
        return config
    except Exception as e:
        logger.error(f"Error Occurred During Loading Configuration File: {e}")
        raise CustomException(e, sys)

def normalize_embeddings(embeddings: torch.Tensor):
        ### L2-normalize the embeddings
        # CLIP compares vectors using cosine similarity.
        # Cosine similarity assumes vectors have unit length.
        # Normalization makes comparisons easier and more stable.
        # Meaning every embedding vector has length exactly 1. 
        return F.normalize(embeddings, dim=-1)
    




