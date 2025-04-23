from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def contextual_negopredict(utterance):
    model_args = ClassificationArgs()

    # Create a ClassificationModel
    model = ClassificationModel(
        'roberta',
        'PATH TO LOAD THE SAVED MODEL',
        num_labels=8,
        args=model_args, use_cuda=False
    ) 
    predictions, raw_outputs = model.predict(utterance)
    predicted_probabilities = raw_outputs
    return predicted_probabilities

def future_negopredict(utterance):
    model_args = ClassificationArgs()

    # Create a ClassificationModel
    model = ClassificationModel(
        'roberta',
        'PATH TO LOAD THE SAVED MODEL',
        num_labels=8,
        args=model_args, use_cuda=False
    ) 
    predictions, raw_outputs = model.predict(utterance)
    predicted_probabilities = raw_outputs
    return predicted_probabilities