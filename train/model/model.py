import torch
import numpy as np
import segmentation_models_pytorch as smp


def create_model(encoder, encoder_weights, num_classes, activation):
    """
    Create a model with the specified encoder and weights.
    :param encoder: the encoder name
    :param encoder_weights: the encoder weights
    :param num_classes: the number of classes you should consider background as well
    :param activation: the activation function
    :return: the model

    """
# create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        classes=num_classes,
        activation=activation,
        # aux_params={'dropout':0.5}
    )
    
    return model

def prep(encoder, encoder_weights):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    return preprocessing_fn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
