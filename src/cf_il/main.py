from argparse import Namespace
import os

import hydra
import torch
import torch.nn as nn

from cf_il.model import CFIL
from cf_il.train import train
from cf_il.conf.constants import ROOT_DIR
from cf_il.conf.config import Config
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset


@hydra.main(config_path=os.path.join(ROOT_DIR, "conf"), config_name="config")
def main(config: Config) -> CFIL:
    """Main function for calling training of the model."""
    args: Namespace = config.cf_il  # type: ignore
    dataset = get_dataset(args)
    assert isinstance(dataset, ContinualDataset) is True
    assert dataset.N_TASKS is not None
    assert dataset.N_CLASSES_PER_TASK is not None
    num_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    image_shape = None
    if dataset.NAME == 'seq-cifar10':
        image_shape = (32, 32, 3)
    else:
        raise ValueError('Image shape cannot be None.')

    # Load model
    backbone: torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    backbone.eval()
    backbone.fc = nn.Linear(512, num_classes)

    model = CFIL(
        backbone=backbone,
        args=args,
        loss=nn.CrossEntropyLoss(),
        transform=dataset.get_transform(),
        image_shape=image_shape,
    )

    train(
        model=model,
        dataset=dataset,
        args=args,
    )

    return model


if __name__ == '__main__':
    main()
