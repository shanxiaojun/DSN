from argparse import Namespace
from typing import List
import os
import numpy as np

import sys
curPath = os.path.abspath(os.path.dirname(""))
# 获取上一级文件路径
rootPath = os.path.split(curPath)[0]
# 加入到搜索路径中
sys.path.append(rootPath)
from cf_il.model import CFIL
from datasets.utils.continual_dataset import ContinualDataset
from utils.loggers import print_mean_accuracy
from utils.status import progress_bar, create_stash
from utils.tb_logger import TensorboardLogger
from utils.training import evaluate


def train(
    model: CFIL,
    dataset: ContinualDataset,
    args: Namespace,
) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    assert dataset.N_TASKS is not None
    assert dataset.N_CLASSES_PER_TASK is not None

    # Assure buffer size is sufficient
    assert args.buffer_size >= dataset.N_CLASSES_PER_TASK

    model.net.to(model.device)

    num_tasks: int = dataset.N_TASKS
    num_classes_per_task: int = dataset.N_CLASSES_PER_TASK
    results: List[List[float]] = []
    results_mask_classes: List[List[float]] = []

    model_stash = create_stash(model, args, dataset)

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    for task_idx in range(num_tasks):
        model.net.train()
        train_loader, _ = dataset.get_data_loaders()

        if task_idx > 0:
            # Recover past experience
            current_num_of_trained_classes = task_idx * num_classes_per_task
            model.recover_memory(num_classes=current_num_of_trained_classes,)

            # Eval current model
            accs = evaluate(model, dataset, last=True)
            results[task_idx - 1] = results[task_idx - 1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[task_idx - 1] = results_mask_classes[task_idx - 1] + accs[1]

        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, task_idx, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, task_idx, i)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = task_idx + 1
        model_stash['epoch_idx'] = 0

        # Finish training
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, task_idx + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, task_idx)
