import os
import random
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing as pp
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pdb
from cf_il.generate_dirichlet import generate_dirichlet
#from datasets.seq_cifar10 import SequentialCIFAR10


def recovery_memory(
    current_task,
    model: nn.Module,
    num_classes: int,
    image_shape: Tuple[int, int, int],
    device: torch.device,
    scale: Tuple[float, float],
    buffer_size: int,
    psi: float,
    tau: float,
    optim_steps: int,
    optim_lr: float,
    synth_img_save_dir: Optional[str],
    synth_img_save_num: Optional[int],
    dirichlet_max_iter: int,
    task_information
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    precision_list=[]
    for key, value in task_information.items():
        if key.isdigit():
            precision_list.append((float(value['precision']), key))

    precision_list.sort(reverse=False)
    ada_num_images_per_class = [0] * num_classes
    if num_classes == 5:
        choose_list = [4,3,1,1,1]
    elif num_classes == 10:
        choose_list = [3,2,1,1,0.5,0.5,0.5,0.5,0.5,0.5]
    for i in range(num_classes):
        classId = int(precision_list[i][1])
        ada_num_images_per_class[classId] = buffer_size // num_classes * choose_list[i]
            
    num_images_per_class = buffer_size // num_classes
    #pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    fc_weights = model.last[2*current_task].detach().cpu().numpy()
    fc_weights = np.transpose(fc_weights)
    fc_weight_norm = pp.normalize(fc_weights, axis=0)

    sim_matrix  = np.matmul(np.transpose(fc_weight_norm), fc_weight_norm)
    with open('output.txt', 'a') as f:
        f.write(str(sim_matrix))
    images_all: List[torch.Tensor] = []
    logits_all: List[torch.Tensor] = []
    images_all_task = []
    logits_all_task = []
    for classes in range(num_classes):
        pseudo_labels = generate_dirichlet(
            batch_size=ada_num_images_per_class[classes],
            class_id=classes,
            scale=scale,
            similarity_matrix=sim_matrix,
            psi=psi,
            max_iter=dirichlet_max_iter,
        )
        rand_img = np.random.uniform(0, 1, size=(ada_num_images_per_class[classes], *reversed(image_shape)))
        rand_img = rand_img.astype(np.float32)
        rand_img_tensor = torch.tensor(
            rand_img,
            requires_grad=True,
            device=device,
        )

        opt = torch.optim.Adam(lr=optim_lr, params=[rand_img_tensor])

        for i in range(optim_steps): # 0, len(r), self.sbatch

            if i % 1500 == 0:
                print(f'Synth image optim step {i+1}/{optim_steps}')
            opt.zero_grad()
            logits = model(current_task, rand_img_tensor)
            logit = logits[0]
            loss = -criterion(logit[current_task] / tau, torch.tensor(pseudo_labels, device=device))
            loss.backward()
            opt.step()

        synth_logits = model(current_task, rand_img_tensor)[0][current_task].detach().cpu()
        synth_images = rand_img_tensor.detach().cpu()


        for i, l in zip(synth_images, synth_logits):
            images_all.append(i.reshape((1, *reversed(image_shape))))  # type: ignore
            logits_all.append(l.reshape((1, sim_matrix.shape[0])))

    return (np.array(images_all, dtype=torch.TensorType), np.array(logits_all, dtype=torch.TensorType))

def save_rand_images(
    images: torch.Tensor,
    cls_id: int,
    task: int,
    samples_num: int,
    dir: str,
) -> None:

    transform_denorm = DeNormalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
    images_ids = random.sample(range(len(images)), samples_num)
    for id in images_ids:
        img = transform_denorm(images[id])
        img = np.transpose(img.numpy(), (1, 2, 0))
        img_min = np.min(img, axis=(0, 1))
        img_max = np.max(img, axis=(0, 1))
        img = (img - img_min) / (img_max - img_min)

        save_time = datetime.now()
        save_time = datetime.strftime(save_time, "%Y_%m_%d_%H_%M_%S")  # type: ignore
        path = os.path.join(dir, f'synth_image_task-{task}_class-{cls_id}_{id}_{save_time}.png')

        plt.imsave(path, img)

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
