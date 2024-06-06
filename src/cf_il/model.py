import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from cf_il.recover_memory import recovery_memory as rec_mem
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer


class CFIL(ContinualModel):
    NAME: str = 'cf-il'  # type: ignore[assignment]
    COMPATIBILITY = [
        'class-il',
    ]

    __image_shape: Tuple[int, int, int]

    psi: float
    tau: float
    scale: Tuple[float, float]
    optim_steps: int
    optim_lr: float
    synth_img_save_dir: Optional[str]
    synth_img_save_num: Optional[int]
    dirichlet_max_iter: int

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        args: Namespace,
        transform: torchvision.transforms,
        image_shape: Tuple[int, int, int],
    ):
        super(CFIL, self).__init__(backbone, loss, args, transform)

        self.__image_shape = image_shape

        self.psi = args.psi
        self.tau = args.tau
        self.scale = args.scale
        self.optim_steps = args.optim_steps
        self.optim_lr = args.optim_lr
        self.synth_img_save_num = args.synth_img_save_num
        self.dirichlet_max_iter = args.dirichlet_max_iter

        if args.synth_img_save_dir:
            print(f'Creating dir if does not exist: {args.synth_img_save_dir}')
            wd_path = Path(__file__).parent.resolve()
            path = os.path.join(wd_path, '..', args.synth_img_save_dir)
            Path(path).mkdir(parents=True, exist_ok=True)
            self.synth_img_save_dir = path

        # Buffer contains synthetic images from RMP
        self.buffer = Buffer(
            buffer_size=self.args.buffer_size,
            device=self.device,
            mode='reservoir',
        )

        self.opt = optim.SGD(
            backbone.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
        )

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Execute training step of backbone network on real data and collected data impressions.

        Args:
            inputs (torch.Tensor): Input data (real).
            labels (torch.Tensor): Labels (real).

        Returns:
            float: Total loss
        """

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        print(f'Real dataset loss: {loss}')

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(self.args.buffer_batch_size)
            buf_outputs = self.net(buf_inputs)
            synth_loss = F.mse_loss(buf_outputs, buf_logits)
            print(f'Synthetic dataset loss: {synth_loss}')
            loss += self.args.alpha * synth_loss

        print(f'Total loss: {loss}')

        loss.backward()
        self.opt.step()

        return loss.item()  # type: ignore[no-any-return]

    def recover_memory(
        self,
        num_classes: int,
    ) -> None:
        """
        Run recovery memory.

        Args:
            num_classes (int): Number of classes which has already been learn by the model.
        """
        net_training_status = self.net.training
        self.net.eval()

        self.buffer.empty()

        synth_images, synth_logits = rec_mem(
            model=self.net,
            num_classes=num_classes,
            buffer_size=self.buffer.buffer_size,
            image_shape=self.__image_shape,
            device=self.device,
            scale=self.scale,
            psi=self.psi,
            tau=self.tau,
            optim_steps=self.optim_steps,
            optim_lr=self.optim_lr,
            synth_img_save_dir=self.synth_img_save_dir,
            synth_img_save_num=self.synth_img_save_num,
            dirichlet_max_iter=self.dirichlet_max_iter,
        )

        for img, logits in zip(synth_images, synth_logits):
            self.buffer.add_data(examples=img, logits=logits)

        self.net.train(net_training_status)
