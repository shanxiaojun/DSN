import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pdb
from torch.autograd import Variable
from . import Metric, classification_accuracy
from .prune import SparsePruner
from .metrics import fv_evaluate
from src.networks import layers as nl


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, task,shared_layer_info, masks, xtrain, ytrain, xvalid, yvalid, begin_prune_step, end_prune_step):
        self.args = args
        self.sbatch = 64
        self.model = model
        self.shared_layer_info = shared_layer_info
        # self.inference_dataset_idx = self.model.datasets.index(args.dataset) + 1
        self.inference_dataset_idx = task
        self.pruner = SparsePruner(self.model, masks, task, self.args, begin_prune_step, end_prune_step, self.inference_dataset_idx)
        self.clipgrad = 10000
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid
        self.criterion = nn.CrossEntropyLoss()
        return

    def train(self, optimizers, epoch_idx, curr_lrs, curr_prune_step,x,y,task):
        # Set model to training mode
        self.model.cuda()
        self.model.train()
        '''train_loss     = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')'''
        r = np.arange(self.xtrain.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)
            # Forward
            outputs = self.model.forward(images)
            loss = self.criterion(outputs, targets)
            # Backward
            optimizers.zero_grad()
            loss.backward()
            #torch.nn.utils1.clip_grad_norm(self.model.parameters(), self.clipgrad)
            # Set fixed param grads to 0.

            self.pruner.do_weight_decay_and_make_grads_zero()

            # Gradient is applied across all ranks
            optimizers.step()

            # Set pruned weights to 0.
            if self.args.mode == 'prune':
                self.pruner.gradually_prune(curr_prune_step)
                curr_prune_step += 1



    #{{{ Evaluate classification
    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(self.val_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    if self.args.cuda:
                        data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    num = data.size(0)
                    val_loss.update(self.criterion(output, target), num)
                    val_accuracy.update(classification_accuracy(output, target), num)

                    if self.inference_dataset_idx == 1:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio(),
                                       'mpl': self.args.network_width_multiplier})
                    else:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                       'shared_ratio': self.pruner.calculate_shared_part_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio(),
                                       'mpl': self.args.network_width_multiplier})
                    t.update(1)

        summary = {'loss': '{:.3f}'.format(val_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(self.pruner.calculate_curr_task_ratio()),
                   'zero ratio': '{:.3f}'.format(self.pruner.calculate_zero_ratio()),
                   'mpl': self.args.network_width_multiplier}
        if self.inference_dataset_idx != 1:
            summary['shared_ratio'] = '{:.3f}'.format(self.pruner.calculate_shared_part_ratio())

        if self.args.log_path:
            logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
                         + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return val_accuracy.avg.item()
    #}}}

    #{{{ Evaluate LFW
    def evalLFW(self, epoch_idx):
        distance_metric = True
        subtract_mean   = False
        self.pruner.apply_mask()
        self.model.eval() # switch to evaluate mode
        labels, embedding_list_a, embedding_list_b = [], [], []
        with torch.no_grad():
            with tqdm(total=len(self.val_loader),
                      desc='Validate Epoch  #{}: '.format(epoch_idx + 1),
                      ascii=True) as t:
                for batch_idx, (data_a, data_p, label) in enumerate(self.val_loader):
                    data_a, data_p = data_a.cuda(), data_p.cuda()
                    data_a, data_p, label = Variable(data_a, volatile=True), \
                                            Variable(data_p, volatile=True), Variable(label)
                    # ==== compute output ====
                    out_a = self.model.forward_to_embeddings(data_a)
                    out_p = self.model.forward_to_embeddings(data_p)
                    # do L2 normalization for features
                    if not distance_metric:
                        out_a = F.normalize(out_a, p=2, dim=1)
                        out_p = F.normalize(out_p, p=2, dim=1)
                    out_a = out_a.data.cpu().numpy()
                    out_p = out_p.data.cpu().numpy()

                    embedding_list_a.append(out_a)
                    embedding_list_b.append(out_p)
                    # ========================
                    labels.append(label.data.cpu().numpy())
                    t.update(1)

        labels = np.array([sublabel for label in labels for sublabel in label])
        embedding_list_a = np.array([item for embedding in embedding_list_a for item in embedding])
        embedding_list_b = np.array([item for embedding in embedding_list_b for item in embedding])
        tpr, fpr, accuracy, val, val_std, far = fv_evaluate(embedding_list_a, embedding_list_b, labels,
                                                distance_metric=distance_metric, subtract_mean=subtract_mean)
        print('In evalLFW(): Test set: Accuracy: {:.5f}+-{:.5f}'.format(np.mean(accuracy),np.std(accuracy)))
        logging.info(('In evalLFW()-> Validate Epoch #{} '.format(epoch_idx + 1)
                     + 'Test set: Accuracy: {:.5f}+-{:.5f}, '.format(np.mean(accuracy),np.std(accuracy))
                     + 'task_ratio: {:.2f}'.format(self.pruner.calculate_curr_task_ratio())))
        return np.mean(accuracy)
    #}}}

    def save_checkpoint(self, optimizers, epoch_idx, save_folder,t):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        for name, module in self.model.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                if module.bias is not None:
                    self.shared_layer_info[t][
                        'bias'][name] = module.bias
                if module.piggymask is not None:
                    self.shared_layer_info[t][
                        'piggymask'][name] = module.piggymask
            elif isinstance(module, nn.BatchNorm2d):
                self.shared_layer_info[t][
                    'bn_layer_running_mean'][name] = module.running_mean
                self.shared_layer_info[t][
                    'bn_layer_running_var'][name] = module.running_var
                self.shared_layer_info[t][
                    'bn_layer_weight'][name] = module.weight
                self.shared_layer_info[t][
                    'bn_layer_bias'][name] = module.bias
            elif isinstance(module, nn.PReLU):
                self.shared_layer_info[t][
                    'prelu_layer_weight'][name] = module.weight

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'dataset_history': self.model.datasets,
            'dataset2num_classes': self.model.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info
        }
        torch.save(checkpoint, filepath)
        return

    def load_checkpoint(self, optimizers, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.state_dict()

            for name, param in state_dict.items():
                if ('piggymask' in name or name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                    # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name][:param.size(0), :param.size(1), :, :].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn and prelu layer
                    curr_model_state_dict[name][:param.size(0)].copy_(param)
                elif 'classifiers' in name:
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                else:
                    try:
                        curr_model_state_dict[name].copy_(param)
                    except:
                        pdb.set_trace()
                        print("There is some corner case that we haven't tackled")
        return

    def load_checkpoint_only_for_evaluate(self, resume_from_epoch, save_folder,t):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.state_dict()

            for name, param in state_dict.items():
                if 'piggymask' in name: # we load piggymask value in main.py
                    continue

                if (name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue

                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name].copy_(
                            param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1), :, :])

                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name].copy_(
                            param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1)])

                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn and prelu layer
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0)])

                else:
                    curr_model_state_dict[name].copy_(param)

            # load the batch norm params and bias in convolution in correspond to curr dataset
            for name, module in self.model.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[t]['bias'][name]

                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[t][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[t][
                        'bn_layer_running_var'][name]
                    module.weight = self.shared_layer_info[t][
                        'bn_layer_weight'][name]
                    module.bias = self.shared_layer_info[t][
                        'bn_layer_bias'][name]

                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[t][
                        'prelu_layer_weight'][name]
        return


    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True).cuda()
            targets=torch.autograd.Variable(y[b],volatile=True).cuda()

            # Forward
            output=self.model.forward(images)
            #output=outputs[t]

            loss=self.criterion(output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num


