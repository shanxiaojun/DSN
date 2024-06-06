import sys, time
import numpy as np
import sklearn
from sklearn.metrics import classification_report
import torch
import math
import os
from torch.nn import functional as F
curPath = os.path.abspath(os.path.dirname(""))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src import utils
import pdb


########################################################################################################################

class Appr(object):

    def __init__(self, model, mask_pre, old_mask, nepochs, sbatch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000, eta=0.75, betamax=400, args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()

        self.eta = eta  
        self.betamax = betamax  
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.eta = float(params[0])
            self.betamax = float(params[1])

        self.mask_pre = mask_pre
        self.old_mask = old_mask
        self.mask_back = None

        self.optimizer = self._get_optimizer()

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_masks(self, net, t):
        old_mask = []
        for i in range(t + 1):
            masks = net.mask(i, s=400)
            masks_ = []
            for m in masks:
                masks_.append(m.detach().clone())

            for n in range(len(masks_)):
                masks_[n] = torch.round(masks_[n])
            masks = masks_
 
            if i == 0:
                prev = masks
            else:
                prev_ = []
                for m1, m2 in zip(prev, masks):
                    prev_.append(torch.max(m1, m2))
                prev = prev_
            old_mask.append(masks)
        old_maskpre = prev
        return old_mask, old_maskpre

    def train(self, appr,t, xtrain, ytrain, xvalid, yvalid, perclass= False):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        jingdu = []

        try:
            for e in range(self.nepochs):

                if e == self.nepochs - 1:
                    perclass = True
                clock0 = time.time()
                self.train_epoch(t, xtrain, ytrain)
                torch.save(appr.model, '..\parameter.pkl')
                net_temp = torch.load('..\parameter.pkl')
                _, old_maskpre = self.get_masks(net_temp, t)

                clock1 = time.time()
                _, train_loss, train_acc, _ = self.eval(t, xtrain, ytrain, perclass)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                            1000 * self.sbatch * (
                                                                                                                    clock1 - clock0) / xtrain.size(
                                                                                                                0),
                                                                                                            1000 * self.sbatch * (
                                                                                                                    clock2 - clock1) / xtrain.size(
                                                                                                                0),
                                                                                                            train_loss,
                                                                                                            100 * train_acc),
                      end='')
                # Valid
                _, valid_loss, valid_acc, perclass_information = self.eval(t, xvalid, yvalid, perclass)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = utils.get_model(self.model)
                    patience = self.lr_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= self.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < self.lr_min:
                            print()
                            break
                        patience = self.lr_patience
                        self.optimizer = self._get_optimizer(lr)
                print()

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        self.model = utils.set_model_(self.model, best_model)
        return jingdu, perclass_information

    def epoch(self, t, xtrain, ytrain, xvalid, yvalid, e, hp, xbuf=None, ybuf=None):

        clock0 = time.time()
        self.train_epoch(t, xtrain, ytrain, xbuf, ybuf, hp)
        clock1 = time.time()
        _, train_loss, train_acc = self.eval(t, xtrain, ytrain)
        clock2 = time.time()
        print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / xtrain.size(
                                                                                                        0),
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / xtrain.size(
                                                                                                        0),
                                                                                                    train_loss,
                                                                                                    100 * train_acc),
              end='')
        # Valid
        _, valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
        # Adapt lr
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_model = utils.get_model(self.model, 'bm')
            print(' *', end='')
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.lr /= self.lr_factor
                print(' lr={:.1e}'.format(self.lr), end='')
                if self.lr < self.lr_min:
                    print('lr too low!')
                    return None
                self.patience = self.lr_patience
        print()
        return 0

    def create_mask_back(self):
        # Weights mask
        self.mask_back = {}
        for n, _ in self.model.named_parameters():
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals
        

    def finish(self, t, xtrain, ytrain, xvalid, yvalid, clock0, e):

        clock1 = time.time()
        _, train_loss, train_acc = self.eval(t, xtrain, ytrain)
        clock2 = time.time()
        print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / xtrain.size(
                                                                                                        0),
                                                                                                    1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / xtrain.size(
                                                                                                        0),
                                                                                                    train_loss,
                                                                                                    100 * train_acc),
              end='')
        # Valid
        _, valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
        # Adapt lr
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_model = utils.get_model(self.model)
            self.patience = self.lr_patience
            print(' *', end='')
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.lr /= self.lr_factor
                print(' lr={:.1e}'.format(self.lr), end='')
                self.patience = self.lr_patience
                self.optimizer = self._get_optimizer(self.lr)

        return

    def train_epoch(self, t, x, y, similarity = 0, thres_cosh=50, thres_emb=6):
        self.model.train()

        r = np.arange(x.size(0))

        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)
  
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)
            s = (self.betamax - 1 / self.betamax) * i / len(r) + 1 / self.betamax

            outputs, masks = self.model.forward(task, images, s=s)
            output = outputs[t]
            loss, _ = self.criterion(output, targets, masks)

            loss = loss + 0.3 * similarity

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

           
            if t > 0:
                for n, p in self.model.named_parameters():
                    if n in self.mask_back and p.grad is not None:
                        p.grad.data *= self.mask_back[n]


            for n, p in self.model.named_parameters():
                if n.startswith('e'):

                    num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad.data *= self.betamax / s * num / den
                    

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()


            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data = torch.clamp(p.data, -thres_emb, thres_emb)
        return 


    def generation_transfer(self, t, ne_max, valid_acc, taskWithSimilarity, args, task_information, thres_cosh=50, thres_emb=6, ):

        best_acc = valid_acc
        taskWithSimilarity_sorted = dict(sorted(taskWithSimilarity.items(),
                                          key=lambda x: x[1], reverse=True))
        task_list = list(taskWithSimilarity_sorted.keys())

        for n, p in self.model.named_parameters():
            # print(n)
            if n.startswith('e') or n.startswith('l'):
                p.requires_grad = True
            else:
                p.requires_grad = True

        lr = 0.05
        op = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        current_num_of_trained_classes = t*20

        for t_ in task_list:

            
            self.best_model = utils.get_model(self.model, 'bm')
            count = 0 
            for n, p in self.model.named_parameters():
                if n.startswith('e'):

                    p.data[t_] = p.data[t_] * (1 - (
                            (1 - self.old_mask[t_][count]) * self.mask_pre[count])).round() + torch.nn.init.normal_(
                        torch.FloatTensor(p.data[t_].shape), mean=0, std=1).cuda() * (
                                         (1 - self.old_mask[t_][count]) * self.mask_pre[count]).round()

                    count += 1
                elif n.startswith('l'):
                    if float(n[5:]) / 2 == t_:  
                        m = torch.unsqueeze(((1 - self.old_mask[t_][-1]) * self.mask_pre[-1]).round(), 0).expand_as(p)
                        reinit = torch.empty_like(p.data)
                        torch.nn.init.kaiming_uniform_(reinit, a=math.sqrt(5))
                        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(reinit)
                        p.data = p.data * (1 - m) + reinit.cuda() * m
                    elif float(n[5:]) / 2 == t_ + 0.5:  
                        b_back = p.data.detach().clone()
                        b = torch.empty_like(p.data)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        torch.nn.init.uniform_(b, -bound, bound)
                        p.data = b

            print('t{}:'.format(t_), end='')
            patience = 5
            v_best = np.inf
            buffer = self.model.recover_memory(num_classes=5, modelmain=self.model,current_task=t_,task_information=task_information[t_])
            example = buffer.examples
            logits = buffer.logits

            task = torch.autograd.Variable(torch.LongTensor([t_]), requires_grad=False).cuda()

            for e in range(ne_max):
                self.model.train()

                d = np.arange(1000)
                np.random.shuffle(d)
                d = torch.LongTensor(d).cuda()
                loss = 0
                for i in range(0, len(d), self.sbatch):
                    if i + self.sbatch <= len(d):
                        b = d[i:i + self.sbatch]
                    else:
                        b = d[i:]
                 
                   
                    buf_inputs = torch.autograd.Variable(example[b], requires_grad=False)

                    buf_outputs, _ = self.model.forward(task, buf_inputs)
                  
                    buf_outputs = buf_outputs[t_]
                    buf_logits = torch.autograd.Variable(logits[b], requires_grad=False)

                    loss = 0.3 * F.mse_loss(buf_outputs, buf_logits)

                    op.zero_grad()
                    loss.backward()

                    s = (self.betamax - 1 / self.betamax) * i / len(example) + 1 / self.betamax

                    j = 0
                  
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            if n.startswith('e'):

                                p.grad.data *= ((1 - self.old_mask[t_][j]) * self.mask_pre[j]).round().expand_as(p)

                                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                                den = torch.cosh(p.data) + 1
                                p.grad.data *= self.betamax / s * num / den
                                j += 1
                            elif n.startswith('l'):
                                if float(n[5:]) / 2 == t_: 
                                    m = torch.unsqueeze(((1 - self.old_mask[t_][-1]) * self.mask_pre[-1]).round(),
                                                        0).expand_as(p)
                                    p.grad.data *= m

                    op.step()
                    for n, p in self.model.named_parameters():
                        if n.startswith('e'):
                            p.data = torch.clamp(p.data, -thres_emb, thres_emb)
                
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    if lr < self.lr_min:
                        break
                    for n, p in self.model.named_parameters():
                        if n.startswith('e') or n.startswith('l'):
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                    op = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
                    patience = 3

            self.model = utils.set_model_(0, self.best_model)

            if v_best >= best_acc[t_]:
                print('-' * 5, v_best, '>=', best_acc[t_], ',step back')

                count = 0

                for n, p in self.model.named_parameters():
                    if n.startswith('e'):
                        p.data[t_] = p.data[t_] * (1 - (
                                (1 - self.old_mask[t_][count]) * self.mask_pre[count])).round() + torch.full_like(
                            p.data[t_],
                            -6).cuda() * ((1 - self.old_mask[t_][count]) * self.mask_pre[count]).round()
                        count += 1
                    elif n.startswith('l'):
                        if float(n[5:]) / 2 == t_:  
                            m = torch.unsqueeze(((1 - self.old_mask[t_][-1]) * self.mask_pre[-1]).round(), 0).expand_as(
                                p)
                            p.data = p.data * (1 - m)
                        elif float(n[5:]) / 2 == t_ + 0.5:
                            p.data = b_back

                self.best_model = utils.get_model(self.model, 'bm')
                break
            else:
                print('update')
                best_acc[t_] = v_best
                self.model = utils.set_model_(0, self.best_model)


        for n, p in self.model.named_parameters():
            p.requires_grad = True
        return best_acc


    def eval(self, t, x, y, perclass):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        total_reg = 0

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=True)

            # Forward
            outputs, masks = self.model.forward(task, images, s=self.betamax)
            output = outputs[t]
            loss, reg = self.criterion(output, targets, masks)
            _, pred = output.max(1)
            hits = (pred == targets).float()
            if i == 0:
                preds = pred
            else:
                preds = torch.cat((preds,pred),0)

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)
            total_reg += reg.data.cpu().numpy().item() * len(b)

        if perclass:
            print(classification_report(y.cpu(), preds.cpu(), labels=None, target_names=None, sample_weight=None,
                                                  digits=3, output_dict=False))
            perclass_information = classification_report(y.cpu(), preds.cpu(), labels=None, target_names=None, sample_weight=None,
                                                  digits=3, output_dict=True)
        else:
            perclass_information = None
        return total_reg / total_num, total_loss / total_num, total_acc / total_num, perclass_information

    def eval_(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=True)

            outputs, masks = self.model.forward(task, images, s=self.betamax)
            output = outputs[t]
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)


        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count

        return self.ce(outputs, targets) + self.eta * reg, reg

    def js_div(self, p_logits, q_logits, get_softmax=True):

        KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = torch.nn.functional.softmax(p_logits)
            q_output = torch.nn.functional.softmax(q_logits)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

########################################################################################################################


