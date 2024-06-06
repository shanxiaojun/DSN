import math
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')
import random
import sys, os, argparse, time
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch import optim

torch.set_printoptions(profile='full')
import utils
import copy


tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('--eta', type=int, default=0.75)
parser.add_argument('--a', type=int, default=0.25)
parser.add_argument('--b', type=int, default=1)
parser.add_argument('--c', type=int, default=1)
parser.add_argument('--d', type=int, default=0.75)
parser.add_argument('--n', type=int, default=10)
parser.add_argument('--betamax', type=int, default=200)
parser.add_argument('--penalty', type=float, default=0.0001, help="the type of bendmark")
parser.add_argument('--actions_num', type=int, default=1, help="how many actions to decide")
parser.add_argument('--hidden_size', type=int, default=100, help="the hidden size of RNN")
parser.add_argument('--num_layers', type=int, default=2, help="the layer of a RNN cell")
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='cifar100', type=str, required=False,
                    choices=['cifar100'],
                    help='(default=%(default)s)')
parser.add_argument('--approach', default='dsn', type=str, required=False,
                    choices=['dsn'], help='(default=%(default)s)')
parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--nepochs', default=50, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--eblation', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--buffer_size', type=int, default=1000, help='(default=%(default)s)')
parser.add_argument('--buffer_batch_size', type=int, default=32, help='(default=%(default)s)')
parser.add_argument('--alpha', type=int, default=0.3, help='(default=%(default)s)')
parser.add_argument('--psi', type=int, default=0.45, help='(default=%(default)s)')
parser.add_argument('--tau', type=int, default=20, help='(default=%(default)s)')
parser.add_argument('--scale', type=tuple, default=[1.0, 0.1], help='(default=%(default)s)')
parser.add_argument('--optim_steps', type=int, default=1500, help='(default=%(default)s)')
parser.add_argument('--optim_lr', type=int, default=0.01, help='(default=%(default)s)')
parser.add_argument('--synth_img_save_dir', type=str, default="./systh_img", help='(default=%(default)s)')
parser.add_argument('--dirichlet_max_iter', type=int, default=1000000, help='(default=%(default)s)')
parser.add_argument('--synth_img_save_num', type=int, default=0, help='(default=%(default)s)')

args = parser.parse_args()
if args.output == '':
    args.output = '../res/' + args.experiment + '_' + args.approach + '_' + str(args.seed) + '.txt'
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:

    print('[CUDA unavailable]')
    sys.exit()

# Args -- Experiment
if args.experiment == 'cifar100':
    from dataloaders import cifar100 as dataloader


# Args -- Approach
if args.approach == 'dsn':
    from approaches import dsn as approach


# Args -- Network
if args.experiment == 'cifar100':
    if args.approach == 'dsn':
        from networks import alexnet_rlrecover as network


def js_div(p_logits, q_logits, get_softmax=True):
    '''
    Function that measures JS divergence between target and output logits:
    '''
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = torch.nn.functional.softmax(p_logits.view(1, -1), dim=1)
        q_output = torch.nn.functional.softmax(q_logits.view(1, -1), dim=1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return 1 - (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def get_masks(net, t):
    old_mask = []
    for i in range(t + 1):
        masks = net.mask(i, s=args.betamax)
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


def calculate_complexity(masks, mask_pre):
    count = 0
    for m, mp in zip(masks, mask_pre):
        old_size = mp.shape
        new_size = m.shape
        try:
            count += list(new_size)[0] / list(old_size)[1]
        except IndexError:
            count += list(new_size)[0] / list(old_size)[0]

    return count / len(masks)


def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False



for i in range(1):
    # Load
    average1 = []
    firstAccuracy = []
    task_information = []
    bwt = []
    print('Load data...')
    data, taskcla, inputsize = dataloader.get(seed=args.seed)
    print('Input size =', inputsize, '\nTask info =', taskcla)
    if not os.path.exists(args.synth_img_save_dir):
        os.mkdir(args.synth_img_save_dir)

    print('Inits...')
    if args.approach == 'dsn' or args.approach == 'rcl':
        if args.experiment == 'amnist' or args.experiment == 'pmnist' or args.experiment == 'rmnist':
            action_num = 2
        else:
            action_num = 3
        net = network.Net(dict(), dict(), None, None, None, inputsize, taskcla,args).cuda()
        appr = approach.Appr(net, None, None, args.nepochs, lr=args.lr, args=args)

    else:
        print(args.approach)
        net = network.Net(inputsize, taskcla).cuda()
        appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args)
    utils.print_model_report(net)

    utils.print_optimizer_config(appr.optimizer)
    print('-' * 100)


    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)



    old_mask = []
    old_maskpre = []
    valid_acc = []
    neuronused = torch.zeros((9, 100))

    for t, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        task_pre = []
        if args.approach == 'dsn':

            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t


        else:
            # Get data
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t
        # Train
        if args.approach == 'dsn':
            if t > 0:
                if args.experiment == 'cifar100':
                    old_conv = dict()
                    old_fc = dict()
                    old_embedding = dict()
                    old_last = net_temp.last
                for n, p in net_temp.named_parameters():
                    if n.startswith('w'): 
                        old_W[n] = p
                    elif n.startswith('b'):  
                        old_b[n] = p
                    elif n.startswith('e'):
                        old_embedding[n] = p
                    elif n.startswith('c'): 
                        old_conv[n] = p
                    elif n.startswith('f'): 
                        old_fc[n] = p

                with torch.no_grad():
                    temp_pre = copy.deepcopy(old_maskpre)
                    temp_masks = copy.deepcopy(old_mask)


                actions=[0,0,0]
                for i, m in enumerate(temp_pre):
                    if i < action_num:
                        try:
                            temp_pre[i] = torch.cat((temp_pre[i], torch.zeros(1, actions[i]).cuda()), dim=1)
                        except IndexError:
                            temp_pre[i] = torch.cat(
                                    (torch.unsqueeze(temp_pre[i], 0), torch.zeros(1, actions[i]).cuda()), dim=1)
                for t_ in range(t):
                    task_pre.append(t_)
                    for i, m in enumerate(temp_pre):
                        if i < action_num:

                            try:
                                temp_masks[t_][i] = torch.cat(
                                        (temp_masks[t_][i], torch.zeros(1, actions[i]).cuda()), dim=1)
                            except IndexError:
                                temp_masks[t_][i] = torch.cat(
                                        (torch.unsqueeze(temp_masks[t_][i], 0), torch.zeros(1, actions[i]).cuda()),
                                        dim=1)

                if args.experiment == 'cifar100':
                    net = network.Net(old_conv, old_fc, old_last, old_embedding, actions, inputsize, taskcla,
                                        t=t, args = args).cuda()

                appr = approach.Appr(net, temp_pre, temp_masks, args.nepochs, lr=args.lr, args=args)
                appr.create_mask_back()
                _,perclass_information = appr.train(appr, t, xtrain, ytrain, xvalid, yvalid)
                task_information.append(perclass_information)
                similarity = [] 
                attention = []

                for i in range(t + 1):
                    mask = net.mask(i, s=args.betamax)
                    attention.append(mask)
                for m in attention:
                    cos = 0

                    for i in range(action_num):
                        cos += torch.cosine_similarity(m[i], attention[t][i], dim=0)

                    similarity.append(float(cos / action_num))


                complexity = calculate_complexity(mask, old_maskpre)
                _, ac = appr.eval_(t, xvalid, yvalid)



                print('transferring knowledge...')
                print('before:', valid_acc)
     
                old_mask, old_maskpre = get_masks(appr.model, t)
                mask_print = torch.zeros((9, 100))
                for i, m in enumerate(old_mask):
                    print('task', i, m[0].sum(), m[1].sum(),
                          m[2].sum())  
                appr.mask_pre = old_maskpre
                appr.old_mask = old_mask
                #task_pre = task_pre[:-1]
                #pdb.set_trace()
                taskWithSimilarity =dict(zip(task_pre, similarity[:-1])) 
                valid_acc = appr.generation_transfer(t, args.n, valid_acc, taskWithSimilarity,args,task_information)
                appr.best_model = utils.get_model(appr.model)

                net_temp = utils.set_model_(0, appr.best_model)

                old_mask, old_maskpre = get_masks(net_temp, t)
                for i, m in enumerate(old_mask):
                    print('task', i, m[0].sum(), m[1].sum())  # ,m[0], m[1])#,m[2].sum(),m[3].sum(),m[4].sum())

                v_acc, _ = appr.eval_(t, xvalid, yvalid)
                valid_acc.append(v_acc)

                a = np.array(mask_print.cpu())
   
                print('total', old_maskpre[0].sum(), old_maskpre[1].sum())  # ,old_maskpre[0], old_maskpre[1])


            else:
                _ = appr.train(appr, t, xtrain, ytrain, xvalid, yvalid)

                torch.save(appr.model, '.\parameter.pkl')
                net_temp = torch.load('.\parameter.pkl')
                old_maskpre = net_temp.mask(t, s=args.betamax)
                a = []
                for m in old_maskpre:
                    a.append(m.detach().clone())

                for i in range(len(a)):
                    a[i] = torch.round(a[i])
                old_maskpre = a
                old_mask.append(a)

                v_acc, _ = appr.eval_(t, xvalid, yvalid)
                valid_acc.append(v_acc)



        else:

            appr.train(task, xtrain, ytrain, xvalid, yvalid)
        print('-' * 100)

        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            if args.approach == 'dsn':
                test_loss, test_acc = appr.eval_(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                          100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss

   

    utils.print_model_report(net)

    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.2f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    if hasattr(appr, 'logs'):
        if appr.logs is not None:
            # save task names
            from copy import deepcopy

            appr.logs['task_name'] = {}
            appr.logs['test_acc'] = {}
            appr.logs['test_loss'] = {}
            for t, ncla in taskcla:
                appr.logs['task_name'][t] = deepcopy(data[t]['name'])
                appr.logs['test_acc'][t] = deepcopy(acc[t, :])
                appr.logs['test_loss'][t] = deepcopy(lss[t, :])
            # pickle
            import gzip
            import pickle

            with gzip.open(os.path.join(appr.logpath), 'wb') as output:
                pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)




########################################################################################################################
