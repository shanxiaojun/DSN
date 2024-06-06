import sys
import torch
import numpy as np
from src import utils
import copy
import math
from torch.nn import functional as F
from utilsv.buffer import Buffer
from cf_il.recover_memory import recovery_memory as rec_mem
class Net(torch.nn.Module):

    def __init__(self,conv, fc, last, embedding, scale, inputsize, taskcla, args, nlayers=2, nhid=2048,
                 pdrop1=0.2,
                 pdrop2=0.5, t=0):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.size = size

        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)


        if t==0:
            self.c1w=torch.empty((64,ncha,size//8,size//8))
            self.c1b=torch.empty((64,))
            self.c1w,self.c1b=self.initialize(self.c1w,self.c1b)
            self.c1w=torch.nn.Parameter(self.c1w)
            self.c1b = torch.nn.Parameter(self.c1b)
            
            self.c2w =torch.empty((128,64,size//10,size//10))
            self.c2b =torch.empty((128,))
            self.c2w, self.c2b = self.initialize(self.c2w, self.c2b)
            self.c2w = torch.nn.Parameter(self.c2w)
            self.c2b = torch.nn.Parameter(self.c2b)
            
            self.c3w=torch.empty((256,128,2,2))
            self.c3b=torch.empty((256,))
            self.c3w, self.c3b = self.initialize(self.c3w, self.c3b)
            self.c3w = torch.nn.Parameter(self.c3w)
            self.c3b = torch.nn.Parameter(self.c3b)
            
            self.fc1=torch.nn.ParameterList()
            w_f= torch.empty((nhid,256*self.smid*self.smid))
            b_f = torch.empty((nhid,))
            w_f,b_f=self.initialize(w_f,b_f)
            self.fc1.append(torch.nn.Parameter(w_f))
            self.fc1.append(torch.nn.Parameter(b_f))
            
            #self.fc2 = torch.nn.Linear(nhid, nhid)
            self.fc2 = torch.nn.ParameterList()
            w_f = torch.empty((nhid, nhid))
            b_f = torch.empty((nhid,))
            w_f, b_f = self.initialize(w_f, b_f)
            self.fc2.append(torch.nn.Parameter(w_f))
            self.fc2.append(torch.nn.Parameter(b_f))

            self.last=torch.nn.ModuleList()

            self.last = torch.nn.ParameterList()
            for t_, n in self.taskcla:
                w_l = torch.empty((n, nhid))
                b_l = torch.empty((n,))
                w_l, b_l = self.initialize(w_l, b_l)
                self.last.append(torch.nn.Parameter(w_l))
                self.last.append(torch.nn.Parameter(b_l))

            self.ec1 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 64), mean=0, std=1))
            self.ec2 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 128), mean=0, std=1))
            self.ec3 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), 256), mean=0, std=1))
            self.efc1 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), nhid), mean=0, std=1))
            self.efc2 = torch.nn.Parameter(torch.nn.init.normal_(torch.FloatTensor(len(self.taskcla), nhid), mean=0, std=1))

        else:
            c1w_exp = torch.empty(scale[0],ncha,size//8,size//8)
            c1b_exp = torch.empty((scale[0],))
            c1w_exp, c1b1_exp = self.initialize(c1w_exp, c1b_exp)
            self.c1w =torch.cat((conv['c1w'],c1w_exp.cuda()),dim=0)
            self.c1b =torch.cat((conv['c1b'],c1b_exp.cuda()),dim=0)
            self.c1w = torch.nn.Parameter(self.c1w)
            self.c1b = torch.nn.Parameter(self.c1b)

            n2_exp=torch.empty(conv['c2w'].shape[0], scale[0], size//10, size//10)
            try:
                torch.nn.init.kaiming_uniform_(n2_exp, a=math.sqrt(5))
            except Exception:
                pass
            n2_exp = torch.cat((conv['c2w'], n2_exp.cuda()), dim=1)
            c2w_exp = torch.empty(scale[1], conv['c1w'].shape[0]+scale[0], size//10, size//10)
            c2b_exp = torch.empty((scale[1],))
            c2w_exp, c2b2_exp = self.initialize(c2w_exp, c2b_exp)
            self.c2w = torch.cat((n2_exp, c2w_exp.cuda()), dim=0)
            self.c2b = torch.cat((conv['c2b'], c2b_exp.cuda()), dim=0)
            self.c2w = torch.nn.Parameter(self.c2w)
            self.c2b = torch.nn.Parameter(self.c2b)

            n3_exp = torch.empty(conv['c3w'].shape[0], scale[1], 2,2)
            try:
                torch.nn.init.kaiming_uniform_(n3_exp, a=math.sqrt(5))
            except Exception:
                pass
            n3_exp = torch.cat((conv['c3w'], n3_exp.cuda()), dim=1)
            c3w_exp = torch.empty(scale[2],conv['c2w'].shape[0]+scale[1],2,2)
            c3b_exp = torch.empty((scale[2],))
            c3w_exp, c3b_exp = self.initialize(c3w_exp, c3b_exp)
            self.c3w =torch.cat((n3_exp,c3w_exp.cuda()),dim=0)
            self.c3b =torch.cat((conv['c3b'],c3b_exp.cuda()),dim=0)
            self.c3w = torch.nn.Parameter(self.c3w)
            self.c3b = torch.nn.Parameter(self.c3b)

            self.fc1 = torch.nn.ParameterList()
            fcs= copy.deepcopy(fc)
            fc1_exp = torch.empty((fcs['fc1.0'].shape[0], 2*2*(scale[2])))
            try:
                torch.nn.init.kaiming_uniform_(fc1_exp, a=math.sqrt(5))
            except Exception:
                pass
            fc1_b=fcs['fc1.1']
            self.fc1.append(torch.nn.Parameter(torch.cat((fcs['fc1.0'], fc1_exp.cuda()), 1)))
            self.fc1.append(torch.nn.Parameter(fc1_b))

            self.fc2 = torch.nn.ParameterList()
            fc2_w = fcs['fc2.0']
            fc2_b = fcs['fc2.1']
            self.fc2.append(torch.nn.Parameter(fc2_w))
            self.fc2.append(torch.nn.Parameter(fc2_b))

            self.last = copy.deepcopy(last)

            expand1 = torch.full((len(self.taskcla), scale[0]), -6)
            self.ec1 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['ec1']), expand1.cuda()), dim=1))

            expand2 = torch.full((len(self.taskcla), scale[1]), -6)
            self.ec2 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['ec2']), expand2.cuda()), dim=1))
            expand3= torch.full((len(self.taskcla), scale[2]), -6)
            self.ec3 = torch.nn.Parameter(torch.cat((copy.deepcopy(embedding['ec3']), expand3.cuda()), dim=1))
            self.efc1 = torch.nn.Parameter(copy.deepcopy(embedding['efc1']))
            self.efc2 = torch.nn.Parameter(copy.deepcopy(embedding['efc2']))

        self.gate=torch.nn.Sigmoid()
        self.buffer = Buffer(
                buffer_size=args.buffer_size,
                device=torch.device('cuda:0'),
                mode='reservoir',
        )
        self.psi = args.psi
        self.tau = args.tau
        self.scaleD = args.scale
        self.optim_steps = args.optim_steps
        self.optim_lr = args.optim_lr
        self.synth_img_save_dir = args.synth_img_save_dir
        self.synth_img_save_num = args.synth_img_save_num
        self.dirichlet_max_iter = args.dirichlet_max_iter

        for n, p in self.named_parameters():
            p.requires_grad = True



        return

    def forward(self,t,x,s=1,draw=False):
        # Gates
        masks=self.mask(t,s=s)
        '''if t == 2:
            print(t)'''
        gc1,gc2,gc3,gfc1,gfc2=masks
        # Gated
        h=F.conv2d(x, self.c1w, self.c1b, stride=1)
        h=self.maxpool(self.drop1(self.relu(h)))
        h=h*gc1.view(1,-1,1,1).expand_as(h)

        h=F.conv2d(h, self.c2w, self.c2b, stride=1)
        h=self.maxpool(self.drop1(self.relu(h)))
        h=h*gc2.view(1,-1,1,1).expand_as(h)

        h=F.conv2d(h, self.c3w, self.c3b, stride=1)
        h=self.maxpool(self.drop2(self.relu(h)))
        h=h*gc3.view(1,-1,1,1).expand_as(h)

        h=h.view(x.size(0),-1)
        a = self.fc1[0]
        b = self.fc1[1]
        h = self.drop2(self.relu(torch.nn.functional.linear(h, self.fc1[0], self.fc1[1])))
        h=h*gfc1.expand_as(h)

        #h=self.drop2(self.relu(self.fc2(h)))
        h = self.drop2(self.relu(torch.nn.functional.linear(h, self.fc2[0], self.fc2[1])))
        h=h*gfc2.expand_as(h)

        y=[]
        for t_,_ in self.taskcla:
            y.append(torch.nn.functional.linear(h, self.last[t_ * 2], self.last[t_ * 2 + 1]))
            #y.append(self.last[i](h))
        return y,masks

    def mask(self,t,s=1):

        gc1=self.gate(s*self.ec1[t])
        gc2=self.gate(s*self.ec2[t])
        gc3=self.gate(s*self.ec3[t])
        gfc1=self.gate(s*self.efc1[t])
        gfc2=self.gate(s*self.efc2[t])
        return [gc1,gc2,gc3,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='fc1.0':
            post=gfc1.data.view(-1,1).expand_as(self.fc1[0]) #post nhid,6456 gfc1 nhid

            pre=gc3.data.view(-1,1,1).expand((self.ec3.shape[1],self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1[0])

            return torch.min(post,pre)
        elif n=='fc1.1':
            return gfc1.data.view(-1)
        elif n=='fc2.0':
            post=gfc2.data.view(-1,1).expand_as(self.fc2[0])
            pre=gfc1.data.view(1,-1).expand_as(self.fc2[0])
            return torch.min(post,pre)
        elif n=='fc2.1':
            return gfc2.data.view(-1)
        elif n=='c1w':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1w)
        elif n=='c1b':
            return gc1.data.view(-1)
        elif n=='c2w':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2w)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2w)
            return torch.min(post,pre)
        elif n=='c2b':
            return gc2.data.view(-1)
        elif n=='c3w':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3w)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3w)
            return torch.min(post,pre)
        elif n=='c3b':
            return gc3.data.view(-1)
        return None

    def initialize(self,w,b):
        try:
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(b, -bound, bound)
        except Exception:pass
        return w,b

    def recover_memory(
        self,
        num_classes: int,
        modelmain,
        current_task,
        task_information
    ) -> None:

        net_training_status = modelmain.training #True
        modelmain.eval()

        self.buffer.empty()

        synth_images, synth_logits = rec_mem(
            current_task = current_task,
            model=modelmain,
            num_classes=num_classes,
            buffer_size=self.buffer.buffer_size,
            image_shape=[32,32,3],
            device=torch.device('cuda:0'),
            scale=self.scaleD,
            psi=self.psi,
            tau=self.tau,
            optim_steps=self.optim_steps,
            optim_lr=self.optim_lr,
            synth_img_save_dir=self.synth_img_save_dir,
            synth_img_save_num=self.synth_img_save_num,
            dirichlet_max_iter=self.dirichlet_max_iter,
            task_information = task_information
        )

        for img, logits in zip(synth_images, synth_logits):
            self.buffer.add_data(examples=img, logits=logits)

        modelmain.train(net_training_status)
        return self.buffer
