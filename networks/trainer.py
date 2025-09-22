import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel, init_weights
import sys
from loss import BCEFocalLoss
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt.arch, opt)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.fc.named_parameters():
                params.append(p) 
            self.model.freeze_backbone()
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()


        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")
        if opt.focalloss:
            print("Use FocalLoss!")
            self.loss_fn = BCEFocalLoss()
        else:
            print("Use BCELoss!")
            self.loss_fn = nn.BCEWithLogitsLoss() 

        self.model.cuda()
    def load_ckpt(self, path):
        print("Load ckpt from:", path)
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model']
        self.model.fc.load_state_dict(state_dict)
        self.model.freeze_backbone()

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].cuda()
        self.label = input[1].cuda().float()

    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        # self.label = self.label * (1 - 0.05) + 0.05 * (1 - self.label)
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



