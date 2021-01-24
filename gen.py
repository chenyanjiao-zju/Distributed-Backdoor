import torch
import copy
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import numpy as np
import image_helper
import random
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import utils as vutils
from models.MnistNet import MnistNet
from models.resnet_cifar import ResNet18

# key_to_maximize = 0

fmap_block = dict()

def forward_hook(module, inp, outp):
    fmap_block['in'] = inp
    fmap_block['out'] = outp

def backward_hook(module, gi, go):
    targeted_grad = torch.zeros_like(gi[1])
    targeted_grad[0][26] = -1
    return gi[0], targeted_grad, gi[2]

def preprocess(ori):
    ori = torch.from_numpy(ori)
    ori = ori.float().div(255)
    return ori

def deprocess(img):
    img = img * 255.0
    img = torch.clamp(img, 0, 255)
    return img

def filter_part(helper, h, w, poison_patterns):
    mask = np.zeros((h, w))
    for pos in poison_patterns:
        mask[pos[0]][pos[1]] = 1
    return mask

def select_neuron(helper):
    model = copy.deepcopy(helper.target_model)
    parm = {}
    for name, parameters in model.named_parameters():
        parm[name] = parameters.cpu().detach().numpy()
    # select layer containing neurons; mnist-fc1 cifar10-i=linear 
    # max_key = np.argsort(-parm['fc1.weight'].sum(axis=1))[0]  
    # print("top1 weight sum of fc1:", max_key)   
    max_key = np.argsort(-parm['linear.weight'].sum(axis=0))[0]
    print("top1 weight sum of fc1:", max_key)
    return max_key

def generate_trigger(helper, adversarial_index):
    w = h = 32 # data size
    poison_patterns = helper.params[str(adversarial_index) + '_poison_pattern']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key_to_maximize = 26 # select_neuron(helper) 
    # for mnist 
    # mnist_model = copy.deepcopy(helper.target_model)
    # mnist_model.eval()
    # mnist_model.fc1.register_forward_hook(forward_hook)
    # mnist_model.fc2.register_backward_hook(backward_hook)
    cifar10_model = copy.deepcopy(helper.target_model)
    cifar10_model.eval()
    cifar10_model.linear.register_forward_hook(forward_hook)
    cifar10_model.linear.register_backward_hook(backward_hook)
    mask_logo = filter_part(helper, h, w, poison_patterns)
    mask = np.float32(mask_logo > 0)

   
    #x = np.random.rand(28, 28)  # 28 * 28 for mnist
    x = np.random.rand(3,32,32)
    #x = x[np.newaxis, np.newaxis, :]
    x = x[np.newaxis, :]
    x = preprocess(x)
    x = x.to(device)

    optimizer = optim.SGD([x], lr=0)
    stepping = [(100 10), (1500, 5), (1000, 1)]
    i = 0
    obj_act = 0
    opt_act = 0
    opt_x = x.clone().detach()
    for rounds, lr in stepping:
        for r in range(rounds):
            #print("@@@@@@Round", i)
            optimizer.zero_grad()
            x = Variable(x, requires_grad=True)
            #output = mnist_model(x)        # for mnist
            output = cifar10_model(x)
            #output_fc1 = fmap_block['out'] # for mnist
            output_fc1 = fmap_block['in'][0]
            obj_act = output_fc1[0][key_to_maximize]
            print("selected neuron", key_to_maximize, ":", obj_act)
            if obj_act > opt_act:
                opt_act = obj_act
                opt_x = x.clone().detach()
            rank = np.argsort(-output_fc1.detach().cpu().numpy())
            print("max", rank[0][0], ":", output_fc1[0][rank[0][0]])
            if i == 0:
                before_act = output_fc1[0][key_to_maximize]

            obj = torch.tensor(2).reshape(-1)
            obj = obj.to(device)
            loss = nn.L1Loss()(output, obj.float())
            loss.backward()
            grad_mean = np.abs(x.grad.data.cpu()).mean()
            x.grad.data.mul_(torch.from_numpy(mask).to(device) * lr / (100 * grad_mean))
            # optimizer.step()
            new_x = x - x.grad.data
            new_x = torch.clamp(new_x, 0, 1.)
            x = new_x
            i += 1

    print("before_act: ", before_act)
    print("after_act: ", opt_act)
    x = deprocess(opt_x)
    trigger = x.to(torch.device('cpu'))
    vutils.save_image(trigger, str(adversarial_index)+'.jpg')
    print(trigger.shape, helper.global_trigger.shape)
    for pos in poison_patterns:
        #helper.global_trigger[0][pos[0]][pos[1]] = trigger[0][0][pos[0]][pos[1]] # for mnist
        helper.global_trigger[0][pos[0]][pos[1]] = trigger[0][0][pos[0]][pos[1]]
        helper.global_trigger[1][pos[0]][pos[1]] = trigger[0][1][pos[0]][pos[1]]
        helper.global_trigger[2][pos[0]][pos[1]] = trigger[0][2][pos[0]][pos[1]]
        
        
        print(pos, trigger[0][0][pos[0]][pos[1]])
