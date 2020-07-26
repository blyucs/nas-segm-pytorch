import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import os
'''
pytorch实现focal loss的两种方式(现在讨论的是基于分割任务)
在计算损失函数的过程中考虑到类别不平衡的问题，假设加上背景类别共有6个类别
'''
def compute_class_weights(histogram):
    classWeights = np.ones(19, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(19):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights
def focal_loss_my(input,target):
    '''
    :param input: shape [batch_size,num_classes,H,W] 仅仅经过卷积操作后的输出，并没有经过任何激活函数的作用
    :param target: shape [batch_size,H,W]
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()
    number_4 = torch.sum(target == 4).item()
    number_5 = torch.sum(target == 5).item()
    number_6 = torch.sum(target == 6).item()
    number_7 = torch.sum(target == 7).item()
    number_8 = torch.sum(target == 8).item()
    number_9 = torch.sum(target == 9).item()
    number_10 = torch.sum(target == 10).item()
    number_11 = torch.sum(target == 11).item()
    number_12 = torch.sum(target == 12).item()
    number_13 = torch.sum(target == 13).item()
    number_14 = torch.sum(target == 14).item()
    number_15 = torch.sum(target == 15).item()
    number_16 = torch.sum(target == 16).item()
    number_17 = torch.sum(target == 17).item()
    number_18 = torch.sum(target == 18).item()
    #
    frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8, number_9,
                              number_10, number_11, number_12, number_13, number_14, number_15, number_16, number_17,number_18,), dtype=torch.float32)
    # number=[]
    # for i in range(20):
    #     number[i] = torch.sm(target == i).item()
    #     frequency = torch.cat(frequency,number[i])
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)
    '''
    根据当前给出的ground truth label计算出每个类别所占据的权重
    '''

    weights=torch.from_numpy(classWeights).float().cuda()
    # weights = torch.from_numpy(classWeights).float()
    focal_frequency = F.nll_loss(F.softmax(input, dim=1), target, reduction='none')
    '''
    上面一篇博文讲过
    F.nll_loss(torch.log(F.softmax(inputs, dim=1)，target)的函数功能与F.cross_entropy相同
    可见F.nll_loss中实现了对于target的one-hot encoding编码功能，将其编码成与input shape相同的tensor
    然后与前面那一项（即F.nll_loss输入的第一项）进行 element-wise production
    相当于取出了 log(p_gt)即当前样本点被分类为正确类别的概率
    现在去掉取log的操作，相当于  focal_frequency  shape  [num_samples]
    即取出ground truth类别的概率数值，并取了负号
    '''

    focal_frequency += 1.0#shape  [num_samples]  1-P（gt_classes）

    focal_frequency = torch.pow(focal_frequency, 2)  # torch.Size([75])
    focal_frequency = focal_frequency.repeat(c, 1)
    '''
    进行repeat操作后，focal_frequency shape [num_classes,num_samples]
    '''
    focal_frequency = focal_frequency.transpose(1, 0)
    loss = F.nll_loss(focal_frequency * (torch.log(F.softmax(input, dim=1))), target, weight=None,
                      reduction='elementwise_mean')
    return loss


def focal_loss_zhihu(input, target):
    '''
    :param input: 使用知乎上面大神给出的方案  https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()
    number_4 = torch.sum(target == 4).item()
    number_5 = torch.sum(target == 5).item()
    number_6 = torch.sum(target == 6).item()
    number_7 = torch.sum(target == 7).item()
    number_8 = torch.sum(target == 8).item()
    number_9 = torch.sum(target == 9).item()
    number_10 = torch.sum(target == 10).item()
    number_11 = torch.sum(target == 11).item()
    number_12 = torch.sum(target == 12).item()
    number_13 = torch.sum(target == 13).item()
    number_14 = torch.sum(target == 14).item()
    number_15 = torch.sum(target == 15).item()
    number_16 = torch.sum(target == 16).item()
    number_17 = torch.sum(target == 17).item()
    number_18 = torch.sum(target == 18).item()
    #
    frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8, number_9,
                              number_10, number_11, number_12, number_13, number_14, number_15, number_16, number_17,number_18,), dtype=torch.float32)

    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)

    weights = torch.from_numpy(classWeights).float().cuda()
    # weights = torch.from_numpy(classWeights).float()
    weights=weights[target.view(-1)]#这行代码非常重要   # after this , unable to get repr

    gamma = 2

    P = F.softmax(inputs, dim=1)  #shape [num_samples,num_classes]

    # torch.cuda.synchronize()

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding

    probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
    log_p = probs.log()

    # print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)
    batch_loss = torch.pow((1-probs),gamma)
    batch_loss = batch_loss * log_p
    weights = weights.view(-1,1)
    batch_loss = -weights * batch_loss
    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    # batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

    # print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
if __name__=='__main__':
    pred=torch.rand((2,19,512,512)).cuda()
    y=torch.from_numpy(np.random.randint(0,19,(2,512,512))).cuda()
    loss1=focal_loss_my(pred,y)
    loss2=focal_loss_zhihu(pred,y)

    print('loss1',loss1)
    print('loss2', loss2)
'''
in calculating batch_loss torch.Size([50]) torch.Size([50, 1]) torch.Size([50, 1])
torch.Size([50, 1])
loss1 tensor(1.3166)
loss2 tensor(1.3166)
'''