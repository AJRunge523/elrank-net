import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def pair2var(train_pair, volatile=False):
    correct_variable = toVar(train_pair[0], volatile=volatile)
    incorrect_variable = toVar(train_pair[1], volatile=volatile)
    label_variable = toVar(train_pair[2], volatile=volatile)
    if use_cuda:
        return (correct_variable.cuda(), incorrect_variable.cuda(), label_variable.cuda())
    else:
        return (correct_variable, incorrect_variable, label_variable)

def toVar(vec, volatile=False):
    return Variable(torch.FloatTensor(vec), volatile=volatile)