import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def pair2var(train_pair, volatile=False):
    correct_variable = Variable(torch.FloatTensor(train_pair[0]), volatile=volatile)
    incorrect_variable = Variable(torch.FloatTensor(train_pair[1]), volatile=volatile)
    label_variable = Variable(torch.FloatTensor(train_pair[2]))
    if use_cuda:
        return (correct_variable.cuda(), incorrect_variable.cuda(), label_variable.cuda())
    else:
        return (correct_variable, incorrect_variable, label_variable)

def load_eval_data(eval_file, num_feats):
    instances = []
    loaded = 0
    with(open(eval_file)) as f:
        queries = []
        currId = "1"
        for line in f:
            loaded += 1
            parts = line.split()
            score = parts[0]
            queryId = parts[1].replace("qid:", "")
            if queryId == currId:
                queries.append((score, convert(parts[2:])))
            else:
                instances.append(queries.copy())
                currId = queryId
                queries.clear()
                queries.append((score, parts[2:], 1.0))
    return instances

def load_train_data(train_file, num_feats):
    instances = []
    loaded = 0
    with(open(train_file)) as f:
        queries = []
        currId = "1"
        for line in f:
            loaded += 1
            parts = line.split()
            score = parts[0]
            queryId = parts[1].replace("qid:", "")
            if queryId == currId:
                queries.append((score, parts[2:]))
            else:
                correct = convert(queries[0][1], num_feats)

                for query in queries[1:]:
                    incorrect = convert(query[1], num_feats)
                    instances.append((correct, incorrect, 1.0))
                currId = queryId
                queries.clear()
                queries.append((score, parts[2:], 1.0))
    return instances

def convert(query, num_feats):
    inst = [0.0 for i in range(num_feats)]
    for feat in query:
        featId, featVal = feat.split(":")
        inst[int(featId)] = float(featVal)
    return inst