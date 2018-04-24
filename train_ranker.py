from utils import pair2var, load_train_data, load_instances
from Ranker import Ranker
import numpy as np
import torch
import torch.nn as nn
from torch import optim

def main():
    train_data_path = "C:/Users/Andrew/Desktop/train.dat"
    feats_path = "C://Users/Andrew/Desktop/model.features"
    num_feats = len([line for line in open(feats_path)])
    batch_size = 64
    ranker = Ranker(num_feats, 256)
    train_instances = load_train_data(train_data_path, num_feats)
    print("Loaded {} training instance".format(len(train_instances)))
    batches = make_batches(train_instances, batch_size)
    loss_fn = nn.MarginRankingLoss(margin=1)
    optimizer = optimizer_factory('Adam', ranker, lr=0.0001)
    num_batches = len(batches)
    for epoch in range(5):
        total_loss = 0
        for i in range(num_batches):
            batch = batches[i]
            if i % 50 == 0:
                print("Processed {} batches of {}".format(i, num_batches))
            optimizer.zero_grad()
            correct, incorrect, labels = pair2var(batch)
            # print(correct.shape, incorrect.shape, labels.shape)
            correct_scores = ranker.forward(correct)
            incorrect_scores = ranker.forward(incorrect)
            # print(correct_scores.shape, incorrect_scores.shape)
            loss = loss_fn(correct_scores, incorrect_scores, labels)
            total_loss += (loss.sum() / batch_size)
            loss.backward()
            optimizer.step()
        print("Total loss at epoch {}: {}".format(total_loss.data(), epoch))
    ranker.save("models/ranker.model")

def optimizer_factory(optim_type, model, **kwargs):
    assert optim_type in ['SGD', 'Adam'], 'Optimizer type not one of currently supported options'
    return getattr(optim, optim_type)(model.parameters(), **kwargs)

def make_batches(train_instances, batch_size=64):
    batches = []
    curr_batch = []
    for i in range(len(train_instances)):
        curr_batch.append(train_instances[i])
        if len(curr_batch) == batch_size:
            correct_batch = np.array([b[0] for b in curr_batch])
            incorrect_batch = np.array([b[1] for b in curr_batch])
            label_batch = np.array([b[2] for b in curr_batch])
            batches.append((correct_batch, incorrect_batch, label_batch))
            curr_batch.clear()
    return batches


if __name__ == '__main__':
    main()