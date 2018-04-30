from utils import pair2var
from data_loading import load_data, make_batches, load_eval_data
from ranker import Ranker
import torch.nn as nn
from torch import optim
import argparse
import logging
import logging.config

from eval import evaluate

logger = logging.getLogger(__name__)

def main(args):
    train_data_path = "data/train.dat"
    dev_data_path = "data/dev.dat"
    eval_data_path = "data/eval.dat"
    feats_path = "data/model.features"
    num_feats = len([line for line in open(feats_path)])
    batch_size = 64
    ranker = Ranker(num_feats, 256)
    ## Instances for training - loaded as pairs
    train_instances = load_data(train_data_path, num_feats)
    dev_instances = load_data(dev_data_path, num_feats)
    dev_eval_instances = load_eval_data(dev_data_path, num_feats)
    eval_instances = load_eval_data(eval_data_path, num_feats)
    logger.info("Loaded {} training instances with {} features".format(len(train_instances), num_feats))
    train_batches = make_batches(train_instances, batch_size)
    print(len(train_batches))
    dev_batches = make_batches(dev_instances, batch_size)
    loss_fn = nn.MarginRankingLoss(margin=1)
    optimizer = optimizer_factory('Adam', ranker, lr=0.0001)
    num_batches = len(train_batches)
    best_dev_loss = 10000000
    tolerance = 3
    epoch = 1
    tolerance_rem = tolerance
    while tolerance_rem > 0:
        train_loss = 0
        for i in range(num_batches):
            batch = train_batches[i]
            if i % 50 == 0:
                logger.info("Processed {} batches of {}".format(i, num_batches))
            optimizer.zero_grad()
            # print(correct.shape, incorrect.shape, labels.shape)
            loss = compute_loss(ranker, batch, loss_fn, volatile=False)
            train_loss += loss
            loss.backward()
            optimizer.step()

        logger.info("Train loss at epoch {}: {}".format(epoch, train_loss.data[0]))

        dev_loss = 0
        for i in range(len(dev_batches)):
            batch = dev_batches[i]
            loss = compute_loss(ranker, batch, loss_fn, volatile=True)
            dev_loss += loss
        dev_loss_val = dev_loss.data[0]
        if dev_loss_val < best_dev_loss:
            best_dev_loss = dev_loss_val
            tolerance_rem = tolerance
        else:
            tolerance_rem -= 1
            logger.info("Tolerance at {}.".format(tolerance_rem))
        logger.info("Dev loss at epoch {}: {}".format(epoch, dev_loss_val))
        epoch += 1
        ranker.eval()
        dev_accuracy = evaluate(dev_eval_instances, ranker, 'output/dev_scores.txt')
        eval_accuracy = evaluate(eval_instances, ranker, 'output/eval_scores.txt')
        logger.info("Dev accuracy: {}, Eval accuracy: {}".format(dev_accuracy, eval_accuracy))
        ranker.train()
    ranker.save("models/ranker.model")




def compute_loss(ranker, batch, loss_fn, volatile=False):
    correct, incorrect, labels = pair2var(batch, volatile=volatile)
    batch_size = correct.shape[0]
    correct_scores = ranker(correct)
    incorrect_scores = ranker(incorrect)
    # print(correct_scores.shape, incorrect_scores.shape)
    loss = loss_fn(correct_scores, incorrect_scores, labels)
    return loss.sum() #/ batch_size

def optimizer_factory(optim_type, model, **kwargs):
    assert optim_type in ['SGD', 'Adam'], 'Optimizer type not one of currently supported options'
    return getattr(optim, optim_type)(model.parameters(), **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--feats", action='store_true')
    parser.add_argument
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'training.log'})
    main(args)