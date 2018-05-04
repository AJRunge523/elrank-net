from utils import pair2var, use_cuda
from data_loading import load_data, make_batches, load_eval_data
from ranker import Ranker
import torch.nn as nn
import torch
import random
from torch import optim
import argparse
import logging
import logging.config

from eval import evaluate

logger = logging.getLogger(__name__)



def main(args):
    torch.manual_seed(333)
    if use_cuda:
        torch.cuda.manual_seed(333)
    random.seed(333)
    train_data_path = "data/training.dat"
    train_eval_data_path = "data/train-eval.dat"
    dev_data_path = "data/full/dev.dat"
    eval_data_path = "data/full/evaluation.dat"
    feats_path = "data/model.features"
    num_feats = len([line for line in open(feats_path)])
    batch_size = 80
    ranker = Ranker(num_feats, 256)
    ## Instances for training - loaded as pairs
    feat_indices = set([i for i in range(num_feats)])
    train_instances = load_data(train_data_path, num_feats, feat_indices)
    train_eval_instances = load_eval_data(train_data_path, num_feats, feat_indices)
    dev_instances = load_data(dev_data_path, num_feats, feat_indices)
    dev_eval_instances = load_eval_data(dev_data_path, num_feats, feat_indices)
    tst_instances = load_eval_data(eval_data_path, num_feats, feat_indices)
    logger.info("Loaded {} training instances with {} features".format(len(train_instances), num_feats))
    trainer = RankerTrainer(ranker, batch_size, 'output/')
    trainer.train(train_instances, dev_instances, train_eval_instances, dev_eval_instances, tst_instances)
    ranker.save('output/ranker.model')

class RankerTrainer:

    def __init__(self, model, batch_size, out_path):
        self.model = model
        self.batch_size = batch_size
        self.out_path = out_path
        pass

    def train(self, train_instances, dev_instances, train_eval_instances, dev_eval_instances, tst_instances):

        train_batches = make_batches(train_instances, self.batch_size)
        dev_batches = make_batches(dev_instances, self.batch_size)
        loss_fn = nn.MarginRankingLoss(margin=1)
        optimizer = optimizer_factory('Adam', self.model, lr=0.0001)
        num_batches = len(train_batches)
        best_dev_loss = 10000000
        tolerance = 3
        epoch = 1
        tolerance_rem = tolerance
        best_dev_kb = 0
        best_dev_nil = 0
        while tolerance_rem > 0:
            train_loss = 0
            for i in range(num_batches):
                batch = train_batches[i]
                if i % 50 == 0:
                    logger.info("Processed {} batches of {}".format(i, num_batches))
                optimizer.zero_grad()
                # print(correct.shape, incorrect.shape, labels.shape)
                loss = compute_loss(self.model, batch, loss_fn, volatile=False)
                train_loss += loss
                loss.backward()
                optimizer.step()

            logger.info("Train loss at epoch {}: {}".format(epoch, train_loss.data[0]))

            dev_loss = 0
            for i in range(len(dev_batches)):
                batch = dev_batches[i]
                loss = compute_loss(self.model, batch, loss_fn, volatile=True)
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
            self.model.eval()
            dev_acc, dev_kb_acc, dev_nil_acc  = evaluate(dev_eval_instances, self.model, self.out_path + '/dev_eval.txt',
                                                     'data/full/dev-query-ids.txt', self.out_path + '/dev_results.txt',
                                                         best_dev_kb, best_dev_nil)
            logger.info(
                "Dev accuracy: {}, Dev KB accuracy: {}, Dev NIL Accuracy: {}".format(dev_acc, dev_kb_acc, dev_nil_acc))
            if dev_kb_acc > best_dev_kb or (dev_kb_acc == best_dev_kb and dev_nil_acc > best_dev_nil):
                logger.info("Evaluating test data")
                best_dev_kb = dev_kb_acc
                if dev_nil_acc > best_dev_nil:
                    best_dev_nil = dev_nil_acc
                eval_acc, eval_kb_acc, eval_nil_acc = evaluate(tst_instances, self.model, self.out_path + '/tst_eval.txt',
                                         'data/full/tst-query-ids.txt', self.out_path + '/tst_results.txt')
                logger.info("Tst accuracy: {}, Tst KB accuracy: {}, Tst NIL accuracy: {}".format(eval_acc, eval_kb_acc, eval_nil_acc))

            self.model.train()


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