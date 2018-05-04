from utils import use_cuda
from data_loading import load_data, make_batches, load_eval_data
from ranker import Ranker
import torch
import random
import logging.config
from train_ranker import RankerTrainer
from os import makedirs

from experiments import addition_experiments, deletion_experiments

logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'del-experiments.log'})
logger = logging.getLogger(__name__)

def main():
    experiment_set = deletion_experiments
    train_data_path = "data/training.dat"
    dev_data_path = "data/full/dev.dat"
    tst_data_path = "data/full/evaluation.dat"
    feats_path = "data/model.features"
    num_feats = len([line for line in open(feats_path)])
    batch_size = 80

    for experiment_name in experiment_set.keys():
        logger.info("Running experiment {}".format(experiment_name))
        exp_features = experiment_set[experiment_name]
        out_path = 'output/experiments/{}'.format(experiment_name)
        makedirs(out_path, exist_ok=True)
        train_instances = load_data(train_data_path, num_feats, exp_features)
        dev_instances = load_data(dev_data_path, num_feats, exp_features)
        dev_eval_instances = load_eval_data(dev_data_path, num_feats, exp_features)
        tst_instances = load_eval_data(tst_data_path, num_feats, exp_features)
        logger.info("Loaded {} training instances with {} features".format(len(train_instances), num_feats))
        for i in range(3):
            iter_path = out_path + '/v{}'.format(i)
            makedirs(iter_path, exist_ok=True)
            ranker = Ranker(num_feats, 256)
            trainer = RankerTrainer(ranker, batch_size, iter_path)
            trainer.train(train_instances, dev_instances, None, dev_eval_instances, tst_instances)

if __name__ == '__main__':
    main()