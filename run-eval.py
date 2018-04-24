import sys
import torch
from utils import load_eval_data
import argparse
import logging

def main(args):
    feats_path = args.feats
    num_feats = len([line for line in open(feats_path)])
    eval_data = load_eval_data(num_feats)
    model = torch.load(args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--feats", action='store_true')
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'training.log'})
    main(args)