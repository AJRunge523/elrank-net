import sys
import torch
from data_loading import load_eval_data
import argparse
import logging
from utils import toVar
import logging.config

logger = logging.getLogger(__name__)

def main(args):
    feats_path = args.feats
    num_feats = len([line for line in open(feats_path)])
    dev_data_path = "data/dev.dat"
    eval_data_path = "data/eval.dat"
    model = torch.load(args.model)
    dev_eval_instances = load_eval_data(dev_data_path, num_feats)
    eval_instances = load_eval_data(eval_data_path, num_feats)
    dev_accuracy = evaluate(dev_eval_instances, model)
    eval_accuracy = evaluate(eval_instances, model)
    logger.info("Dev accuracy: {}, Eval accuracy: {}".format(dev_accuracy, eval_accuracy))


def evaluate(instances, ranker, score_file):
    with open(score_file, 'w+') as f:
        num_correct = 0
        num_nil_correct = 0
        num_kb_correct = 0
        num_kb = 0
        num_nil = 0
        total = 0
        for i in instances:
            best_score = -100000000
            best_rank = 0
            bestIsNil = False
            f.write("##Query ID {}##\n".format(i[0][0]))
            for q in i:
                rank = q[1]
                vec = q[2]
                isNil = sum(vec) == 1
                if isNil and rank == '2':
                    num_nil += 1
                elif rank == '2':
                    num_kb += 1
                score = ranker(toVar(vec)).data[0]
                f.write(str(score) + '\n')
                if score > best_score:
                    bestIsNil = isNil
                    best_score = score
                    best_rank = rank
            if best_rank == '2':
                if bestIsNil:
                    num_nil_correct += 1
                else:
                    num_kb_correct += 1
                num_correct += 1
            total += 1
    if num_kb < 100:
        num_kb = 90
    else:
        num_kb = 1020
    print("KB Correct: {}, NIL Correct: {}".format(num_kb_correct / num_kb, num_nil_correct / num_nil))
    print("Number KB: {}, Number NIL: {}".format(num_kb, num_nil))
    print("Number correct: {}, Total Queries: {}".format(num_correct, total))
    return num_correct / total



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--feats", action='store_true')
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'training.log'})
    main(args)