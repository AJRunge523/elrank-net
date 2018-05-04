import sys
import torch
from data_loading import load_eval_data
import argparse
import logging
from utils import toVar
import logging.config
from functools import reduce

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


def evaluate(instances, ranker, entity_eval_file, id_file, results_file, best_kb_accuracy=0, best_nil_accuracy=0):
    ids = [line.strip() for line in open(id_file)]
    entity_type_macro = {}
    entity_macro = {}
    query_idx = 0
    num_correct = 0
    num_nil_correct = 0
    num_kb_correct = 0
    num_kb = 0
    num_nil = 0
    total = 0
    eval_entries = []
    for i in instances:
        best_score = -100000000
        best_rank = 0
        bestIsNil = False
        best_id = ''
        query_type = i[0][5]
        if query_type not in entity_type_macro:
            entity_type_macro[query_type] = [0, 0]
        gold_id = i[0][4]
        if gold_id != 'NIL':
            if gold_id not in entity_macro:
                entity_macro[gold_id] = [0, 0]
            entity_macro[gold_id][1] += 1
        best_found = False
        for q in i:
            rank = q[1]
            vec = q[2]
            id = q[3]
            isNil = sum(vec) == 1
            if isNil and rank == '2':
                num_nil += 1
                best_found = True
            elif rank == '2':
                num_kb += 1
                best_found = True
            score = ranker(toVar(vec)).data[0]
            # f.write(str(score) + '\n')
            if score > best_score:
                bestIsNil = isNil
                best_score = score
                best_rank = rank
                best_id = id
        if not best_found:
            num_kb += 1
        # f.write('{}\t{}\t{}'.format(ids[query_idx], best_id, gold_id))
        if best_rank == '2':
            entity_type_macro[query_type][0] += 1
            if bestIsNil:
                num_nil_correct += 1
            else:
                entity_macro[gold_id][0] += 1
                num_kb_correct += 1
            num_correct += 1
            eval_entries.append((ids[query_idx], best_id, gold_id, ''))
        else:
            eval_entries.append((ids[query_idx], best_id, gold_id, '*'))
        entity_type_macro[query_type][1] += 1

        total += 1
        query_idx += 1
    kb_accuracy = num_kb_correct / num_kb
    nil_accuracy = num_nil_correct / num_nil
    write = False
    # None passed in, assume we should always write
    if best_kb_accuracy == best_nil_accuracy == 0:
        write = True
    # If this is the best kb accuracy, write to a different file
    elif kb_accuracy > best_kb_accuracy or (kb_accuracy == best_kb_accuracy and nil_accuracy > best_nil_accuracy):
        write = True
    if write:
        logger.info("Writing results")
        with open(entity_eval_file, 'w+') as f:
            for entry in eval_entries:
                f.write('{}\t{}\t{}\t{}\n'.format(entry[0], entry[1], entry[2], entry[3]))

        with open(results_file, 'w+') as results:
            results.write("Micro-Averages:\n")
            results.write("{} queries: {}\n".format(total, (num_correct / total)))
            results.write("{} KB: {}\n".format(num_kb, num_kb_correct / num_kb))
            results.write("{} NIL: {}\n".format(num_nil, num_nil_correct / num_nil))
            results.write("Entity Type Macro Averages:\n")
            results.write("{} PER queries: {}\n".format(entity_type_macro['PER'][1],
                                                        entity_type_macro['PER'][0] / entity_type_macro['PER'][1]))
            results.write("{} GPE queries: {}\n".format(entity_type_macro['GPE'][1],
                                                        entity_type_macro['GPE'][0] / entity_type_macro['GPE'][1]))
            results.write("{} ORG queries: {}\n".format(entity_type_macro['ORG'][1],
                                                        entity_type_macro['ORG'][0] / entity_type_macro['ORG'][1]))

            macro_accuracy = reduce(lambda a, b: a + b, [entity_macro[key][0] / entity_macro[key][1] for key in entity_macro.keys()]) / len(entity_macro.keys())

            results.write("Entity Macro Averages:\n")
            results.write("{} gold entities: {}\n".format(len(entity_macro.keys()), macro_accuracy))
    # logger.info("{} gold entities: {}".format
    # logger.info("KB Correct: {}, NIL Correct: {}".format(num_kb_correct / num_kb, num_nil_correct / num_nil))
    # logger.info("Number KB: {}, Number NIL: {}".format(num_kb, num_nil))
    # logger.info("Number correct: {}, Total Queries: {}".format(num_correct, total))
    return num_correct / total, kb_accuracy, num_nil_correct / num_nil



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--feats", action='store_true')
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()
    logging.config.fileConfig('config/logging.conf', disable_existing_loggers=False, defaults={'filename': 'training.log'})
    main(args)