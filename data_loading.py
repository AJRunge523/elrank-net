import numpy as np


def make_batches(train_instances, batch_size=64):
    batches = []
    curr_batch = []
    for i in range(len(train_instances)):
        curr_batch.append(train_instances[i])
        if len(curr_batch) == batch_size or i == (len(train_instances) - 1):
            correct_batch = np.array([b[0] for b in curr_batch])
            incorrect_batch = np.array([b[1] for b in curr_batch])
            label_batch = np.array([b[2] for b in curr_batch])
            batches.append((correct_batch, incorrect_batch, label_batch))
            curr_batch.clear()
    return batches


def load_eval_data(eval_file, num_feats, feat_indices):
    instances = []
    loaded = 0
    with(open(eval_file, encoding='utf-8')) as f:
        queries = []
        curr_id = "1"
        for line in f:
            cand_id = ''
            query_type = ''
            loaded += 1
            parts = line.split('#')
            data = parts[0].split()
            annotations = parts[1].split()
            if len(parts) > 1:
                annotations = parts[1].strip().split("_")
                cand_id = annotations[0]
                gold_id = annotations[1]
                gold_ner = annotations[2]
                if len(annotations) > 3:
                    cand_name = annotations[3]
                    gold_name = annotations[4]
            score = data[0]
            query_id = data[1].replace("qid:", "")
            if query_id != curr_id:
                instances.append(queries.copy())
                curr_id = query_id
                queries.clear()
            queries.append((query_id, score, convert(data[2:], num_feats, feat_indices), cand_id,
                            gold_id, gold_ner, cand_name, gold_name))
        instances.append(queries.copy())

    return instances


def load_data(data_file, num_feats, feat_indices):
    instances = []
    loaded = 0
    with(open(data_file, encoding='utf-8')) as f:
        queries = []
        curr_id = "1"
        for line in f:
            loaded += 1
            parts = line.split('#')
            data = parts[0].split()
            score = data[0]
            query_id = data[1].replace("qid:", "")
            if query_id == curr_id:
                queries.append((score, data[2:]))
            else:
                correct = convert(queries[0][1], num_feats, feat_indices)
                for query in queries[1:]:
                    incorrect = convert(query[1], num_feats, feat_indices)
                    instances.append((correct, incorrect, 1.0))
                curr_id = query_id
                queries.clear()
                queries.append((score, data[2:], 1.0))
        correct = convert(queries[0][1], num_feats, feat_indices)
        for query in queries[1:]:
            incorrect = convert(query[1], num_feats, feat_indices)
            instances.append((correct, incorrect, 1.0))
    return instances


def convert(query, num_feats, feat_indices):
    inst = [0.0 for _ in range(num_feats)]
    for feat in query:
        feat_id, feat_val = feat.split(":")
        try:
            feat_id = int(feat_id)
            if feat_id in feat_indices:
                inst[feat_id] = float(feat_val)
        except ValueError:
            inst[int(feat_id)] = 0.0
    return inst
