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
                queries.append((queryId, score, convert(parts[2:], num_feats)))
            else:
                instances.append(queries.copy())
                currId = queryId
                queries.clear()
                queries.append((queryId, score, convert(parts[2:], num_feats)))
    return instances

def load_data(data_file, num_feats):
    instances = []
    loaded = 0
    with(open(data_file)) as f:
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