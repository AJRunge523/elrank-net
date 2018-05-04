from collections import Set

# The base set of textual features
nil_set = set([0, 1])
text_set = set([i for i in range(2, 130)])

#Set of NER features
ner_match_set = set([i for i in range(130, 147)])

base_set = ner_match_set.union(text_set)

in_links = set([147])
out_links = set([148])
in_links_binned = set(i for i in range(149, 159))
out_links_binned = set(i for i in range(159, 169))

all_links = set(i for i in  range(147, 169))

tfidf_val = set([169])
full_emb_val = set([170])
window_emb_val = set([171])
full_emb_tfidf_val = set([172])
window_emb_tfidf_val = set([173])

tfidf_bin = set(i for i in range(175, 180))
full_emb_bin = set(i for i in range(180, 185))
window_emb_bin = set(i for i in range(185, 190))
full_emb_tfidf_bin = set(i for i in range(190, 195))
window_emb_tfidf_bin = set(i for i in range(195, 200))

all_context = set(i for i in range(169, 200))

topic_25_val = set([174, 200])
topic_50_val = set([201, 202])
topic_100_val = set([203, 204])
topic_200_val = set([205, 206])
topic_300_val = set([207, 208])
topic_400_val = set([209, 210])
topic_500_val = set([211, 212])

topic_25_bin = set(i for i in range(213, 223))
topic_50_bin = set(i for i in range(223, 233))
topic_100_bin = set(i for i in range(233, 243))
topic_200_bin = set(i for i in range(243, 253))
topic_300_bin = set(i for i in range(253, 263))
topic_400_bin = set(i for i in range(263, 273))
topic_500_bin = set(i for i in range(273, 283))

all_topic = set(i for i in range(200, 283))
all_topic.add(174)

coref_match = set([283])
coref_val = set([284])
outref_val = set([285])
inref_val = set([286])
coref_bin = set(i for i in range(287, 294))
outref_bin = set(i for i in range(294, 301))
inref_bin = set(i for i in range(301, 308))

all_coref = set(i for i in range(283, 308))

ib_dist_val = set([308, 309])
ib_cn_dist_val = set([310, 311])
ib_dist_bin = set(i for i in range(312, 322))
ib_cn_dist_bin = set(i for i in range(322, 332))

all_ib = set(i for i in range(308, 332))

full_set = set(i for i in range(0, 332))

addition_experiments = {}


def add_experiment(name, *args):
    experiment = set()
    for a in args:
        experiment = experiment.union(a)
    addition_experiments[name] = experiment


"""ADDITION EXPERIMENTS"""
add_experiment('text', text_set)
add_experiment('base', base_set)

# Link Feature Addition Experiments
add_experiment('in-links', base_set, in_links)
# add_experiment('in-links-bin', base_set, in_links_binned)
add_experiment('in-links-all', base_set, in_links, in_links_binned)
add_experiment('out-links', base_set, out_links)
# add_experiment('out-links-bin', base_set, out_links_binned)
add_experiment('out-links-all', base_set, out_links, out_links_binned)
# add_experiment('all-links', base_set, in_links, out_links)
# add_experiment('all-links-bin', base_set, in_links, out_links)
add_experiment('all-links-all', base_set, in_links, out_links)

# Topic Model Addition Experiments
add_experiment('topic-25', base_set, topic_25_val)
# add_experiment('topic-25-bin', base_set, topic_25_bin)
add_experiment('topic-25-all', base_set, topic_25_val, topic_25_bin)
add_experiment('topic-50', base_set, topic_50_val)
# add_experiment('topic-50-bin', base_set, topic_50_bin)
add_experiment('topic-50-all', base_set, topic_50_val, topic_50_bin)
add_experiment('topic-100', base_set, topic_100_val)
# add_experiment('topic-100-bin', base_set, topic_100_bin)
add_experiment('topic-100-all', base_set, topic_100_val, topic_100_bin)
add_experiment('topic-200', base_set, topic_200_val)
# add_experiment('topic-200-bin', base_set, topic_200_bin)
add_experiment('topic-200-all', base_set, topic_200_val, topic_200_bin)
add_experiment('topic-300', base_set, topic_300_val)
# add_experiment('topic-300-bin', base_set, topic_300_bin)
add_experiment('topic-300-all', base_set, topic_300_val, topic_300_bin)
add_experiment('topic-400', base_set, topic_400_val)
# add_experiment('topic-400-bin', base_set, topic_400_bin)
add_experiment('topic-400-all', base_set, topic_400_val, topic_400_bin)
add_experiment('topic-500', base_set, topic_500_val)
# add_experiment('topic-500-bin', base_set, topic_500_bin)
add_experiment('topic-500-all', base_set, topic_500_val, topic_500_bin)

# Context Type Addition Experiments
add_experiment('tfidf', base_set, tfidf_val)
add_experiment('tfidf-all', base_set, tfidf_val, tfidf_bin)
add_experiment('full-emb', base_set, full_emb_val)
add_experiment('full-emb-all', base_set, full_emb_val, full_emb_bin)
add_experiment('win-emb', base_set, window_emb_val)
add_experiment('win-emb-all', base_set, window_emb_val, window_emb_bin)
add_experiment('tfidf-full-emb', base_set, full_emb_tfidf_val)
add_experiment('tfidf-full-emb-all', base_set, full_emb_tfidf_val, full_emb_tfidf_bin)
add_experiment('tfidf-win-emb', base_set, window_emb_tfidf_val)
add_experiment('tfidf-win-emb-all', base_set, window_emb_tfidf_val, window_emb_tfidf_bin)

# Coref Type Addition Experiments
add_experiment('coref-match', base_set, coref_match)
add_experiment('coref', base_set, coref_val)
add_experiment('coref-all', base_set, coref_val, coref_bin)
add_experiment('inref', base_set, inref_val)
add_experiment('inref-all', base_set, inref_val, inref_bin)
add_experiment('outref', base_set, outref_val)
add_experiment('outref-all', base_set, outref_val, outref_bin)
add_experiment('full-refs', base_set, coref_match, coref_val, inref_val, outref_val)
add_experiment('full-refs-all', base_set, coref_match, coref_val, coref_bin, inref_val, inref_bin, outref_val, outref_bin)

# Entity Context Addition Experiments

add_experiment('info-dist', base_set, ib_dist_val)
add_experiment('info-dist-all', base_set, ib_dist_val, ib_dist_bin)
add_experiment('info-cn-dist', base_set, ib_cn_dist_val)
add_experiment('info-cn-dist-all', base_set, ib_cn_dist_val, ib_cn_dist_bin)

"""DELETION EXPERIMENTS"""
deletion_experiments = {}


def del_experiment(name, *args):
    experiment = full_set
    for a in args:
        experiment = experiment.difference(a)
    deletion_experiments[name] = experiment
    print(experiment)


del_experiment('del-text', text_set)
del_experiment('del-ner', ner_match_set)
del_experiment('del-links', all_links)
del_experiment('del-context', all_context)
del_experiment('del-topics', all_topic)
del_experiment('del-coref', all_coref)
del_experiment('del-ib', all_ib)
