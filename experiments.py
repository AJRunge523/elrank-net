from collections import Set

# The base set of textual features
nil_set = set([0, 1])
text_set = set([i for i in range(2, 130)])

#Set of NER features
ner_match_set = set([i for i in range(130, 147)])

base_set = ([i for i in range(0, 147)])

in_links = set([147])
out_links = set([148])
in_links_binned = set(i for i in range(149, 159))
out_links_binned = set(i for i in range(159, 169))

all_links_vals = set([147, 148])
all_links_binned = set(i for i in range(149, 169))

all_links = set(i for i in  range(147, 169))

tfidf_val = set([169])
full_emb_val = set([170])
window_emb_val = set([171])
full_emb_tfidf_val = set([172])
window_emb_tfidf_val = set([173])

tfidf_bin = set(i for i in range(174, 179))
full_emb_bin = set(i for i in range(179, 184))
window_emb_bin = set(i for i in range(184, 189))
full_emb_tfidf_bin = set(i for i in range(189, 194))
window_emb_tfidf_bin = set(i for i in range(194, 199))

all_context_vals = set(i for i in range(169, 174))

all_context_bins = set(i for i in range(174, 199))

all_context = set(i for i in range(169, 199))

topic_25_val = set([199, 200, 201])
topic_50_val = set([202, 203, 204])
topic_100_val = set([205, 206, 207])
topic_200_val = set([208, 209, 210])
topic_300_val = set([211, 212, 213])
topic_400_val = set([214, 215, 216])
topic_500_val = set([217, 218, 219])

all_topic_vals = set(i for i in range(199, 218))

topic_25_bin = set(i for i in range(220, 235))
topic_50_bin = set(i for i in range(235, 250))
topic_100_bin = set(i for i in range(250, 265))
topic_200_bin = set(i for i in range(265, 280))
topic_300_bin = set(i for i in range(280, 295))
topic_400_bin = set(i for i in range(295, 310))
topic_500_bin = set(i for i in range(310, 325))

all_topic_bins = set(i for i in range(220, 325))

all_topic = set(i for i in range(199, 325))

coref_match = set([325])
coref_val = set([326])
outref_val = set([327])
inref_val = set([328])
coref_bin = set(i for i in range(329, 336))
outref_bin = set(i for i in range(336, 343))
inref_bin = set(i for i in range(343, 350))

all_coref_vals = set(i for i in range(325, 329))

all_coref_bins = set(i for i in range(329, 350))

all_coref = set(i for i in range(325, 350))

ib_dist_val = set([351, 352, 353])
ib_cn_dist_val = set([354, 355, 356])
ib_dist_bin = set(i for i in range(356, 371))
ib_cn_dist_bin = set(i for i in range(371, 386))

all_ib_vals = set(i for i in range(351, 357))

all_ib_bins = set(i for i in range(356, 386))

all_ib = set(i for i in range(351, 386))

full_set = set(i for i in range(0, 386))

addition_experiments = {}


def add_experiment(name, *args):
    experiment = set()
    for a in args:
        experiment = experiment.union(a)
    addition_experiments[name] = experiment


"""ADDITION EXPERIMENTS"""
add_experiment('text', text_set, nil_set)
add_experiment('base', base_set)

# Link Feature Addition Experiments
add_experiment('in-links', base_set, in_links)
add_experiment('in-links-bin', base_set, in_links_binned)
add_experiment('in-links-all', base_set, in_links, in_links_binned)
add_experiment('out-links', base_set, out_links)
add_experiment('out-links-bin', base_set, out_links_binned)
add_experiment('out-links-all', base_set, out_links, out_links_binned)
add_experiment('all-links', base_set, in_links, out_links)
add_experiment('all-links-bin', base_set, in_links, out_links)
add_experiment('all-links-all', base_set, in_links, out_links)

# Topic Model Addition Experiments
add_experiment('topic-25', base_set, topic_25_val)
add_experiment('topic-25-bin', base_set, topic_25_bin)
add_experiment('topic-25-all', base_set, topic_25_val, topic_25_bin)
add_experiment('topic-50', base_set, topic_50_val)
add_experiment('topic-50-bin', base_set, topic_50_bin)
add_experiment('topic-50-all', base_set, topic_50_val, topic_50_bin)
add_experiment('topic-100', base_set, topic_100_val)
add_experiment('topic-100-bin', base_set, topic_100_bin)
add_experiment('topic-100-all', base_set, topic_100_val, topic_100_bin)
add_experiment('topic-200', base_set, topic_200_val)
add_experiment('topic-200-bin', base_set, topic_200_bin)
add_experiment('topic-200-all', base_set, topic_200_val, topic_200_bin)
add_experiment('topic-300', base_set, topic_300_val)
add_experiment('topic-300-bin', base_set, topic_300_bin)
add_experiment('topic-300-all', base_set, topic_300_val, topic_300_bin)
add_experiment('topic-400', base_set, topic_400_val)
add_experiment('topic-400-bin', base_set, topic_400_bin)
add_experiment('topic-400-all', base_set, topic_400_val, topic_400_bin)
add_experiment('topic-500', base_set, topic_500_val)
add_experiment('topic-500-bin', base_set, topic_500_bin)
add_experiment('topic-500-all', base_set, topic_500_val, topic_500_bin)

# Context Type Addition Experiments
add_experiment('tfidf', base_set, tfidf_val)
add_experiment('tfidf-bin', base_set, tfidf_bin)
add_experiment('tfidf-all', base_set, tfidf_val, tfidf_bin)
add_experiment('full-emb', base_set, full_emb_val)
add_experiment('full-emb-bin', base_set, full_emb_bin)
add_experiment('full-emb-all', base_set, full_emb_val, full_emb_bin)
add_experiment('win-emb', base_set, window_emb_val)
add_experiment('win-emb-bin', base_set, window_emb_bin)
add_experiment('win-emb-all', base_set, window_emb_val, window_emb_bin)
add_experiment('tfidf-full-emb', base_set, full_emb_tfidf_val)
add_experiment('tfidf-full-emb-bin', base_set, full_emb_tfidf_bin)
add_experiment('tfidf-full-emb-all', base_set, full_emb_tfidf_val, full_emb_tfidf_bin)
add_experiment('tfidf-win-emb', base_set, window_emb_tfidf_val)
add_experiment('tfidf-win-emb-bin', base_set, window_emb_tfidf_bin)
add_experiment('tfidf-win-emb-all', base_set, window_emb_tfidf_val, window_emb_tfidf_bin)

# Coref Type Addition Experiments
add_experiment('coref-match', base_set, coref_match)
add_experiment('coref', base_set, coref_val)
add_experiment('coref-bin', base_set, coref_bin)
add_experiment('coref-all', base_set, coref_val, coref_bin)
add_experiment('inref', base_set, inref_val)
add_experiment('inref-bin', base_set, inref_bin)
add_experiment('inref-all', base_set, inref_val, inref_bin)
add_experiment('outref', base_set, outref_val)
add_experiment('outref-bin', base_set, outref_bin)
add_experiment('outref-all', base_set, outref_val, outref_bin)
add_experiment('full-refs', base_set, coref_match, coref_val, inref_val, outref_val)
add_experiment('full-refs-bin', base_set, coref_match, coref_bin, inref_bin, outref_bin)
add_experiment('full-refs-all', base_set, coref_match, coref_val, coref_bin, inref_val, inref_bin, outref_val, outref_bin)

# Entity Context Addition Experiments

add_experiment('info-dist', base_set, ib_dist_val)
add_experiment('info-dist-bin', base_set, ib_dist_bin)
add_experiment('info-dist-all', base_set, ib_dist_val, ib_dist_bin)
add_experiment('info-cn-dist', base_set, ib_cn_dist_val)
add_experiment('info-cn-dist-bin', base_set, ib_cn_dist_bin)
add_experiment('info-cn-dist-all', base_set, ib_cn_dist_val, ib_cn_dist_bin)

# Full set of features
add_experiment('all-feats', full_set)
"""DELETION EXPERIMENTS"""
deletion_experiments = {}


def del_experiment(name, *args):
    experiment = full_set
    for a in args:
        experiment = experiment.difference(a)
    deletion_experiments[name] = experiment


del_experiment('del-text', text_set)
del_experiment('del-ner', ner_match_set)
del_experiment('del-link-vals', all_links_vals)
del_experiment('del-link-binned', all_links_binned)
del_experiment('del-links', all_links)
del_experiment('del-context-vals', all_context_vals)
del_experiment('del-context-bins', all_context_bins)
del_experiment('del-context', all_context)
del_experiment('del-topic-vals', all_topic_vals)
del_experiment('del-topic-bins', all_topic_bins)
del_experiment('del-topics', all_topic)
del_experiment('del-coref-vals', all_coref_vals)
del_experiment('del-coref-bins', all_coref_bins)
del_experiment('del-coref', all_coref)
del_experiment('del-ib-vals', all_coref_vals)
del_experiment('del-ib-bins', all_coref_bins)
del_experiment('del-ib', all_ib)

all_experiments = {}
for key in addition_experiments:
    all_experiments[key] = addition_experiments[key]
for key in deletion_experiments:
    all_experiments[key] = deletion_experiments[key]

final_experiment = {}

def add_final_experiment(name, *args):
    experiment = set()
    for a in args:
        experiment = experiment.union(a)
    final_experiment[name] = experiment


add_final_experiment('final-allcoref', base_set, all_coref, all_ib, tfidf_val, tfidf_bin, full_emb_val,
                     full_emb_bin, window_emb_val, window_emb_bin)
add_final_experiment('final-outlinks-allcoref', base_set, all_coref, all_ib, tfidf_val, tfidf_bin, full_emb_val,
                     full_emb_bin, window_emb_val, window_emb_bin, out_links, out_links_binned)