
base_dir = 'J:/Education/CMU/2018/Spring/Computational Semantics/Project Files/experiment med results/'
base_file = base_dir + 'base_med_tst_eval.txt'


# results_dir = 'J:/Education/CMU/2018/Spring/Computational Semantics/Project Files/experiment med results/'
#
# compare_file = results_dir + 'info-dist-all_med_tst_eval.txt'

results_dir = 'J:/Education/CMU/2018/Spring/Computational Semantics/Project Files/experiment best results/'

compare_file = results_dir + 'final_best_tst_eval.txt'

gold_file = '../data/eval-gold.txt'

error_counts = {}

types = ['PER', 'ORG', 'GPE']
errors = ['spur', 'over', 'ambig']
for t in types:
    error_counts[t] = {}
    for error in errors:
        error_counts[t][error] = []


spur_fixed = 0
over_fixed = 0
ambig_fixed = 0
ambig_to_over = 0
over_to_ambig = 0
ambig_changed = 0
new_errors = 0
new_spur = 0
new_over = 0
new_ambig = 0

new_ambig_lines = []
fixed_ambig_lines = []

with open(base_file) as base, open(compare_file) as compare, open(gold_file) as gold:
    line_num = 1
    for base_line, compare_line, gold_line in zip(base, compare, gold):
        gold_type = gold_line.split()[2]
        base_line = base_line.strip()
        compare_line = compare_line.strip()
        if compare_line.endswith('*'):
            compare_parts = compare_line.split('\t')
            if compare_parts[1] == 'NIL':
                error_counts[gold_type]['over'].append(compare_line)
            elif compare_parts[2] == 'NIL':
                error_counts[gold_type]['spur'].append(compare_line)
                new_spur += 1
            else:
                error_counts[gold_type]['ambig'].append(compare_line)


for t in types:
    print('============================================{}==================================================='.format(t))
    for e in errors:
        print('{}=========================================='.format(e.upper()))
        for l in error_counts[t][e]:
            print(l)
    print()

total = 0
for t in types:
    print('{}\t'.format(t), end='')
    for e in errors:
        print('{}: {}, '.format(e, len(error_counts[t][e])), end='')
        total += len(error_counts[t][e])
    print()
print('Total: {}'.format(total))