
base_dir = 'J:/Education/CMU/2018/Spring/Computational Semantics/Project Files/experiment med results/'
base_file = base_dir + 'base_med_tst_eval.txt'


# results_dir = 'J:/Education/CMU/2018/Spring/Computational Semantics/Project Files/experiment med results/'
#
# compare_file = results_dir + 'info-dist-all_med_tst_eval.txt'

results_dir = 'J:/Education/CMU/2018/Spring/Computational Semantics/Project Files/experiment best results/'

compare_file = results_dir + 'final_best_tst_eval.txt'

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

with open(base_file) as base, open(compare_file) as compare:
    line_num = 1
    for base_line, compare_line in zip(base, compare):
        base_line = base_line.strip()
        compare_line = compare_line.strip()
        if base_line.endswith('*'):
            if compare_line.endswith('*'):
                if 'NIL' in compare_line and 'NIL' not in base_line:
                    ambig_to_over += 1
                elif 'NIL' in base_line and 'NIL' not in compare_line:
                    new_ambig_lines.append(str(line_num) + '\t' + compare_line + '\t' + base_line)
                    over_to_ambig += 1
                else:
                    compare_parts = compare_line.split('\t')
                    base_parts = base_line.split('\t')
                    if base_parts[1] != compare_parts[1]:
                        ambig_changed +=1
            else:
                base_parts = base_line.split('\t')
                if base_parts[1] == 'NIL':
                    over_fixed += 1
                elif base_parts[2] == 'NIL':
                    spur_fixed += 1
                else:
                    fixed_ambig_lines.append(str(line_num) + '\t' + compare_line + '\t' + base_line)
                    ambig_fixed += 1
        elif compare_line.endswith('*'):
            compare_parts = compare_line.split('\t')
            if compare_parts[1] == 'NIL':
                new_over += 1
            elif compare_parts[2] == 'NIL':
                new_spur += 1
            else:
                new_ambig += 1
                new_ambig_lines.append(str(line_num) + '\t' + compare_line + '\t' + base_line)
            new_errors += 1

        line_num += 1

total_fixed = spur_fixed + over_fixed + ambig_fixed
total_new = new_spur + new_over + new_ambig
for line in fixed_ambig_lines:
    print(line)
print('======================================================')
for line in new_ambig_lines:
    print(line)
print('======================================================')
print("Spurious fixed: {}, Overlooked fixed: {}, Ambiguous fixed: {}, Ambiguous changed to overlook: {}, Overlook changed to ambiguous: {}, Total: {}"
      .format(spur_fixed, over_fixed, ambig_fixed, ambig_to_over, over_to_ambig, total_fixed))
print("Spurious added: {}, Overlooked added: {}, Ambiguous added: {}, Ambiguous changed: {}, Total: {}".format(new_spur, new_over, new_ambig, ambig_changed, total_new))