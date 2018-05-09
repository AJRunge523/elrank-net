import os
import os.path as path

for folder in os.listdir('output/experiments_v3'):
    scores = {'dev': [0, 0, 0, 0, 0, 0, 0], 'tst': [0, 0, 0, 0, 0, 0, 0]}
    exp_folder = path.join('output/experiments_v3', folder)
    for exp in os.listdir(exp_folder):
        exp_inst_folder = path.join(exp_folder, exp)
        for file in os.listdir(exp_inst_folder):
            if file.endswith('_results.txt'):
                res_type = file.split('_')[0]
                i = 0
                for line in open(path.join(exp_inst_folder, file)):
                    line = line.split(':')
                    if len(line) == 1 or len(line[1].strip()) == 0:
                        continue
                    scores[res_type][i] += float(line[1])
                    i += 1

    for key in scores:
        for i in range(len(scores[key])):
            scores[key][i] /= 5
    devstr = ''
    for v in scores['dev']:
        devstr += str(v) + '\t'
    tststr = ''
    for v in scores['tst']:
        tststr += str(v) + '\t'

    print(folder + '\t' + devstr + '\t\t' + tststr)


