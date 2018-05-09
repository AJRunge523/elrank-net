
inlink_avg = 0
inlink_num = 0
outlink_avg = 0
outlink_num = 0


for line in open("data/training.dat", encoding='utf8'):
    if line.startswith('2'):
        line = line.strip().split()
        for part in line:
            if part.startswith('147:'):
                part = float(part.split(':')[1])
                inlink_avg += part
                inlink_num += 1
            elif part.startswith('148:'):
                part = float(part.split(':')[1])
                outlink_avg += part
                outlink_num += 1
inlink_avg /= inlink_num
outlink_avg /= outlink_num

print('Average inlink: {}, Average outlink: {}'.format(inlink_avg, outlink_avg))