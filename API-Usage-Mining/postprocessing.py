from collections import Counter


# necessary for weka only, left as an example to highlight that different libraries will demand different postprocessing to fine tune results.
def rcore(s):
    for item in s[0]:
        if ')weka.core' not in item:
            return True
    return False


weka = '/resultdirectory'

with open(weka) as f:
    seqs = f.readlines()
    seqs = map(lambda x: x.strip().rsplit(':', 1), seqs)
    seqs = map(lambda x: [x[0].replace('[', '').replace(']', '').replace(' ', '').split(','), int(x[1])], seqs)
    without_core = filter(rcore, seqs)
    for item in without_core[:]:
        for i in item[0][:]:
            if ')weka' not in i:
                item[0].remove(i)
            elif 'println' in i:
                item[0].remove(i)
            elif '$' in i:
                without_core.remove(item)
                break
    g = filter(lambda x: len(x[0]) == 1, without_core)
    without_core = filter(lambda x: len(x[0]) > 3, without_core)


def group(res):
    # set of input
    setres = []
    print len(res)
    for index, item in enumerate(res):
        # drop repeated commands, and keep a tuple (index in original list, actual item)
        if (len(set(item[0])) == len(item[0])):
            setres.append((index, set(item[0])))
        else:
            continue
    # check for subsets, and drop them
    setres = sorted(setres, key=lambda x: len(x[1]), reverse=True)
    print "len is", len(setres)
    setres = filter(lambda x: x[1] is not None, setres)
    print "len is", len(setres)
    for item in setres[:]:
        print len(setres)
        for item1 in setres[:]:
            if item[1] != item1[1]:
                if item[1] >= item1[1]:
                    setres.remove(item1)
                    break
    # return list to original form
    map_to_values = map(lambda x: res[x[0]], setres)
    return map_to_values


grouped_seqs = group(without_core)

d = dict()
# postprocessing based on weka examples, different libraries require different approach
# groups sequences based on the most frequent package name, denoted by the first two tokens (eg. org.classifiers)
grouped_seqs = map(lambda y: map(lambda x: '.'.join(x.split(')')[-1].split('.', 2)[:2]), y[0]), grouped_seqs)
grouped_seqs = map(lambda y: map(lambda x: x.rsplit(':', 1)[0], y), grouped_seqs)
grouped_seqs = map(lambda x: sorted(Counter(x).items(), key=lambda x: x[1])[-1][0], grouped_seqs)

for index, element in enumerate(grouped_seqs):
    try:
        d[element].append(grouped_seqs[index])
    except KeyError:
        d[element] = [grouped_seqs[index]]
#builds dict where keys are package names, and values are all the corresponding sequences sorted by frequency
d = {k: sorted(d[k], key=lambda x: x[1], reverse=True) for k in d.keys()}
print "DONE"

