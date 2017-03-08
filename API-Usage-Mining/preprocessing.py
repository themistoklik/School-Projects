import os
from orderedset import OrderedSet as oset
import sys

'''accepts list of filenames to preprocess, outputs sequences for the sequence mining algorithm
 identifier : what to look for in a line to qualify for keeping'''

def preprocess(identifier, filenames):
    count = 0
    total = len(filenames)
    final = []  # holds all raw sequences after preprocessing
    treeDict = {}

    for file in filenames:
        print file
        with open(os.path.join(dirname, file)) as f:
            for line in iter(f.readline, ''):

                contents = line.split()
                # any/all -> [any] is if there's at least 1 weka mention in each line of callgraph maker result, [all] requires callgraph edge nodes to both contain weka in their full names
                # if any([True if identifier in item.lower() else False for item in contents]):
                if identifier in contents[1].lower():

                    # if there aren't 2 tokens in each line, treat file as corrupt
                    assert (len(contents) == 2)

                    # build a caller->calee sequence tree.
                    try:
                        treeDict[contents[0]].append(contents[1])
                    except KeyError:
                        # the first time the key will not exist in dict, so initialize the node
                        treeDict[contents[0]] = [contents[1]]

            # foreach node get all seqs
            seqs = [treeDict[node] for node in treeDict.keys()]
            # no use for single item sequences.
            seqs = filter(lambda item: (len(item) > 1), seqs)

            count += 1

            print "Processed->", (float(count) / float(total)) * 100, "%"

            # extend the list holding all the sequences
            final.extend(seqs)

    # finally remove too lengthy (too specific) and too short (too broad) seqs and seqs with $1 token in them
    final = filter(lambda x: len(x) >= 3, final)
    # additional filtering to remove tokens and cleanup Nulls, and single item seqs
    final = map(lambda x: filter(lambda i: ':' in [j for j in i], x), final)
    final = map(lambda x: filter(lambda i: '$' not in [j for j in i], x), final)
    final = filter(None, final)
    final = map(lambda x: list(oset(x)), final)
    final = filter(lambda x: len(x) > 1, final)
    final = filter(lambda x: ':' in [i for i in x[0]], final)
    print "Final constructed!", len(final), max(map(len, final))

    return final
