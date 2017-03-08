'''
Functions to compute various metrics. While argument passing may change depending on implementation, the core logic remains the same.

'''


def cohesion(patterns, threshold=1):
    res=[]
    for p in patterns:
        total = 0
        ratio = 0
        for q in patterns:
            if p!=q:
                if len(set(p)&set(q))>=threshold:
                    r= float(len(set(p)&set(q)))/float(len(set(p)))
                    ratio += r
                    total += 1
        if total:
            res.append((p,float(ratio)/float(total)))
    if res:
        return res
    else:
        return [(0,0.0)]

def dissimilarity(patterns):
    res=[]
    for p in patterns:
        ratio = 0
        for q in patterns:
            if p!=q:
                r = float(len(set(p) & set(q))) / float(len(set(p)))
                ratio += r
        res.append((p,1-float(ratio)/float(len(patterns))))
    return res

rslt=[]


for k in d.keys():

    t = dissimilarity(map(lambda x: x[0], d[k]))
    rr = map(lambda x: x[1] * 100, t)
    rslt.append(sum(rr) / len(rr))
print "avg dissimilarity"
print sum(rslt) / len(rslt)
for i, e in enumerate(rslt):
    print rslt[i]
