# encoding=utf-8
# rankedlist / unrankedlist structure: list of pairs of (predict_value, real_value)
# unrankedlist_allusers: dict of different users' unrankedlist, keys are the user ID
# metrics include: MAP, nDCG@K, precison@K, MRR
import math
import numpy as np


def cal_dcg_k(rankedlist, k):
    res = 0.0
    for i in xrange(k):
        numerator = 2 ** rankedlist[i][1] - 1.0
        denominator = math.log(2+i, 2)
        res += numerator / denominator

    return res


def cal_ndcg_k(unrankedlist, k):
    rankedlist = sorted(unrankedlist, key=lambda x: x[0], reverse=True)
    dcg = cal_dcg_k(rankedlist, k)

    rankedlist = sorted(unrankedlist, key=lambda x: x[1], reverse=True)
    idcg = cal_dcg_k(rankedlist, k)
    # If all in list are not relevant, the list is not counted
    if idcg == 0:
        return -1
    else:
        return dcg / idcg


def cal_ndcg_all(unrankedlist_allusers, k):  # Cal nDCG@K over all users
    ndcg = 0.0
    users_count = len(unrankedlist_allusers.keys())
    for u in unrankedlist_allusers.keys():
        unrankedlist = unrankedlist_allusers[u]
        if k <= len(unrankedlist) and len(unrankedlist) >= 2:
            res = cal_ndcg_k(unrankedlist, k)
            if res != -1:
                ndcg += res
            else:
                users_count -= 1
        else:
            users_count -= 1
    if users_count <= 0:
        ndcg = -1
    else:
        ndcg /= users_count

    return ndcg


def cal_ndcg_nok(unrankedlist_allusers):  # Cal nDCG over all users
    ndcg = 0.0
    users_count = len(unrankedlist_allusers.keys())
    for u in unrankedlist_allusers.keys():
        unrankedlist = unrankedlist_allusers[u]
        k = len(unrankedlist)
        if k >= 2:
            res = cal_ndcg_k(unrankedlist, k)
            if res != -1:
                ndcg += res
            else:
                users_count -= 1
        else:
            users_count -= 1
    if users_count <= 0:
        ndcg = -1
    else:
        ndcg /= users_count

    return ndcg


def cal_mrr(unrankedlist_allusers):  # Cal MRR over all users
    mrr = 0.0
    users_count = 0
    for u in unrankedlist_allusers.keys():
        unrankedlist = unrankedlist_allusers[u]
        l = len(unrankedlist)
        if l >= 2:
            rankedlist = sorted(unrankedlist, key=lambda x: x[0], reverse=True)
            pos = 1
            for m in xrange(l):
                if rankedlist[m][1] == 1:
                    break
                pos += 1
            if pos != l + 1:
                mrr += 1 / float(pos)
                users_count += 1
    if users_count <= 0:
        mrr = -1
    else:
        mrr /= users_count

    return mrr


def cal_ap(rankedlist):
    totalsum = np.sum(rankedlist, dtype=np.int)
    # If all in list are not relevant, the list is not counted
    if totalsum == 0:
        return -1
    pos_ones = np.nonzero(rankedlist)[0] + 1
    ones = np.array([n for n in range(1, totalsum + 1)])
    ap = ones.astype(np.float) / pos_ones

    return ap.mean()


def cal_map(unrankedlist_allusers):  # Cal MAP over all users
    MAP = 0.0
    users_count = 0
    for u in unrankedlist_allusers.keys():
        rankedlist = unrankedlist_allusers[u]
        if len(rankedlist) >= 2:
            rankedlist = sorted(rankedlist, key=lambda x: x[0], reverse=True)
            rankedlist = np.array(rankedlist)
            ap = cal_ap(rankedlist[:, 1])
            if ap != -1:
                users_count += 1
                MAP += ap
    if users_count <= 0:
        MAP = -1
    else:
        MAP /= users_count

    return MAP


def cal_pn(rankedlist, n):
    ones = 0.0
    for m in xrange(n):
        if rankedlist[m][1] == 1:
            ones += 1

    return ones/float(n)


def cal_precision_N(unrankedlist_allusers, n):  # Cal pre@N over all users
    precision = 0.0
    users_count = len(unrankedlist_allusers.keys())
    for u in unrankedlist_allusers.keys():
        unrankedlist = unrankedlist_allusers[u]
        # If all in list are not relevant, the user is not counted
        totalsum = 0.0
        for m in xrange(len(unrankedlist)):
            totalsum += unrankedlist[m][1]
        if totalsum == 0:
            users_count -= 1
            continue
        rankedlist = sorted(unrankedlist, key=lambda x: x[0], reverse=True)
        if n <= len(rankedlist) and len(rankedlist) >= 2:
            precision += cal_pn(rankedlist, n)
        else:
            users_count -= 1
    if users_count <= 0:
        precision = -1
    else:
        precision /= users_count

    return precision


if __name__ == '__main__':

    uNum = 3
    scorelist = {}
    scorelist[0] = [(0.5, 1), (0.8, 0), (0.2, 0)]
    scorelist[1] = [(0.5, 1), (0.8, 1), (0.2, 0), (0.7, 0)]
    scorelist[2] = [(0.4, 1), (0.5, 1), (0.8, 0), (0.2, 0)]
    s = cal_map(scorelist)
    z = cal_precision_N(scorelist, 3)
    x = cal_ndcg_all(scorelist, 3)
    y = cal_mrr(scorelist)
    print "MAP: ", s
    print "P@3: ", z
    print "nDCG@3: ", x
    print "MRR: ", y


