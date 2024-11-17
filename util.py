import numpy as np
from math import floor, ceil
#helper function to generate splits
def get_splt(tstep = 1000, max_levels = 12, ):
    
    lll = max_levels
    splt = [[] for _ in range(lll+1)]
    
    def gen_half_splits(j=np.arange(5000),lll = 0, levels=12):
        if lll > levels:
            return
        spl = ceil(len(j) / 2)
        splt[lll].append(spl)
        gen_half_splits(j = j[:spl],lll=lll+1,levels=levels)
        gen_half_splits(j = j[spl:],lll=lll+1,levels=levels)
    
    gen_half_splits(np.arange(tstep),lll=0, levels=max_levels)

    for ind, v in enumerate(splt):
        splt[ind] = list(np.cumsum(v*2))

    for ind, v in enumerate(splt):
        v[-1] = tstep-3

    for ind, v in enumerate(splt):
        splt[ind] = [vv for vv in v if vv < tstep]
 
    splt[0] = [splt[0][0]]  
    return splt


def get_sorted_nodes_in_level(nodes, level):
    # get length of time dimension
    start = min([nd.start for nd in nodes])
    stop = max([nd.stop for nd in nodes])
    t = stop - start

    # extract relevant nodes
    nodes = [n for n in nodes if n.level == level]
    nodes = sorted(nodes, key=lambda n: n.bin_num)
    return nodes, t      