import math

# ItemCF算法
def ItemSimilarity(train):
    C=dict()
    N=dict()
    for u,item in train.items():
        for i in item.keys():
            N[i]+=1
            for j in item.keys():
                if i==j:
                    continue
                C[i][j]+=1
    W=dict()
    for i,related_items in C.items():
        for j,cij in related_items():
            W[i][j]=cij/math.sqrt(N[i]*N[j])
    return W