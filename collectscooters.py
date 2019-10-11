import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib as plt
from matplotlib import pyplot
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors

data = pd.DataFrame(pd.read_csv(r"database", delimiter = ','));X = pd.DataFrame(zip(data.xcoordinate, data.ycoordinate))
kmeans = KMeans(n_clusters=18); kmeans.fit(X); y_kmeans = kmeans.predict(X); data['Cluster'] = y_kmeans; needscharged = []
for i in data.power_level:
    if int(data.power_level[i]) == 5:
        needscharged.append(False)
    else:
        needscharged.append(True)
clusters = (pd.unique(data.Cluster)); data['Needs_Charged'] = needscharged; position = [20.19, 20.19]; time = 0;van = []
def timetotravel(pt1, pt2):
    global time
    dist = math.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2); addtime = dist/50; time += addtime


def getallincluster(array, pos):
    global van; global time; idx = pd.Index(array.scooter_id)
    s = idx.get_loc(pos, method='ffill'); array.sort_values(by=['ycoordinate', 'xcoordinate'], ascending= False)
    for i in range(0,len(array.scooter_id)):
        van.extend([data.scooter_id[i], time]); pt1 = [array.xcoordinate[s], array.ycoordinate[s]]; pt2 = [array.xcoordinate[i], array.ycoordinate[i]]; timetotravel(pt1, pt2)
        s = i
    return array.scooter_id[(len(array.scooter_id)-1)]


def makecluster(cluster):
    global data
    carray= []
    for i in range(0,len(data.xcoordinate)):
        if data.Cluster[i] == cluster:

            if data.Needs_Charged[i] == True:
                carray.append([data.scooter_id[i],data.xcoordinate[i], data.ycoordinate[i]])
    df = pd.DataFrame({
        'scooter_id': [x[0] for x in carray],
        'xcoordinate': [x[1] for x in carray],
        'ycoordinate': [x[2] for x in carray]})
    return df



timetotravel(position, [data.xcoordinate[25667],data.ycoordinate[25667]])
posidx = 25667
poscluster = data.Cluster[posidx]
cluster0 = makecluster(0);cluster1 = makecluster(1);cluster2 = makecluster(2);cluster3 = makecluster(3); cluster4 = makecluster(4)
cluster5 = makecluster(5); cluster6 = makecluster(6); cluster7 = makecluster(7); cluster8 = makecluster(8); cluster9 = makecluster(9)
cluster10 = makecluster(10); cluster11 = makecluster(11); cluster12 = makecluster(12); cluster13 = makecluster(13)
cluster14 = makecluster(14); cluster15 = makecluster(15); cluster16 = makecluster(16); cluster17 = makecluster(17)
clusters = [cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10,
            cluster11, cluster12, cluster13, cluster14, cluster15, cluster16, cluster17]
visited = []
toprightcorner = []

for i in range(0, 17):
    x = clusters[i].ycoordinate.idxmax
    toprightcorner.append([clusters[i].scooter_id[x], clusters[i].xcoordinate[x],clusters[i].ycoordinate[x]])

def getnextcluster(posidx):
    global data
    global toprightcorner
    global visited
    pt = [data.xcoordinate[posidx], data.ycoordinate[posidx]]
    A = []
    for i in range(0, len(toprightcorner)):
        if i not in visited:
            dist = math.sqrt((pt[0] - toprightcorner[i][1]) ** 2 + (pt[1] - toprightcorner[i][2]) ** 2)
            A.append([dist, i, toprightcorner[i][0]])
    df = pd.DataFrame({
        'Distances': [x[0] for x in A],
        'Cluster':[x[1] for x in A],
        'scooter_id':[x[2] for x in A]})
    x = df.Distances.idxmin
    visited.append(df.Cluster[x])
    id = df.scooter_id[x]
    pt2 = [data.xcoordinate[x], data.ycoordinate[x]]
    timetotravel(pt, pt2)
    nextcluster = df.Cluster[x]
    return id, nextcluster





pos = getallincluster(clusters[poscluster], posidx)
while len(visited) < 17:
    pos, cluster = getnextcluster(pos)
    pos = getallincluster(clusters[cluster], pos)
print(time+5)
