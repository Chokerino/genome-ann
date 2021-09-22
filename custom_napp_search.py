import pickle
import numpy as np
import nmslib
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import glob 
import timeit
import time
import random 
import multiprocessing
import subprocess
from fastaai_testing.fastaai_preprocessing import *
from joblib import Parallel, delayed
from FastAAI_redux.FastAAI import *
from tqdm import tqdm
import pickle
import heapq

numPivot = 15
numPivotIndex = 4
numPivotSearch = 2
num_pivots = 15

num_threads = multiprocessing.cpu_count()

#using fastaai to compute distance between distance and target
def compute_dist(x, y):
    query_protein = "/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/" + x + ".faa"
    query_hmm = "/global/cscratch1/sd/bhavay07/.fastaai_hmms/" + x + ".hmm"
    target_protein = "/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/" + y + ".faa"
    target_hmm = "/global/cscratch1/sd/bhavay07/.fastaai_hmms/" + y + ".hmm"

    shared_opts = [".aai_output", 1, 0]
    query_opts = [None, query_protein, query_hmm]
    target_opts = [None, target_protein, target_hmm]
    q, t, aai = single_query(query_opts, target_opts, shared_opts)
    return [y, x, aai]

#load index and other files
data_napp = np.array(pickle.load(open('/global/cscratch1/sd/bhavay07/fastaai_napp.pkl', 'rb')))
data_pivots = np.array(pickle.load(open('/global/cscratch1/sd/bhavay07/fastaai_pivots.pkl', 'rb')))
data_test = np.array(pickle.load(open('/global/cscratch1/sd/bhavay07/fastaai_testfiles.pkl', 'rb')))

num_data_points = int(len(data_napp)/num_pivots)

#preprocess index into a dictionary
start_total = time.time()
data_dict = {}
start = time.time()
for i in data_napp:
    if i[0] not in data_dict:
        data_dict[i[0]] = [(float(i[2]),i[1])]
    else:
        data_dict[i[0]].append((float(i[2]),i[1]))

#index top numPivotIndex queries
for i in data_dict:
    k_largest = heapq.nlargest(numPivotIndex, data_dict[i])
    data_dict[i] = k_largest
end = time.time()
print("preprocessing time", end - start)

#search the test set for their nearest neighbour
start = time.time()
nearest_neighbors = []
for i in data_test:
    distances = []
    for pivot in data_pivots:
        _, _, aai = compute_dist(i, pivot)
        distances.append((aai, pivot))
    k_largest = heapq.nlargest(numPivotIndex, distances)
    aai_search = Parallel(n_jobs=num_threads)(delayed(compute_dist)(data_point, i) for data_point in tqdm(data_dict) if len(set(map(lambda x: x[1], k_largest)).intersection(set(map(lambda x: x[1], data_dict[data_point])))) >= numPivotSearch)
    nearest = min(aai_search, key = lambda x: x[2])
    nearest_neighbors.append((nearest[1], nearest[2]) if nearest[2] < heapq.heappop(k_largest)[0] else heapq.heappop(k_largest))

end = time.time()
print("search time", end - start)

with open('/global/cscratch1/sd/bhavay07/fastaai_predictions.pkl', 'wb') as f:
    pickle.dump(nearest_neighbors, f, pickle.HIGHEST_PROTOCOL)

nearest_neighbors = np.array(pickle.load(open('/global/cscratch1/sd/bhavay07/fastaai_predictions.pkl', 'rb')))

#brute force search the test set for their nearest neighbour
start = time.time()
brute_nearest_neighbors = []
for i in data_test:
    distances = []
    for pivot in data_pivots:
        _, _, aai = compute_dist(i, pivot)
        distances.append((aai, pivot))
    k_largest = heapq.nlargest(numPivotIndex, distances)
    aai_search = Parallel(n_jobs=num_threads)(delayed(compute_dist)(data_point, i) for data_point in tqdm(data_dict))
    nearest = min(aai_search, key = lambda x: x[2])
    brute_nearest_neighbors.append((nearest[1], nearest[2]) if nearest[2] < heapq.heappop(k_largest)[0] else heapq.heappop(k_largest))
end = time.time()
print("brute search time", end - start)

with open('/global/cscratch1/sd/bhavay07/fastaai_brute_predictions.pkl', 'wb') as f:
    pickle.dump(brute_nearest_neighbors, f, pickle.HIGHEST_PROTOCOL)

#Check performance
count = 0
for n, i in enumerate(brute_nearest_neighbors):
    if i[0] == nearest_neighbors[n][0]:
        count += 1
print(count)
end_total = time.time()
print("total time", end_total - start_total)