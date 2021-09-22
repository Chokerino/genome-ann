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
from fastaai_testing.test import *
from joblib import Parallel, delayed
from FastAAI_redux.FastAAI import *
from tqdm import tqdm
import pickle

numPivot = 15
numPivotIndex = 4
numPivotSearch = 2

#using fastaai to compute distance between distance and target
def compute_dist(x, y):
    global query_time
    global file_time
    global dist_index

    query_protein = "/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/" + x + ".faa"
    query_hmm = "/global/cscratch1/sd/bhavay07/.fastaai_hmms/" + x + ".hmm"
    target_protein = "/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/" + y + ".faa"
    target_hmm = "/global/cscratch1/sd/bhavay07/.fastaai_hmms/" + y + ".hmm"

    shared_opts = [".aai_output", 1, 0]
    query_opts = [None, query_protein, query_hmm]
    target_opts = [None, target_protein, target_hmm]
    q, t, aai = single_query(query_opts, target_opts, shared_opts)
    return [y, x, aai]

dataset = 'typemat'
num_threads = multiprocessing.cpu_count()

if dataset == 'xantho':
    filenames = glob.glob("/global/homes/c/cjain7/shared/projects/ANI/data/Xanthomonas/*.fna")
elif dataset == 'baci':
    filenames = glob.glob("/global/homes/c/cjain7/shared/projects/ANI/data/Bacillus_anthracis/*.fna")
else:
    filenames = glob.glob("/global/homes/c/cjain7/shared/projects/ANI/data/TypeMat/*.fna")

#prepare directories
subprocess.run(["mkdir","/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/"])
subprocess.run(["mkdir","/global/cscratch1/sd/bhavay07/.fastaai_hmms/"])
files_not_done = []
files_done = []
for i in glob.glob('/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/*.faa'):
    files_done.append(i.split("/")[-1][:-4])

for i in filenames:
    if i.split("/")[-1][:-4] not in files_done:
        files_not_done.append(i)

#convert genomes to proteins and hmms
if len(files_not_done) > 0:
    start = time.time()
    Parallel(n_jobs=num_threads)(delayed(genomes_to_protein)(i) for i in files_not_done)
    #genomes_to_protein(files_not_done)
    end = time.time()
    print(f'Preprocessing time to proteins for num threads {num_threads} = {end-start}')
    start = time.time()
    proteins = glob.glob('/global/cscratch1/sd/bhavay07/.fastaai_predicted_proteins/*.faa')
    errors = Parallel(n_jobs=num_threads)(delayed(proteins_to_hmm)(i) for i in proteins)
    #errors = proteins_to_hmm(proteins)
    end = time.time()
    print(f'Preprocessing time to hmms for num threads {num_threads} = {end-start}')


formatted_filenames = []
for i in filenames:
    formatted_filenames.append(i.split("/")[-1][:-4])

#seperate 100 files for testing
test_filenames = list(set(random.sample(formatted_filenames, 100)))
filenames = list(set(formatted_filenames).difference(test_filenames))

#select pivots randomly
pivots = list(set(random.sample(filenames, numPivot)))
filenames = list(set(filenames).difference(pivots))

dataset = []
for i in pivots:
    for j in filenames:
        dataset.append((i, j))

#create index
start = time.time()
aai_opts = Parallel(n_jobs=num_threads)(delayed(compute_dist)(x, y) for x, y in tqdm(dataset))
end = time.time()
print("Indexing Time:", end-start)

#save files
start = time.time()
with open('/global/cscratch1/sd/bhavay07/fastaai_napp.pkl', 'wb') as f:
    pickle.dump(aai_opts, f, pickle.HIGHEST_PROTOCOL)
with open('/global/cscratch1/sd/bhavay07/fastaai_pivots.pkl', 'wb') as f:
    pickle.dump(pivots, f, pickle.HIGHEST_PROTOCOL)
with open('/global/cscratch1/sd/bhavay07/fastaai_testfiles.pkl', 'wb') as f:
    pickle.dump(test_filenames, f, pickle.HIGHEST_PROTOCOL)
end = time.time()
print("Save Time", end-start)
