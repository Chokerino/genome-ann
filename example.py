from pygenANN import Index
import glob
from pathlib import Path
import random

index = Index(genome_path = "/global/homes/c/cjain7/shared/projects/ANI/data/GTDB_SPECIES/", 
              protein_path = "/global/cscratch1/sd/bhavay07/GTDB/proteins", 
              hmm_path = "/global/cscratch1/sd/bhavay07/GTDB/hmms",
              database_path = "/global/cscratch1/sd/bhavay07/GTDB/",
              num_threads = 10) #Use this to create a new database

index = Index(genome_path = "/global/homes/c/cjain7/shared/projects/ANI/data/GTDB_SPECIES/", 
              protein_path = "/global/cscratch1/sd/bhavay07/GTDB/proteins", 
              hmm_path = "/global/cscratch1/sd/bhavay07/GTDB/hmms",
              database_path = "/global/cscratch1/sd/bhavay07/GTDB/",
              database = '/global/cscratch1/sd/bhavay07/GTDB/GTDB_SPECIES_database.pkl',
              num_threads = 10) #Use this to load database

#initiate index
index.init_index(num_pivots = 100, num_pivots_index = 25)
#add new genomes. 
index.add_genomes(genome_file = 'fastaai_files/target.txt',protein_path = "/global/cscratch1/sd/bhavay07/GTDB/proteins", hmm_path = "/global/cscratch1/sd/bhavay07/GTDB/hmms",)
#Select 100 random genomes from the database
random_genomes = random.sample(list(set(list(index.database["Data"].keys())).difference(index.database["Pivots"])), 100)
random_genomes = [x[:-8] for x in random_genomes]
#Search nearest neighbour
index.query(genomes=random_genomes, num_pivot_search = 21)