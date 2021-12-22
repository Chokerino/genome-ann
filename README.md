# genome-ann

Approximate Nearest Neighbour for Genomes. Implementation of NAPP with indexing and searching. Extremely fast and accurate search(<1 sec) for nearest neighbours of genomes. 

NAPP relies on 3 parameters namely - 
 - numPivots - Number of pivots from the data points randomly selected. 
 - numPivotsIndex - Number of closest pivots that are indexed for each data point.
 - numPivotsSearch - Number of indexed pivots which if match between the query and data point, distance is calculated between them.

The package uses 3 simple steps - 
 - Preprocessing - Initialising the class will allow you to either preprocess a directory containing genomes to proteins and then to best hits to create a database or load an existing database. 
```
   index = Index(genome_path = "path/to/genomes/", 
              protein_path = "path/to/store/proteins", 
              hmm_path = "path/to/store/hmms",
              database_path = "path/to/store/database",
              database = 'path/to/database/if/previously/created/',
              num_threads = number of threads to use) 
 ```
 - Indexing - The pair wise distance between `num_pivot` random datapoints and the remaining points is calculated and the top `num_pivots_index` pivots for each datapoint are stored as a binary string.
```
index.init_index(num_pivots = number_of_pivots, num_pivots_index = top_k_closest_pivots_tosave)
```
 - Searching - For each input genome, ANN search is performed using NAPP. The parameter `num_pivots_search` controls the search time/ accuracy trade-off.
```
index.query(genomes = list_of_genomes_to_search, num_pivot_search = minimum_number_of_pivots_to_match)
```

Dependencies - 
  - [Ray](https://github.com/ray-project/ray)
  - [Numpy](https://github.com/numpy/numpy)
  - [tqdm](https://github.com/tqdm/tqdm)
