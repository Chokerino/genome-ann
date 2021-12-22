# genome-ann

Approximate Nearest Neighbour for Genomes. Implementation of NAPP with indexing and searching. Extremely fast and accurate search(<1 sec) for nearest neighbours of genomes. 

NAPP relies on 3 parameters namely - 
 - numPivots - Number of pivots from the data points randomly selected.
 - numPivotsIndex - Number of closest pivots that are indexed for each data point.
 - numPivotsSearch - Number of indexed pivots which if match between the query and data point, distance is calculated between them.

The package uses 3 simple steps - 
 - Preprocessing - Initialising the class will allow you to either preprocess a directory containing genomes to proteins and then to best hits to create a database or load an existing database. 
 - Indexing - The pair wise distance between `num_pivot` random datapoints and the remaining points is calculated and the top `num_pivots_index` pivots for each datapoint are stored as a binary string.
 - Searching - For each input genome, ANN search is performed using NAPP. The parameter `num_pivots_search` controls the search time/ accuracy trade-off.


Dependencies - 
  - [Ray](https://github.com/ray-project/ray)
  - [Numpy](https://github.com/numpy/numpy)
  - [tqdm](https://github.com/tqdm/tqdm)
