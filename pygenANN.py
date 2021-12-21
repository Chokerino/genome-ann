import numpy as np
import glob 
import time
import random 
import subprocess
from numpy.lib.function_base import average
from tqdm import tqdm
import pickle
import ray
import heapq
import timeit
from pathlib import Path

class agnostic_reader:
	def __init__(self, file):
		self.path = file
		with open(file, 'rb') as test_gz:
			#Gzip magic number
			is_gz = (test_gz.read(2) == b'\x1f\x8b')
		self.is_gz = is_gz
		if is_gz:
			self.handle = gzip.open(self.path)
		else:
			self.handle = open(self.path)
			
	def __iter__(self):
		return agnostic_reader_iterator(self)
		
	def close(self):
		self.handle.close()

class agnostic_reader_iterator:
	def __init__(self, reader):
		self.handle_ = reader.handle
		self.is_gz_ = reader.is_gz
	def __next__(self):
		if self.is_gz_:
			line = self.handle_.readline().decode()
		else:
			line = self.handle_.readline()
		#Ezpz EOF check
		if line:
			return line
		else:
			raise StopIteration

@ray.remote
def genomes_to_protein(genome, path):
	'''
	FastAAI implementation of converting a genome to protein
	'''
	basename = Path(genome).name
	folder = Path(path)
	protein_output = folder / (basename + '.faa')
	output_11 = folder / (basename + '.faa.11')
	output_4 = folder / (basename + '.faa.4')
	temp_output = folder / (basename + '.temp')
	intermediate = folder / (basename + '_genome_intermediate.fasta')

	genome_parser = agnostic_reader(genome)
	start_time = time.time()
	intermediate = genome
	subprocess.call(["prodigal", "-i", str(intermediate), "-a", str(output_11), "-p", "meta", "-q", "-o", str(temp_output)])
	subprocess.call(["prodigal", "-i", str(intermediate), "-a", str(output_4), "-p", "meta", "-g", "4", "-q", "-o", str(temp_output)])
	#We can get rid of the temp file immediately, we won't be using it
	temp_output.unlink()
	if genome_parser.is_gz:
		#If the file was copied, delete. Otw. this would delete the input and we don't want that.
		intermediate.unlink()
	# Compare translation tables
	length_4 = 0
	length_11 = 0
	with open(output_4, 'r') as table_4:
		for line in table_4:
			if line.startswith(">"):
				continue
			else:
				length_4 += len(line.strip())
	with open(output_11, 'r') as table_11:
		for line in table_11:
			if line.startswith(">"):
				continue
			else:
				length_11 += len(line.strip())
	#Select the winning translation table and remove the other. Open the winner.
	if (length_4 / length_11) >= 1.1:
		output_11.unlink()
		#self.trans_table = "4"
		chosen_protein = open(output_4, 'r')
		table_11 = False
	else:
		output_4.unlink()
		#self.trans_table = "11"
		chosen_protein = open(output_11, 'r')
		table_11 = True
	destination = open(protein_output, "w")
	#Clean the winning output.
	for line in chosen_protein:
		if line.startswith(">"):
			destination.write("{}".format(line))
		else:
			line = line.replace('*', '')
			destination.write("{}".format(line))
	destination.close()
	chosen_protein.close()
	# Remove the winning intermediate file, since we have the cleaned output
	if table_11:
		output_11.unlink()
	else:
		output_4.unlink()
	end = time.time()

#change 'path_to_hmm' to as required
@ray.remote
def proteins_to_hmm(protein, path, path_to_hmm="/global/homes/b/bhavay07/FastAAI_redux/00.Libraries/01.SCG_HMMs/Complete_SCG_DB.hmm"):
	'''
	FastAAI implementation of converting a protein to hmm
	'''
	err_log = ""
	basename = Path(protein).name
	folder = Path(path)
	hmm_output = folder / (basename + '.hmm')
	temp_output = folder / (basename + '.temp')
	intermediate = folder / (basename + '_protein_intermediate.faa')
	current_protein = ""
	current_seq = ""
	protein_parser = agnostic_reader(protein)
	#File was a gzip; decompress it to an intermediate file and then run prodigal; delete after
	#print("unzipping input...")
	#Keeps track of \n chars in the protein sequences.
	line_ct = 0
	midpoint = open(intermediate, "w")
	for line in protein_parser:
		if line.startswith(">"):
			if len(current_seq) > 0:
				if len(current_seq) < 100000:
					midpoint.write(current_protein)
					midpoint.write(current_seq)
				else:
					err_log += "Protein " + current_protein.strip().split()[0][1:] + " was observed to have >100K amino acids ( " + str(len(current_seq) - line_ct) + " AA found ). It was skipped. "
					#print("Protein", current_protein.strip()[1:], "was observed to have >100K amino acids (", len(current_seq) - line_ct, "AA found ).", file = sys.stderr)
					#print("HMMER cannot handle sequences that long, and the protein is almost certainly erroneous, anyway.", file = sys.stderr)
					#print("The protein will be skipped, and FastAAI will continue without it.", file = sys.stderr)
			current_protein = line
			current_seq = ""
			line_ct = 0
		else:
			line_ct += 1
			current_seq += line
	protein_parser.close()
	#Finally, last prot
	if len(current_seq) > 0:
		if len(current_seq) < 100000:
			midpoint.write(current_protein)
			midpoint.write(current_seq)
		else:
			err_log += "Protein " + current_protein.strip().split()[0][1:] + " was observed to have >100K amino acids ( " + str(len(current_seq) - line_ct) + " AA found ). It was skipped. "
			#print("Protein", current_protein.strip()[1:], "was observed to have >100K amino acids (", len(current_seq) - line_ct, "AA found ).", file = sys.stderr)
			#print("HMMER cannot handle sequences that long, and the protein is almost certainly erroneous, anyway.", file = sys.stderr)
			#print("The protein will be skipped, and FastAAI will continue without it.", file = sys.stderr)
	midpoint.close()
	#Should locate the DBs regardless of path.
	script_path = Path(__file__)
	script_dir = script_path.parent
	hmm_complete_model = path_to_hmm
	subprocess.call(["hmmsearch", "--tblout", str(hmm_output), "-o", str(temp_output), "--cut_tc", "--cpu", "1",
					str(hmm_complete_model), str(intermediate)])
	temp_output.unlink()
	intermediate.unlink()

@ray.remote
def prot_and_hmm_to_besthits(protein, hmm):
	'''
	FastAAI implementation of converting protein and hmm to best hits
	'''
	prots = []
	accs = []
	scores = []
	kmer_index = create_kmer_index()
	name = Path(protein).name
	f = agnostic_reader(hmm)
	for line in f:
		if line.startswith("#"):
			continue
		else:
			segs = line.strip().split()
			prots.append(segs[0])
			accs.append(segs[3])
			scores.append(segs[8])
	f.close()
	hmm_file = np.transpose(np.array([prots, accs, scores]))
	
	#hmm_file = np.loadtxt(hmm_file_name, comments = '#', usecols = (0, 3, 8), dtype=(str))
	#Sort the hmm file based on the score column in descending order.
	hmm_file = hmm_file[hmm_file[:,2].astype(float).argsort()[::-1]]
	
	#Identify the first row where each gene name appears, after sorting by score; 
	#in effect, return the highest scoring assignment per gene name
	#Sort the indices of the result to match the score-sorted table instead of alphabetical order of gene names
	hmm_file = hmm_file[np.sort(np.unique(hmm_file[:,0], return_index = True)[1])]
	
	#Filter the file again for the unique ACCESSION names, since we're only allowed one gene per accession, I guess?
	#Don't sort the indices, we don't care about the scores anymore.
	hmm_file = hmm_file[np.unique(hmm_file[:,1], return_index = True)[1]]
	
	best_hits = dict(zip(hmm_file[:,0], hmm_file[:,1]))
	
	best_hits_kmers = {}
	current_seq = ""
	current_prot = ""
	is_besthit = False
	
	protein_count = 0
	protein_kmer_count = {}
	prot = agnostic_reader(protein)
	
	for line in prot:
		if line.startswith(">"):
			if len(current_seq) > 0:
				#print(current_seq, current_prot)
				kmer_set = unique_kmers(current_seq, 4, kmer_index)
				protein_kmer_count[current_prot] = kmer_set.shape[0]
				protein_count += 1
				best_hits_kmers[current_prot] = kmer_set
			#Select the best hit accession for this protein and just record that. We do not care about the names of the proteins.
			current_prot = line[1:].strip().split(" # ")[0]
			if current_prot in best_hits:
				current_prot = best_hits[current_prot]
				is_besthit = True
			else:
				is_besthit = False
			current_seq = ""
		else:
			if is_besthit:
				current_seq += line.strip()
			
	prot.close()

	#Final iter. doesn't happen otw.
	if current_prot in best_hits:
		kmer_set = unique_kmers(current_seq, 4)
		#kmer_set = [kmer_index[k] for k in  kmer_set]
		protein_kmer_count[current_prot] = kmer_set.shape[0]
		protein_count += 1
		best_hits_kmers[current_prot] = kmer_set
	
	return [name, protein_kmer_count, protein_count, best_hits_kmers]

def unique_kmers(seq, ksize, kmer_index):
	n_kmers = len(seq) - ksize + 1
	kmers = []
	for i in range(n_kmers):
		kmers.append(kmer_index[seq[i:i + ksize]])
	#We care about the type because we're working with bytes later.
	return np.unique(kmers).astype(np.int32)

def create_kmer_index():
	valid_chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', '*']
	#This meshgrid method will produce all unique tetramers from AAAA to **** in a consistent order.
	#Rightmost char to leftmost, A to * in the same order as valid_chars
	kmer_index_ = np.stack(np.meshgrid(valid_chars, valid_chars, valid_chars, valid_chars), -1).reshape(-1, 4)
	#Unless someone is passing more than 2.1 billion genomes, int32 will be enough.
	kmer_index_ = dict(zip([''.join(kmer_index_[i,]) for i in range(0, kmer_index_.shape[0])], np.arange(kmer_index_.shape[0], dtype = np.int32)))
	
	return kmer_index_

def intersect_kmer_lists(pair):
		#intersection = np.intersect1d(pair[0], pair[1]).shape[0]
		intersection = len(set(pair[0]).intersection(set(pair[1]))) #faster
		union = pair[0].shape[0] + pair[1].shape[0] - intersection
		return (intersection/union)
 
def single_query_batch(query, targets):
	'''
	Helper function for query_batch which computes FastAAI for a given set of genomes as a single step.
	'''
	query_accs = list(query[1].keys())
	num_targets= len(targets)
	gak = []
	for i in targets:
		gak.append(i[0])
	target_accs = []
	for i in targets:
		target_accs.extend(list(i[1].keys()))
	target_accs = list(set(target_accs))
	target_kmer_cts = {}
	for accession in np.intersect1d(query_accs, target_accs):
			target_kmer_cts[accession] = np.zeros(num_targets, dtype = np.int16)
			for n,g in enumerate(gak):
				if accession in targets[n][1]:
					target_kmer_cts[accession][n] = len(targets[n][1][accession])

	results = np.zeros(shape = (len(query_accs), num_targets), dtype = np.float64)
	shared_acc_counts = np.zeros(num_targets, dtype = np.int16)
	row = 0
	for accession in query_accs:
		if accession in target_accs:
			#The accession was found for this target genome, for each tgt genome.
			shared_acc_counts[np.nonzero(target_kmer_cts[accession])] += 1
			these_intersections = np.zeros(num_targets, dtype = np.int16)
			for i in range(num_targets):
				if accession in targets[i][1]:
					these_intersections[i] = len(set(query[1][accession]).intersection(set(targets[i][1][accession])))
			results[row] = np.divide(these_intersections, np.subtract(np.add(query[1][accession].shape[0], np.asarray(list(target_kmer_cts[accession]))), these_intersections))
		row+=1
	jaccard_averages = np.divide(np.mean(results, axis = 0), shared_acc_counts)
	non_zero = np.where(jaccard_averages > 0)
	is_zero  = np.where(jaccard_averages <= 0)
	jaccard_averages[non_zero] = np.log(jaccard_averages[non_zero])
	aai_est = np.multiply(np.subtract(np.multiply(np.exp(np.negative(np.power(np.multiply(jaccard_averages, -0.2607023), (1/3.435)))), 1.810741), 0.3087057), 100)
	return [(i[0], 200.0 - aai_est[n]) for n,i in enumerate(targets)] 

@ray.remote
def query_batch(database, genome, num_pivot_search):
	'''
	Function to compute the ANN given the database, genome and the number of pivots to be searched.
	'''
	distances = []
	start_sample_total = timeit.default_timer()
	start_sample_cpu = time.process_time()
	put_file = database['Data'][f"{genome}.fna.faa"]['best_hits']
	results = single_query_batch([genome, put_file], [(pivot, database['Data'][pivot]['best_hits']) for pivot in database['Pivots']])
	results2 = []
	distances = [(x[1], x[0]) for x in results] #value, pivot
	k_largest = heapq.nlargest(database['num_pivots_index'], distances)
	#convert to bitwise
	bitwise = np.zeros(database['num_pivots'], dtype=int)
	for j in k_largest:
		bitwise[np.where(np.asarray(database["Pivots"]) == j[1])] = 1
	bitwise = ''.join(str(x) for x in bitwise)
	query_data = []
	for data_point in database['Data']:
		if data_point not in database['Pivots'] and bin(int(bitwise,2)&int(database['Pairwise'][data_point][1], 2)).count('1') >= num_pivot_search:
			query_data.append((data_point, database['Data'][data_point]['best_hits']))
	aai_search = single_query_batch([genome, put_file], query_data)#bitwise #fastest
	try:
		nearest = min(aai_search, key = lambda x: x[1])
	except:
		pass
	closest_pivot = heapq.heappop(k_largest)
	if len(aai_search) == 0:
		nearest_neighbors = (closest_pivot[1], closest_pivot[0])
	else:
		nearest_neighbors = (nearest[0], nearest[1]) if nearest[1] < closest_pivot[0] else (closest_pivot[1], closest_pivot[0])
	end_sample_cpu = time.process_time()
	end_sample_total = timeit.default_timer()
	return nearest_neighbors, end_sample_total-start_sample_total, end_sample_cpu - start_sample_cpu

@ray.remote
def single_query(q, query, t, target):
	'''
	Function to compute FastAAI between a pair of genomes.
	'''
	accs_to_view = set(query.keys()).intersection(set(target.keys()))
	results = [intersect_kmer_lists([query[acc], target[acc]]) for acc in accs_to_view] #faster
	jacc_mean = np.mean(results)
	aai_est = (-0.3087057 + 1.810741 * (np.exp(-(-0.2607023 * np.log(jacc_mean))**(1/3.435))))*100
	return q, t, 200 - aai_est

class Index():
	def __init__(self, genome_path = None, protein_path = None, hmm_path = None, database = None, database_path = None, num_threads = None):
		'''
		Initialise the Index
		Arguments:
			- genome_path: Path to the directory where genomes are stored
			- protein_path: Path to the directory where proteins are to be stored. If None, a new directory is created in the genome_path
			- hmm_path: Path to the directory where hmms are to be stored. If None, a new directory is created in the genome_path
			- database: Path to the database if the Index has been previously created
			- database_path: Path to the where the database is to be stored.
			- num_threads: Number of threads to use. By default will use all available
		'''
		ray.init(num_cpus = num_threads, include_dashboard = False)
		self.num_threads = num_threads
		self.genome_path = Path(genome_path)
		if database != None:
			self.database = pickle.load(open(database, 'rb'))
			print(f"Database loaded from {self.database['Path']}")
		else:
			database = f"{self.genome_path.name}_database.pkl"
			if protein_path == None:
				subprocess.run(["mkdir", f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins"])
				self.protein_path = Path(f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins")
			else:
				if Path(protein_path).is_dir():
					self.protein_path = Path(protein_path)
				else:
					subprocess.run(["mkdir", protein_path])
					self.protein_path = Path(protein_path)
			if hmm_path == None:
				subprocess.run(["mkdir", f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins"])
				self.hmm_path = Path(f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins")
			else:
				if Path(hmm_path).is_dir():
					self.hmm_path = Path(hmm_path)
				else:
					subprocess.run(["mkdir", hmm_path])
					self.hmm_path = Path(hmm_path)
			raw_database = self.preprocess(glob.glob(f"{self.genome_path}/*.fna"))
			self.database = {'Path':f'{Path(database_path)}/{database}', 'Pivots': None, 'Data': {}, 'Pairwise': None}
			for i in raw_database:
				self.database['Data'][i[0]] = {'best_hits': i[3]}
			self.save_database()
			print(f"Database saved at {self.database['Path']}")
			del raw_database

	def save_database(self):
		'''
		Save the database at the path specified.
		'''
		with open(self.database['Path'], 'wb') as f:
			pickle.dump(self.database, f, pickle.HIGHEST_PROTOCOL)

	def add_genomes(self, genome_file = None, protein_path = None, hmm_path = None):
		'''
		Add new genomes to the database.
		Arguments:
			- genome_file: File containing the path to the genomes. 1 per line.
			- protein_path: Path to the directory where proteins are to be stored. If None, a new directory is created in the genome_path
			- hmm_path: Path to the directory where hmms are to be stored. If None, a new directory is created in the genome_path
		'''
		if protein_path == None:
				subprocess.run(["mkdir", f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins"])
				self.protein_path = Path(f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins")
		else:
			if Path(protein_path).is_dir():
				self.protein_path = Path(protein_path)
			else:
				subprocess.run(["mkdir", protein_path])
				self.protein_path = Path(protein_path)
		if hmm_path == None:
			subprocess.run(["mkdir", f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins"])
			self.hmm_path = Path(f"{self.genome_path.parent}/{self.genome_path.parent.name}_proteins")
		else:
			if Path(hmm_path).is_dir():
				self.hmm_path = Path(hmm_path)
			else:
				subprocess.run(["mkdir", hmm_path])
				self.hmm_path = Path(hmm_path)
		with open(genome_file, 'r') as fp:
			genome_files = [x[:-1] for x in fp.readlines()]
		raw_database = self.preprocess(genome_files, flag=1)
		for i in raw_database:
			self.database['Data'][i[0]] = {'best_hits': i[3]}
		self.save_database()

	def preprocess(self, genomes, flag = 0):
		'''
		Preprocess the genomes by converting them to proteins, hmms and best hits.
		'''
		start_total = timeit.default_timer()
		start_cpu = time.process_time()
		ray.get([genomes_to_protein.remote(i, self.protein_path) for i in tqdm(genomes)])
		end_cpu = time.process_time()
		end_total = timeit.default_timer()
		print(f'Preprocessing Wall time to proteins for num threads {self.num_threads} = {end_total-start_total}, CPU Time = {end_cpu - start_cpu}')
		if flag == 0:
			protein_files = glob.glob(f"{self.protein_path}/*.faa")
		else:
			protein_files = [f"{self.protein_path}/{Path(x).name}.faa" for x in genomes]
		start_total = timeit.default_timer()
		start_cpu = time.process_time()
		ray.get([proteins_to_hmm.remote(i, self.hmm_path) for i in tqdm(protein_files)])
		end_cpu = time.process_time()
		end_total = timeit.default_timer()
		print(f'Preprocessing time to hmms for num threads {self.num_threads} = {end_total-start_total}, CPU Time = {end_cpu - start_cpu}')
		if flag == 0:
			hmm_files = glob.glob(f"{self.hmm_path}/*.hmm")
		else:
			hmm_files = [f"{self.hmm_path}/{Path(x).name}.faa.hmm" for x in genomes]
		start_total = timeit.default_timer()
		start_cpu = time.process_time()
		if flag == 0:
			raw_database = ray.get([prot_and_hmm_to_besthits.remote(i, f"{self.hmm_path}/{Path(i).name}.hmm") for i in tqdm(protein_files)])
		else:
			raw_database = ray.get([prot_and_hmm_to_besthits.remote(i, f"{self.hmm_path}/{Path(i).name}.hmm") for i in tqdm(protein_files)])
		end_cpu = time.process_time()
		end_total = timeit.default_timer()
		print(f'Preprocessing time to proteins and hmms to best hits for num threads {self.num_threads} = {end_total-start_total}, CPU Time = {end_cpu - start_cpu}')
		return raw_database
	
	def to_list(self, genomes = None):
		'''
		Helper Function to read genomes.
		'''
		if isinstance(genomes, str):
			if Path(genomes).is_dir():
				genomes = glob.glob(f"{Path(genomes)}/*.fna")
			else:
				with open(genomes,"r") as fp:
					genomes = fp.readlines()
					fp.close()
		return genomes
		
	def init_index(self, num_pivots, num_pivots_index):
		'''
		This function initiates the index creation. It calculates pairwise distance between the pivots 
		and the datapoints then stores the top k distances as permutation which can then be used to search for the nearest neighbour.
		Arguments:
			- num_pivots: Number of random pivots to choose from the database
			- num_pivots_index: Number of closest pivots to save for each data point.
		'''
		self.database['num_pivots'] = num_pivots
		self.database['num_pivots_index'] = num_pivots_index
		self.database["Pivots"] = np.asarray(list(set(random.sample(list(self.database["Data"].keys()), num_pivots))))
		filenames = list(set(list(self.database["Data"].keys())).difference(self.database["Pivots"]))
		print(f"---------{num_pivots} randomly sampled---------")
		self.put_data()
		print(f"---------Best Hits transferred to memory---------")
		dataset = []
		for i in filenames:
			for j in self.database["Pivots"]:
				dataset.append((i, j))
		start_total = timeit.default_timer()
		start_cpu = time.process_time()
		pairwise_distance = ray.get([single_query.remote(x, self.database['Data_Ray'][x], y, self.database['Data_Ray'][y]) for x, y in tqdm(dataset)])
		end_cpu = time.process_time()
		end_total = timeit.default_timer()
		print(f"Indexing Wall Time: {end_total-start_total}, CPU Time: {end_cpu - start_cpu} ")
		print("---------Preprocessing the distances to the pivots---------")
		self.database["Pairwise"] = {}
		for i in pairwise_distance:
			if i[0] not in self.database["Pairwise"]:
				self.database["Pairwise"][i[0]] = [(float(i[2]), i[1])] #aai value, pivot
			else:
				self.database["Pairwise"][i[0]].append((float(i[2]), i[1]))
		for i in self.database["Pairwise"]:
			k_largest = heapq.nlargest(num_pivots_index, self.database["Pairwise"][i])
			bitwise = np.zeros(num_pivots, dtype=int)
			for j in k_largest:
				bitwise[np.where(self.database["Pivots"] == j[1])] = 1
			self.database["Pairwise"][i] = [k_largest,''.join(str(x) for x in bitwise)]
		del self.database['Data_Ray']
		self.save_database()
		print("---------Database updated with pairwise distances with pivots---------")
	
	def put_data(self):
		'''
		Transfer data to memory using Ray.
		'''
		self.database['Data_Ray'] = {}
		for i in self.database['Data']:
			if isinstance(self.database['Data'][i], tuple):
				self.database['Data_Ray'][i] = ray.put(self.database['Data'][i]['best_hits'][0])
			else:
				self.database['Data_Ray'][i] = ray.put(self.database['Data'][i]['best_hits'])
	
	def query(self, genomes, num_pivot_search):
		'''
		This function performs ANN for the given input genomes. The pairwise FastAAI value calculation is parallelized while each genome is input sequentially.
		Arguments:
			- genomes: Name of the genome in the database to be searched.
			- num_pivot_search: Minimum number of indexed pivot permutations that need to be matched.
		'''
		time_list_cpu = []
		time_list_total = []
		nearest_neighbors = []
		genomes = self.to_list(genomes)
		start_total = timeit.default_timer()
		self.put_data()
		end_total = timeit.default_timer()
		print(f"---------Best Hits transferred to memory in {end_total-start_total}---------")
		start_total = timeit.default_timer()
		start_cpu = time.process_time()
		for n, i in enumerate(genomes):
			distances = []
			start_sample_total = timeit.default_timer()
			start_sample_cpu = time.process_time()
			put_file = self.database['Data_Ray'][f"{Path(i).name}.faa"]
			results = ray.get([single_query.remote(i, put_file, pivot, self.database['Data_Ray'][pivot]) for pivot in tqdm(self.database['Pivots'])])
			distances = [(x[2], x[1]) for x in results] #value, pivot
			k_largest = heapq.nlargest(self.database['num_pivots_index'], distances)
			#convert to bitwise
			bitwise = np.zeros(self.database['num_pivots'], dtype=int)
			for j in k_largest:
				bitwise[np.where(np.asarray(self.database["Pivots"]) == j[1])] = 1
			bitwise = ''.join(str(x) for x in bitwise)
			#print("genome", i, bitwise,int(self.database['Pairwise'][f"{i}.fna.faa"][1], 2))
			aai_search = ray.get([single_query.remote(i, put_file, data_point, self.database['Data_Ray'][data_point]) for data_point in tqdm(self.database['Data_Ray']) if data_point not in self.database['Pivots'] and bin(int(bitwise,2)&int(self.database['Pairwise'][data_point][1], 2)).count('1') >= num_pivot_search])#bitwise #fastest
			#print([data_point for data_point in tqdm(self.database['Data_Ray']) if data_point not in self.database['Pivots'] and bin(int(bitwise,2)&int(self.database['Pairwise'][data_point][1], 2)).count('1') >= num_pivot_search ])
			try:
				nearest = min(aai_search, key = lambda x: x[2])
			except:
				pass
			closest_pivot = heapq.heappop(k_largest)
			if len(aai_search) == 0:
				nearest_neighbors.append((closest_pivot[1], closest_pivot[0])) #value, pivot
			else: #value, pivot
				nearest_neighbors.append((nearest[1], nearest[2]) if nearest[2] < closest_pivot[0] else (closest_pivot[1], closest_pivot[0]))
			del put_file
			end_sample_cpu = time.process_time()
			end_sample_total = timeit.default_timer()
			time_list_total.append(end_sample_total-start_sample_total)
			time_list_cpu.append(end_sample_cpu - start_sample_cpu)
		end_cpu = time.process_time()
		end_total = timeit.default_timer()
		count = 0
		for n, i in enumerate(genomes):
			if i == nearest_neighbors[n][0][:-8]:
				count+=1
		print("Accuracy", count)
		print(f"Search Wall Time: {end_total-start_total}, CPU Time: {end_cpu - start_cpu} ")
		print(f"Search Minimum Wall Time: {min(time_list_total)}, Search Minimum CPU Time: {min(time_list_cpu)} ")
		print(f"Search Average Wall Time: {average(time_list_total)}, Search Average CPU Time: {average(time_list_cpu)} ")
		print(f"Search Maximum Wall Time: {max(time_list_total)}, Search Maximum CPU Time: {max(time_list_cpu)} ")
		print(f"Search Sum Wall Time: {sum(time_list_total)}, Search Sum CPU Time: {sum(time_list_cpu)} ")
	
	def query_batch(self, genomes, num_pivot_search):
		'''
		This function performs ANN for the given input genomes. The pairwise FastAAI value calculation is as a sequential batch while calculation for each genome parallelized.
		Arguments:
			- genomes: Name of the genome in the database to be searched.
			- num_pivot_search: Minimum number of indexed pivot permutations that need to be matched.
		'''
		time_list_cpu = []
		time_list_total = []
		nearest_neighbors = []
		genomes = self.to_list(genomes)
		start_total = timeit.default_timer()
		#self.put_data()
		db = ray.put(self.database)
		end_total = timeit.default_timer()
		print(f"---------Best Hits transferred to memory in {end_total-start_total}---------")
		start_total = timeit.default_timer()
		start_cpu = time.process_time()
		query_args = []
		aai_search = []
		aai_search = ray.get([query_batch.remote(db, genome, num_pivot_search) for genome in genomes])#bitwise #fastest
		end_cpu = time.process_time()
		end_total = timeit.default_timer()
		print(f"Search Wall Time: {end_total-start_total}, CPU Time: {end_cpu - start_cpu} ")
		count = 0
		for n, i in enumerate(genomes):
			if i == aai_search[n][0][0][:-8]:
				count+=1
		print("Accuracy", count)