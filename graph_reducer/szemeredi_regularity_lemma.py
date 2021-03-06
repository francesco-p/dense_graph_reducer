import random
import numpy as np

from classes_pair import ClassesPair
from classes_pair import WeightedClassesPair


class SzemerediRegularityLemma:
    # methods required by the algorithm. User can provide its own implementations provided that they respect the input/output conventions
    partition_initialization = None
    """The method used to build the initial partition"""
    refinement_step = None
    """The method used to refine the current partition"""
    conditions = []
    """The conditions used to check the regularity/irregularity of the pairs"""

    # main data structure
    sim_mat = np.empty((0, 0), dtype='float32')
    """The similarity matrix representing the graph (used only if is_weighted is set to True)"""
    adj_mat = np.empty((0, 0), dtype='float32')
    """The adjacency matrix representing the graph"""
    reduced_sim_mat = np.empty((0, 0), dtype='float32')
    """the resulting similarity matrix"""
    classes = np.empty((0, 0), dtype='int16')
    """array with size equal to the number of nodes in the graph. Each element is set to the class whose node belongs"""
    degrees = np.empty((0, 0), dtype='int16')
    """array containing the indices of the nodes ordered by the degree"""

    # main parameters of the algorithm
    N = 0
    """number of nodes in the graph"""
    k = 0
    """number of classes composing the partition"""
    classes_cardinality = 0
    """cardinality of each class"""
    epsilon = 0.0
    """epsilon parameter"""
    index_vec = []
    """index measuring the goodness of the partition"""

    # auxiliary structures used to keep track of the relation between the classes of the partition
    certs_compls_list = []
    """structure containing the certificates and complements for each pair in the partition"""
    regularity_list = []
    """list of lists of size k, each element i contains the list of classes regular with class i"""

    # flags to specify different behaviour of the algorithm
    is_weighted = False
    """flag to specify if the graph is weighted or not"""
    drop_edges_between_irregular_pairs = False
    """flag to specify if the reduced matrix is fully connected or not"""

    # debug structures
    condition_verified = []
    """this attribute is kept only for analysis purposes. For each iteration it stores the number of times that
       condition 1/condition 2/condition 3/no condition has been verified"""

    def __init__(self, sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs):
        if is_weighted:
            self.sim_mat = sim_mat
        self.adj_mat = (sim_mat > 0.0).astype('float32')
        self.epsilon = epsilon
        self.N = self.adj_mat.shape[0]
        self.degrees = np.argsort(self.adj_mat.sum(0))

        # flags
        self.is_weighted = is_weighted
        self.drop_edges_between_irregular_pairs = drop_edges_between_irregular_pairs

    def generate_reduced_sim_mat(self):
        """
        generate the similarity matrix of the current classes
        :return sim_mat: the reduced similarity matrix
        """
        self.reduced_sim_mat = np.zeros((self.k, self.k), dtype='float32')

        for r in range(2, self.k + 1):
            for s in (range(1, r) if not self.drop_edges_between_irregular_pairs else self.regularity_list[r - 2]):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)
                self.reduced_sim_mat[r - 1, s - 1] = self.reduced_sim_mat[s - 1, r - 1] = cl_pair.bip_density

    def reconstruct_original_mat(self, thresh, intracluster_weight=0):
        """
        reconstruct a similarity matrix with size equals to the original one, from the reduced similarity matrix
        :param thresh: a threshold parameter to prune the edges of the graph
        """
        reconstructed_mat = np.zeros((self.N, self.N), dtype='float32')

        r_nodes = self.classes > 0
        reconstructed_mat[np.ix_(r_nodes, r_nodes)] = intracluster_weight 

        for r in range(2, self.k + 1):
            r_nodes = self.classes == r
            reconstructed_mat[np.ix_(r_nodes, r_nodes)] = intracluster_weight
            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                s_nodes = self.classes == s
                if cl_pair.bip_density > thresh:
                    reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = cl_pair.bip_density
        np.fill_diagonal(reconstructed_mat, 0.0)
        return reconstructed_mat


    def check_pairs_regularity(self):
        """
        perform step 2 of Alon algorithm, determining the regular/irregular pairs and their certificates and complements
        :return certs_compls_list: a list of lists containing the certificate and complement
                for each pair of classes r and s (s < r). If a pair is epsilon-regular
                the corresponding complement and certificate in the structure will be set to the empty lists
        :return num_of_irregular_pairs: the number of irregular pairs
        """
        # debug structure
        self.condition_verified = [0] * (len(self.conditions) + 1)

        num_of_irregular_pairs = 0
        index = 0.0

        for r in range(2, self.k + 1):
            self.certs_compls_list.append([])
            self.regularity_list.append([])

            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                is_verified = False
                for i, cond in enumerate(self.conditions):
                    is_verified, cert_pair, compl_pair = cond(self, cl_pair)
                    if is_verified:
                        self.certs_compls_list[r - 2].append([cert_pair, compl_pair])

                        if cert_pair[0]:
                            num_of_irregular_pairs += 1
                        else:
                            self.regularity_list[r - 2].append(s)

                        self.condition_verified[i] += 1
                        break

                if not is_verified:
                    # if no condition was verified then consider the pair to be regular
                    self.certs_compls_list[r - 2].append([[[], []], [[], []]])
                    self.condition_verified[-1] += 1

                index += cl_pair.compute_bip_density() ** 2.0

        index *= (1.0 / self.k ** 2.0)
        self.index_vec.append(index)
        return num_of_irregular_pairs

    def check_partition_regularity(self, num_of_irregular_pairs):
        """
        perform step 3 of Alon algorithm, checking the regularity of the partition
        :param num_of_irregular_pairs: the number of found irregular pairs in the previous step
        :return: True if the partition is irregular, False otherwise
        """
        return num_of_irregular_pairs <= self.epsilon * ((self.k * (self.k - 1)) / 2.0)

    def run(self, b=2, compression_rate=0.05, iteration_by_iteration=False, verbose=False):
        """
        run the Alon algorithm.
        :param b: the cardinality of the initial partition (C0 excluded)
        :param compression_rate: the minimum compression rate granted by the algorithm, if set to a value in (0.0, 1.0]
                                 the algorithm will stop when k > int(compression_rate * |V|). If set to a value > 1.0
                                 the algorithm will stop when k > int(compression_rate)
        :param iteration_by_iteration: if set to true, the algorithm will wait for a user input to proceed to the next
               one
        :param verbose: if set to True some debug info is printed
        :return the reduced similarity matrix
        """
        np.random.seed(314)
        random.seed(314)

        if 0.0 < compression_rate <= 1.0:
            max_k = int(compression_rate * self.N)
        elif compression_rate > 1.0:
            max_k = int(compression_rate)
        else:
            raise ValueError("incorrect compression rate. Only float greater than 0.0 are accepted")

        iteration = 0
        
        self.partition_initialization(self, b)
        
        while True:
            self.certs_compls_list = []
            self.regularity_list = []
            self.condition_verified = [0] * len(self.conditions)
            iteration += 1
            num_of_irregular_pairs = self.check_pairs_regularity()
            total_pairs = (self.k * (self.k - 1)) / 2.0

            if self.check_partition_regularity(num_of_irregular_pairs):
                if verbose:
                    print("{0}, {1}, {2}, {3}, regular".format(iteration, self.k, self.classes_cardinality, self.index_vec[-1]))
                    
                break

            if self.k >= max_k:
                if verbose:
                    # Cardinality too low irregular by definition
                    print("{0}, {1}, {2}, {3}, irregular".format(iteration, self.k, self.classes_cardinality, self.index_vec[-1]))
                return (False, self.k, None)
                
            if verbose:
                print("{0}, {1}, {2}, {3} irregular".format(iteration, self.k, self.classes_cardinality, self.index_vec[-1]) )        

            res = self.refinement_step(self)

            if not res:
                if verbose:
                    # Irregular by definition
                    print("{0}, {1}, {2}, {3}, irregular".format(iteration, self.k, self.classes_cardinality, self.index_vec[-1]))
                return (False, self.k, None)

        #self.generate_reduced_sim_mat()
        #return (True, self.k, self.reduced_sim_mat)
        return (True, self.k, self.classes)
