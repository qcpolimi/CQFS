import re
import time

import dimod
import numpy as np
from dwave.system import DWaveSampler, LeapHybridSampler, EmbeddingComposite
from neal import SimulatedAnnealingSampler

from core.CQFSSampler import CQFSSampler, CQFSEmbedding, CQFSSimulatedAnnealingSampler, \
    CQFSQBSolvSampler, CQFSQBSolvTabuSampler
from recsys.Base.DataIO import DataIO
from utils.naming import get_experiment_id
from utils.samplers import get_hybrid_from_topology
from utils.statistics import similarity_statistics, BQM_statistics


class CQFS:
    SAVED_MODELS_FILE = 'saved_CQFS_models.zip'
    STATISTICS_FILE = 'statistics'
    TIMINGS_FILE = 'timings'

    ACCEPTED_SOLVERS = [DWaveSampler, LeapHybridSampler, SimulatedAnnealingSampler, CQFSSimulatedAnnealingSampler,
                        CQFSQBSolvSampler]
    ACCEPTED_TOPOLOGIES = ['pegasus', 'chimera']

    N_EMBEDDING_CASES = 5
    EMBEDDING_LIMITS = {
        'pegasus': 180,
        'chimera': 65,
    }

    def __init__(self, ICM_train, S_CF, S_CBF, base_folder_path, solver_class=DWaveSampler, qpu_topology='pegasus',
                 statistics=None):

        self.n_items, self.n_features = ICM_train.shape
        self.ICM_train = ICM_train.copy()

        if re.match('.*/.*ICM.*/.*Recommender.*/', base_folder_path) is None:
            self.__print("[WARNING] base_folder_path has a custom format, we suggest to use the following one for "
                         "compatibility with other classes:\n"
                         "DatasetName/ICMName/CFRecommenderName/")

        self.base_folder_path = base_folder_path if base_folder_path[-1] == '/' else f"{base_folder_path}/"
        self.dataIO = DataIO(self.base_folder_path)

        self.statistics = {}
        if statistics is not None:
            self.statistics = statistics
        self.timings = self.__load_timings()

        ##################################################
        # Model variables

        # self.IPMs = {}
        # self.FPMs = {}
        self.BQMs = {}
        self.selections = {}

        ##################################################
        # Load previously saved models

        self.saved_models = self.__load_previously_saved_models()

        if not self.saved_models['K'] or not self.saved_models['E']:
            ##################################################
            # Model initialization

            self.__print("Building the base models...")
            base_model_time = time.time()

            # self.S_CF = S_CF.copy()
            # self.S_CBF = S_CBF.copy()
            assert S_CF.shape == S_CBF.shape, "The two sparse matrices have different shapes."
            assert S_CF.shape == (self.n_items, self.n_items), "The similarity matrices do not have the right shape."

            S_CF.data = np.ones_like(S_CF.data)
            S_CBF.data = np.ones_like(S_CBF.data)

            S_CF_bool = S_CF.astype(np.bool)
            S_CBF_bool = S_CBF.astype(np.bool, copy=True)

            #########################
            # Compute the bonus for similarities in common

            K_time = time.time()
            self.K = S_CBF.multiply(S_CF_bool)
            self.K.data = -self.K.data
            K_time = time.time() - K_time

            #########################
            # Compute the penalization for similarities not in common

            E_time = time.time()
            S_intersection = S_CBF_bool.multiply(S_CF_bool)
            self.E = S_CBF_bool - S_intersection
            self.E = S_CBF.multiply(self.E)
            E_time = time.time() - E_time

            assert self.K.nnz + self.E.nnz == S_CBF.nnz, "The number of items to keep and to penalize is not correct."

            S_union = S_CF_bool + S_CBF_bool

            self.statistics = similarity_statistics(S_CF, S_CBF, S_intersection, S_union, statistics=self.statistics)
            # self.statistics = error_statistics(S_CF, S_CBF, N=S_union.nnz, suffix="_dot", statistics=self.statistics)
            self.__save_statistics()

            base_model_time = time.time() - base_model_time
            self.timings['base_model_time'] = base_model_time
            self.timings['K_time'] = K_time
            self.timings['E_time'] = E_time

            self.timings['avg_IPM_time'] = 0
            self.timings['avg_FPM_time'] = 0
            self.timings['avg_BQM_time'] = 0
            self.timings['avg_QUBO_time'] = 0
            self.timings['n_fit_experiments'] = 0
            self.timings['avg_response_time'] = {}
            self.timings['n_select_experiments'] = {}
            self.__save_timings()

            self.saved_models['K'] = True
            self.saved_models['E'] = True

            self.dataIO.save_data(self.SAVED_MODELS_FILE, self.saved_models)
            self.dataIO.save_data('K', {'K': self.K})
            self.dataIO.save_data('E', {'E': self.E})

            self.__print("Base models successfully built.")
        else:
            self.__print("Base models successfully loaded.")

        ##################################################
        # Solver initialization

        self.qpu_only = False

        assert qpu_topology in self.ACCEPTED_TOPOLOGIES, f"QPU topology should be one of {self.ACCEPTED_TOPOLOGIES}."
        self.qpu_topology = qpu_topology

        if solver_class is DWaveSampler:
            assert self.n_features <= self.EMBEDDING_LIMITS[qpu_topology], \
                f"It is not possible to embed {self.n_features} variables onto the {qpu_topology} topology.\n" \
                f"The maximum number of variables embeddable is {self.EMBEDDING_LIMITS[qpu_topology]}."

            self.sampler = DWaveSampler(solver={'topology__type': qpu_topology})
            self.embedder = CQFSEmbedding(target_sampler=self.sampler, qpu_topology=qpu_topology,
                                          n_embedding_cases=self.N_EMBEDDING_CASES,
                                          embedding_folder_path=base_folder_path)
            self.solver = EmbeddingComposite(self.sampler)
            self.selection_type = f"qpu_{qpu_topology}"
            self.qpu_only = True

        elif solver_class is LeapHybridSampler:
            self.solver = get_hybrid_from_topology(topology=qpu_topology)
            self.selection_type = f"hybrid_{qpu_topology}"

        elif solver_class is SimulatedAnnealingSampler:
            self.solver = solver_class()
            self.selection_type = "simulated_annealing_dwave"

        elif solver_class is CQFSSimulatedAnnealingSampler:
            self.solver = solver_class()
            self.selection_type = "cqfs_simulated_annealing_dwave"

        elif solver_class is CQFSQBSolvSampler:
            self.solver = solver_class()
            self.selection_type = "cqfs_qbsolv_dwave"

        elif solver_class is CQFSQBSolvTabuSampler:
            self.solver = solver_class()
            self.selection_type = "cqfs_qbsolv_tabu"

        else:
            assert solver_class in self.ACCEPTED_SOLVERS, f"Solver class should be one of {self.ACCEPTED_SOLVERS}."
            self.solver = solver_class()
            self.selection_type = "classical"

        if self.timings['avg_response_time'].get(self.selection_type) is None:
            self.timings['avg_response_time'][self.selection_type] = 0
            self.timings['n_select_experiments'][self.selection_type] = 0

    @staticmethod
    def __print(msg):
        print(f"CQFS: {msg}")

    def __save_statistics(self):
        self.dataIO.save_data(self.STATISTICS_FILE, self.statistics)

    def __save_timings(self):
        self.dataIO.save_data(self.TIMINGS_FILE, self.timings)

    def __load_timings(self):
        timings = {}
        try:
            timings = self.dataIO.load_data(self.TIMINGS_FILE)
        except FileNotFoundError:
            self.__print("No timings file found.")
        return timings

    def __load_base_model(self, model):

        model_file = f'{model}.zip'

        try:
            if model == 'K':
                self.K = self.dataIO.load_data(model_file)['K']
            elif model == 'E':
                self.E = self.dataIO.load_data(model_file)['E']
            return True

        except FileNotFoundError:
            return False

    def __load_previously_saved_models(self):

        self.__print("Trying to load previously saved models.")

        saved_models = {
            'K': False,
            'E': False,
        }

        try:
            saved_models = self.dataIO.load_data(self.SAVED_MODELS_FILE)

            for model in saved_models:
                if saved_models[model]:
                    saved_models[model] = self.__load_base_model(model)

        except FileNotFoundError:
            self.__print("No model saved for this set of experiments.")

        self.dataIO.save_data(self.SAVED_MODELS_FILE, saved_models)
        return saved_models

    @staticmethod
    def __p_to_k(p, n_features):
        assert p is not None, "Please, choose a selection percentage." \
                              "The value should be between 0 and 1 or between 0 and 100."

        if 1 < p <= 100:
            p /= 100
        elif p > 100 or p < 0:
            raise ValueError("Percentage value should be between 0 and 1 or between 0 and 100.")

        return n_features * p

    def fit(self, alpha=1, beta=1, vartype='BINARY', save_FPM=False):

        fitID = get_experiment_id(alpha, beta)

        self.__print(f"[{fitID}] Fitting experiment.")

        # if self.FPMs.get(fitID) is None:
        if self.BQMs.get(fitID) is None:
            # IPM = self.IPMs.get(fitID)
            # if IPM is None:
            #     IPM = alpha * self.K + beta * self.E

            QUBO_time = time.time()
            IPM_time = time.time()
            IPM = alpha * self.K + beta * self.E
            IPM.eliminate_zeros()
            IPM_time = time.time() - IPM_time

            FPM_time = time.time()
            IFPM = IPM * self.ICM_train
            IFPM.eliminate_zeros()
            FPM = self.ICM_train.T * IFPM
            FPM.eliminate_zeros()
            FPM_time = time.time() - FPM_time

            BQM_time = time.time()
            BQM = dimod.as_bqm(FPM.toarray(), vartype)
            BQM_time = time.time() - BQM_time
            QUBO_time = time.time() - QUBO_time

            experiment_timings = {
                'IPM_time': IPM_time,
                'FPM_time': FPM_time,
                'BQM_time': BQM_time,
                'QUBO_time': QUBO_time,
            }
            experiment_dataIO = self.__get_experiment_dataIO(fitID)
            experiment_dataIO.save_data(self.TIMINGS_FILE, experiment_timings)

            n_experiments = self.timings['n_fit_experiments']
            total_IPM_time = self.timings['avg_IPM_time'] * n_experiments
            total_FPM_time = self.timings['avg_FPM_time'] * n_experiments
            total_BQM_time = self.timings['avg_BQM_time'] * n_experiments
            total_QUBO_time = self.timings['avg_QUBO_time'] * n_experiments

            n_experiments += 1
            self.timings['n_fit_experiments'] = n_experiments
            self.timings['avg_IPM_time'] = (total_IPM_time + IPM_time) / n_experiments
            self.timings['avg_FPM_time'] = (total_FPM_time + FPM_time) / n_experiments
            self.timings['avg_BQM_time'] = (total_BQM_time + BQM_time) / n_experiments
            self.timings['avg_QUBO_time'] = (total_QUBO_time + QUBO_time) / n_experiments

            self.__save_timings()

            self.__print(f"[{fitID}] Fitted in {QUBO_time} sec.")

            FPM_statistics_file = 'FPM_statistics'
            try:
                experiment_dataIO.load_data(FPM_statistics_file)
                self.__print(f"[{fitID}] Found a previously saved {FPM_statistics_file} file."
                             f" Skipping statistics computation.")
            except FileNotFoundError:

                linear = FPM.diagonal()

                quadratic = FPM.tocsr()
                quadratic.setdiag(0)
                quadratic.eliminate_zeros()
                quadratic_data = quadratic.data

                FPM_statistics = BQM_statistics(linear, quadratic_data, FPM.shape, prefix='FPM_')
                self.__print(f"[{fitID}] Computed {FPM_statistics_file}. Saving to file.")
                experiment_dataIO.save_data(FPM_statistics_file, FPM_statistics)

            if save_FPM:
                FPM_file = 'FPM'
                try:
                    experiment_dataIO.load_data(FPM_file)
                    self.__print(f"[{fitID}] Found a previously saved {FPM_file} file."
                                 f" Skipping FPM saving.")
                except FileNotFoundError:
                    self.__print(f"[{fitID}] Saving FPM to file...")
                    experiment_dataIO.save_data(FPM_file, {'FPM': FPM})

            # self.IPMs[fitID] = IPM.copy()
            # self.FPMs[fitID] = FPM.copy()
            self.BQMs[fitID] = BQM.copy()

        # else:
        #     FPM = self.FPMs[fitID]
        #     BQM = dimod.as_bqm(FPM.toarray(), vartype)
        #
        #     self.BQMs[fitID] = BQM.copy()

    def fit_many(self, alphas, betas, vartype='BINARY', save_FPM=False):

        for alpha in alphas:
            for beta in betas:
                self.fit(alpha, beta, vartype, save_FPM)

    def __get_selection_from_sample(self, sample):

        selection = np.zeros(self.n_features, dtype=bool)
        for k, v in sample.items():
            if v == 1:
                ind = int(k)
                selection[ind] = True

        return selection

    def __get_experiment_dataIO(self, expID):
        experiment_folder_path = f"{self.base_folder_path}{expID}/"
        return DataIO(experiment_folder_path)

    def __get_selection_dataIO(self, expID):
        selection_folder_path = f"{self.base_folder_path}{expID}/{self.selection_type}/"
        return DataIO(selection_folder_path)

    def __save_selection(self, expID, selection, response):

        selection_dataIO = self.__get_selection_dataIO(expID)
        selection_dict = {
            'selection': selection,
            'response': response.to_serializable(),
        }
        selection_dataIO.save_data(self.selection_type, selection_dict)
        self.__print(f"[{expID}] Selection and response saved.")

    def __load_selection(self, expID):

        selection_dataIO = self.__get_selection_dataIO(expID)

        try:
            selection = selection_dataIO.load_data(self.selection_type)['selection']
            self.__print(f"[{expID}] Found an existing selection.")
            self.selections[expID] = selection.copy()

        except FileNotFoundError:
            self.__print(f"[{expID}] No previous selection found.")

    def __compute_and_save_BQM_statistics(self, BQM, prefix, experiment_dataIO):
        file_name = f"{prefix}statistics"

        try:
            experiment_dataIO.load_data(file_name)
            self.__print(f"Found a previously saved {file_name} file. Skipping statistics computation.")
        except FileNotFoundError:
            BQM_linear, (_, _, BQM_quadratic_data), _ = BQM.to_numpy_vectors()
            shape = (BQM.num_variables, BQM.num_variables)
            statistics = BQM_statistics(BQM_linear, BQM_quadratic_data, shape, prefix=prefix)

            self.__print(f"Computed {file_name}. Saving to file.")
            experiment_dataIO.save_data(file_name, statistics)

    def select(self, alpha, beta, combination_strength=1):
        raise NotImplementedError("Method not implemented yet.")

    def select_p(self, p, alpha, beta, combination_strength=1, vartype='BINARY', normalize=True, save_FPM=False,
                 save_BQM=False):
        k = self.__p_to_k(p, self.n_features)

        expID = get_experiment_id(alpha, beta, p=p, combination_strength=combination_strength)
        self.__load_selection(expID)
        if self.selections.get(expID) is not None:
            return self.selections[expID].copy()

        self.__print(f"[{expID}] Starting selection.")

        BQM = self.BQMs.get(expID)
        if BQM is None:
            fitID = get_experiment_id(alpha, beta)
            BQM = self.BQMs.get(fitID)
            if BQM is None:
                self.fit(alpha=alpha, beta=beta, vartype=vartype, save_FPM=save_FPM)
                BQM = self.BQMs.get(fitID)

        BQM_time = time.time()
        BQM_k = dimod.generators.combinations(self.n_features, k, strength=combination_strength, vartype=vartype)
        BQM_k = dimod.AdjVectorBQM(BQM_k)
        BQM_k.update(BQM)
        BQM_time = time.time() - BQM_time

        self.BQMs[expID] = BQM_k.copy()

        experiment_dataIO = self.__get_experiment_dataIO(expID)
        self.__compute_and_save_BQM_statistics(BQM_k, 'BQM_', experiment_dataIO)

        BQM_normalization_time = 0
        if normalize:
            BQM_normalization_time = time.time()
            BQM_k.normalize()
            BQM_normalization_time = time.time() - BQM_normalization_time
            self.__compute_and_save_BQM_statistics(BQM_k, 'BQM_norm_', experiment_dataIO)

        if save_BQM:
            self.__print(f"[{expID}] Saving BQM to file...")
            BQM_dict = {'BQM': BQM_k.to_numpy_matrix()}
            experiment_dataIO.save_data('BQM', BQM_dict)

        self.__print(f"[{expID}] Starting problem sampling.")

        if self.qpu_only:
            self.__print(f"[{expID}] Computing problem embedding.")
            embedding = self.embedder.get_embedding(BQM_k, expID, resume_from_saved=True)
            self.solver = CQFSSampler(embedding, target_sampler=self.sampler, qpu_topology=self.qpu_topology)

        self.__print(f"[{expID}] Sampling the problem.")
        response_time = time.time()
        response = self.solver.sample(BQM_k)
        response_time = time.time() - response_time

        experiment_timings = {
            'BQM_time': BQM_time,
            'BQM_normalization_time': BQM_normalization_time,
            'response_time': response_time,
        }
        selection_dataIO = self.__get_selection_dataIO(expID)
        selection_dataIO.save_data(self.TIMINGS_FILE, experiment_timings)

        n_experiments = self.timings['n_select_experiments'][self.selection_type]
        total_response_time = self.timings['avg_response_time'][self.selection_type] * n_experiments
        n_experiments += 1
        self.timings['n_select_experiments'][self.selection_type] = n_experiments
        self.timings['avg_response_time'][self.selection_type] = (total_response_time + response_time) / n_experiments
        self.__save_timings()

        best_sample = response.first.sample
        selection = self.__get_selection_from_sample(best_sample)

        self.__print(f"[{expID}] Selected {selection.sum()} features in {response_time} sec.")

        self.selections[expID] = selection.copy()
        self.__save_selection(expID, selection, response)
        return selection

    def select_many_p(self, ps, alphas, betas, combination_strengths, vartype='BINARY', normalize=True, save_FPMs=False,
                      save_BQMs=False, parameter_product=True):

        if parameter_product:
            for p in ps:
                for alpha in alphas:
                    for beta in betas:
                        for combination_strength in combination_strengths:
                            self.select_p(p, alpha, beta, combination_strength, vartype, normalize, save_FPMs,
                                          save_BQMs)
                        # self.BQMs.clear()
        else:
            args_zip = zip(ps, alphas, betas, combination_strengths)
            for args in args_zip:
                self.select_p(args[0], args[1], args[2], args[3], vartype, normalize, save_FPMs, save_BQMs)
