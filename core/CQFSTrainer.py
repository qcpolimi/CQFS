import itertools

import numpy as np

from recsys.Base.DataIO import DataIO
from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout
from recsys.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from recsys.ParameterTuning.run_parameter_search import runParameterSearch_Content
from utils.multithreading import parallelize_function
from utils.naming import get_experiment_id
from utils.sparse import select_columns
from utils.statistics import results_similarity_statistics, error_statistics


class CQFSTrainer:
    N_CASES = 50
    N_RAN_STARTS = 15

    STATISTICS_FILE = "S_CQFS_statistics"

    def __init__(self, CQFS, URM_train, URM_validation, URM_train_last_test, URM_test, ICM_train, ICM_train_last_test,
                 ICM_name, dataset_name, cf_recommender_name, cutoff_list_validation=None, cutoff_list_test=None,
                 ignore_items_validation=None, ignore_items_test=None, n_cases=N_CASES, n_random_starts=N_RAN_STARTS,
                 similarity_type_list=None, parallelize=False):
        self.CQFS = CQFS
        self.selection_type = CQFS.selection_type

        self.URM_train = URM_train
        self.URM_validation = URM_validation
        self.URM_train_last_test = URM_train_last_test
        self.URM_test = URM_test

        self.ICM_train = ICM_train
        self.ICM_train_last_test = ICM_train_last_test
        self.ICM_name = ICM_name

        self.dataset_name = dataset_name
        self.cf_recommender_name = cf_recommender_name

        self.cutoff_list_validation = cutoff_list_validation if cutoff_list_validation is not None else [10]
        self.cutoff_list_test = cutoff_list_test if cutoff_list_test is not None else [10]
        self.ignore_items_validation = ignore_items_validation
        self.ignore_items_test = ignore_items_test

        self.n_cases = n_cases
        self.n_random_starts = n_random_starts
        self.similarity_type_list = ['cosine'] if similarity_type_list is None else similarity_type_list

        self.parallelize = parallelize

        if not self.parallelize:
            self.evaluator_validation = self.__get_evaluator_validation()
            self.evaluator_test = self.__get_evaluator_test()

    @staticmethod
    def __print(msg):
        print(f"CQFSTrainer: {msg}")

    def __get_evaluator_validation(self):
        return EvaluatorHoldout(self.URM_validation, cutoff_list=self.cutoff_list_validation,
                                ignore_items=self.ignore_items_validation)

    def __get_evaluator_test(self):
        return EvaluatorHoldout(self.URM_test, cutoff_list=self.cutoff_list_test, ignore_items=self.ignore_items_test)

    def __get_selection(self, expID):

        experiment_folder_path = f"{self.CQFS.base_folder_path}{expID}/{self.CQFS.selection_type}/"
        selection_dataIO = DataIO(experiment_folder_path)

        try:
            selection = selection_dataIO.load_data(self.CQFS.selection_type)['selection']
            self.__print(f"[{expID}] Found an existing selection.")
            return selection

        except FileNotFoundError:
            self.__print(f"[{expID}] No previous selection found.")
            return None

    def train(self, expID):

        selection = self.__get_selection(expID)
        if selection is None:
            _print_train_failed(expID)
            return

        train(expID, self.selection_type, self.evaluator_validation, self.evaluator_test, self.dataset_name,
              self.ICM_name, self.cf_recommender_name, selection, self.URM_train, self.URM_train_last_test,
              self.ICM_train, self.ICM_train_last_test, self.n_cases, self.n_random_starts, self.similarity_type_list)

    def __get_experiment_ids(self, ps, alphas, betas, combination_strengths, parameter_product=True):

        if parameter_product:
            expIDs = [get_experiment_id(x[0], x[1], x[2], x[3]) for x in
                      itertools.product(alphas, betas, ps, combination_strengths)]
        else:
            expIDs = [get_experiment_id(x[0], x[1], x[2], x[3]) for x in zip(alphas, betas, ps, combination_strengths)]

        return expIDs

    def train_many(self, ps, alphas, betas, combination_strengths, parameter_product=True, cpu_count_div=2,
                   cpu_count_sub=0):

        expIDs = self.__get_experiment_ids(ps, alphas, betas, combination_strengths,
                                           parameter_product=parameter_product)

        if self.parallelize:
            args = [(
                expID,
                self.selection_type,
                self.__get_evaluator_validation(),
                self.__get_evaluator_test(),
                self.dataset_name,
                self.ICM_name,
                self.cf_recommender_name,
                self.__get_selection(expID),
                self.URM_train,
                self.URM_train_last_test,
                self.ICM_train,
                self.ICM_train_last_test,
                self.n_cases,
                self.n_random_starts,
                self.similarity_type_list
            ) for expID in expIDs]
            parallelize_function(train, args, count_div=cpu_count_div, count_sub=cpu_count_sub)
        else:
            for expID in expIDs:
                self.train(expID)

    def compute_statistics(self, S_CF, S_CBF, ps, alphas, betas, combination_strengths, parameter_product=True):

        assert S_CF.shape == S_CBF.shape, "The two sparse matrices have different shapes."

        expIDs = self.__get_experiment_ids(ps, alphas, betas, combination_strengths,
                                           parameter_product=parameter_product)

        S_CF.data = np.ones_like(S_CF.data)
        S_CBF.data = np.ones_like(S_CBF.data)

        S_CF_bool = S_CF.astype(np.bool)
        S_CBF_bool = S_CBF.astype(np.bool, copy=True)
        S_intersection = S_CBF_bool.multiply(S_CF_bool)

        for expID in expIDs:

            selection = self.__get_selection(expID)
            if selection is None:
                continue

            new_ICM_train = select_columns(self.ICM_train, selection)

            output_folder_path = f"../../results/{self.dataset_name}/{self.ICM_name}/{self.cf_recommender_name}/{expID}/" \
                                 f"{self.selection_type}/{ItemKNNCBFRecommender.RECOMMENDER_NAME}/"
            dataIO = DataIO(output_folder_path)

            for similarity_type in self.similarity_type_list:
                try:
                    CQFS_S_CBF = dataIO.load_data(
                        f"{ItemKNNCBFRecommender.RECOMMENDER_NAME}_{self.ICM_name}_{similarity_type}_best_model_last.zip")
                    CQFS_S_CBF = CQFS_S_CBF['W_sparse']
                    CQFS_S_CBF.data = np.ones_like(CQFS_S_CBF.data)

                    CQFS_S_CBF_bool = CQFS_S_CBF.astype(bool, copy=True)
                    CQFS_S_intersection = CQFS_S_CBF_bool.multiply(S_CF_bool)

                    CQFS_S_union = S_CF_bool + CQFS_S_CBF_bool

                    CQFS_CBF_warm_items = np.ediff1d(new_ICM_train.indptr) != 0

                    statistics = results_similarity_statistics(
                        CQFS_S_CBF, S_CF, S_CBF, intersection=S_intersection, CQFS_intersection=CQFS_S_intersection,
                        CQFS_union=CQFS_S_union, items_with_features=CQFS_CBF_warm_items)
                    statistics = error_statistics(S_CF, CQFS_S_CBF, statistics=statistics)

                    dataIO.save_data(self.STATISTICS_FILE, statistics)

                except FileNotFoundError:
                    print(f"{output_folder_path}: it was impossible to load the best model obtained from the tuning.")


def __print(msg):
    print(f"CQFSTrainer: {msg}")


def _print_train_failed(expID):
    __print(f"{expID} It is not possible to train without the proper selection.\n"
            f"{expID} Please run the selection procedure through CQFS first.")


def train(expID, selection_type, evaluator_validation, evaluator_test, dataset_name, ICM_name, CF_recommender_name,
          selection, URM_train, URM_train_last_test, ICM_train, ICM_train_last_test, n_cases, n_random_starts,
          similarity_type_list):
    if selection is None:
        _print_train_failed(expID)
        return

    __print(f"[{expID}] Tuning parameters.")

    output_folder_path = f"../../results/{dataset_name}/{ICM_name}/{CF_recommender_name}/{expID}/{selection_type}/" \
                         f"{ItemKNNCBFRecommender.RECOMMENDER_NAME}/"

    new_ICM_train = select_columns(ICM_train, selection)
    new_ICM_train_last_test = select_columns(ICM_train_last_test, selection)

    runParameterSearch_Content(ItemKNNCBFRecommender, URM_train, new_ICM_train,
                               ICM_name, URM_train_last_test=URM_train_last_test,
                               ICM_last_test=new_ICM_train_last_test,
                               n_cases=n_cases, n_random_starts=n_random_starts,
                               resume_from_saved=True, save_model='best',
                               evaluator_validation=evaluator_validation,
                               evaluator_test=evaluator_test,
                               output_folder_path=output_folder_path,
                               similarity_type_list=similarity_type_list)

    __print(f"[{expID}] Parameter tuning ended.")
