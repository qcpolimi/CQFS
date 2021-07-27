import numpy as np
from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython

from core.CQFS import CQFS
from core.CQFSTrainer import CQFSTrainer
from data.DataLoader import DataLoader
from recsys.Base.DataIO import DataIO
from recsys.Recommender_import_list import ItemKNNCFRecommender
from utils.sparse import merge_sparse_matrices
from utils.statistics import warm_similarity_statistics


def train_CQFS(data_loader: DataLoader, ICM_name, percentages, alphas, betas, combination_strengths, solver_class,
               CF_recommender_classes, parameter_product=True, cpu_count_div=2, cpu_count_sub=0):
    N_CASES = 50
    N_RAN_STARTS = 15
    SIMILARITY_TYPE = 'cosine'

    CUTOFF_VALIDATION = [10]
    CUTOFF_TEST = [5, 10, 20, 50]

    ##################################################
    # Data loading and splitting

    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the cold split
    URM_train, URM_validation, URM_test = data_loader.get_cold_split()

    # Create the last test URM by merging the train and validation matrices
    URM_train_validation = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    ICM_train, original_ICM_train = data_loader.get_ICM_train_from_name(ICM_name, return_original=True)
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    ##################################################
    # Evaluators instantiation

    # Obtain the array of train item indices, to be ignored during validation
    train_warm_item_mask = np.ediff1d(URM_train.tocsc().indptr) != 0
    train_warm_item_mask = np.arange(len(train_warm_item_mask))[train_warm_item_mask]

    # Obtain the array of train and validation item indices, to be ignored during testing
    train_validation_warm_item_mask = np.ediff1d(URM_train_validation.tocsc().indptr) != 0
    train_validation_warm_item_mask = np.arange(len(train_validation_warm_item_mask))[train_validation_warm_item_mask]

    ##################################################
    # Quantum Feature Selection

    # The CBF similarity used for the selection is a simple dot product
    topK = n_items
    CBF_Similarity = Compute_Similarity_Cython(ICM_train.T, topK=topK, shrink=0, normalize=False, similarity='cosine')
    # S_CBF_original = CBF_Similarity.compute_similarity()
    S_CBF = CBF_Similarity.compute_similarity()

    for CF_recommender_class in CF_recommender_classes:
        ##################################################
        # Setup collaborative filtering recommender

        # Get Collaborative Filtering best hyperparameters
        cf_recommender_name = CF_recommender_class.RECOMMENDER_NAME
        cf_path = f"../../results/{dataset_name}/{cf_recommender_name}/"
        cf_dataIO = DataIO(cf_path)
        cf_similarity = "cosine_" if CF_recommender_class is ItemKNNCFRecommender else ""
        cf_dict = cf_dataIO.load_data(f"{cf_recommender_name}_{cf_similarity}metadata.zip")
        cf_best_hyperparameters = cf_dict['hyperparameters_best']

        # Create Collaborative Filtering Recommender and fit with the best hyperparameters
        CF_recommender = CF_recommender_class(URM_train_validation)
        CF_recommender.fit(**cf_best_hyperparameters)

        # Get CF and CBF Similarity Matrices
        S_CF = CF_recommender.W_sparse.copy()
        # S_CBF = S_CBF_original.copy()

        assert S_CF.shape == S_CBF.shape, "The two sparse matrices have different shapes!"
        assert S_CF.shape == (n_items, n_items), "The similarity matrices do not have the right shape."

        ##################################################
        # Setup CQFS

        # Get the warm items (CF) and the items with features (CBF)
        CF_warm_items = np.ediff1d(URM_train_validation.tocsc().indptr) != 0
        CBF_items_with_interactions = np.ediff1d(ICM_train.indptr) != 0

        # Compute warm items statistics
        statistics = warm_similarity_statistics(S_CF, S_CBF, CF_warm_items=CF_warm_items,
                                                CBF_items_with_interactions=CBF_items_with_interactions)

        base_folder_path = f"../../results/{dataset_name}/{ICM_name}/{cf_recommender_name}/"
        CQFS_selector = CQFS(ICM_train, S_CF, S_CBF, base_folder_path, solver_class=solver_class,
                             statistics=statistics)

        ##################################################
        # Train CQFS ItemKNNCBFRecommender

        CQFS_trainer = CQFSTrainer(CQFS_selector, URM_train, URM_validation, URM_train_validation, URM_test, ICM_train,
                                   original_ICM_train, ICM_name, dataset_name, cf_recommender_name,
                                   cutoff_list_validation=CUTOFF_VALIDATION, cutoff_list_test=CUTOFF_TEST,
                                   ignore_items_validation=train_warm_item_mask,
                                   ignore_items_test=train_validation_warm_item_mask, n_cases=N_CASES,
                                   n_random_starts=N_RAN_STARTS, similarity_type_list=[SIMILARITY_TYPE],
                                   parallelize=True)

        CQFS_trainer.train_many(percentages, alphas, betas, combination_strengths, parameter_product=parameter_product,
                                cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub)
        CQFS_trainer.compute_statistics(S_CF, S_CBF, percentages, alphas, betas, combination_strengths)
