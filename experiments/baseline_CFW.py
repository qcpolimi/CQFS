import numpy as np

from data.DataLoader import DataLoader
from recsys.Base.DataIO import DataIO
from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout
from recsys.FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython, EvaluatorCFW_D_wrapper
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.ParameterTuning.run_parameter_search import runParameterSearch_FeatureWeighting
from utils.multithreading import parallelize_function
from utils.recsys import remove_ICM_item_interactions
from utils.sparse import merge_sparse_matrices


def train_CFW(CFW_recommender, ICM_name, output_folder_path, W_train, evaluator_validation, evaluator_test,
              evaluator_validation_earlystopping, URM_train, URM_train_last_test, ICM_train, ICM_train_last_test,
              n_cases, n_random_starts):
    runParameterSearch_FeatureWeighting(CFW_recommender, URM_train, W_train, ICM_train, ICM_name,
                                        URM_train_last_test=URM_train_last_test, ICM_last_test=ICM_train_last_test,
                                        n_cases=n_cases, n_random_starts=n_random_starts, resume_from_saved=True,
                                        evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
                                        evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                        output_folder_path=output_folder_path)


def baseline_CFW(data_loader: DataLoader, ICM_name, CF_recommenders, n_cases=50, n_random_starts=15, parallelize=True):
    ##################################################
    # Data loading and splitting

    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the split used for CBF
    URM_train, URM_validation, URM_test = data_loader.get_cold_split()

    # Get the warm items URM
    URM_train_validation = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    # Get the filtered ICM
    original_ICM_train = data_loader.get_original_ICM_train_from_name(ICM_name)

    # Remove cold test items interactions from the ICM
    test_items = np.ediff1d(URM_test.tocsc().indptr) != 0
    test_items = np.arange(len(test_items))[test_items]
    ICM_validation = remove_ICM_item_interactions(original_ICM_train, test_items)

    # Check if the removal was correct
    no_ICM_interaction_item_mask = np.ediff1d(ICM_validation.indptr) == 0
    assert np.alltrue(
        no_ICM_interaction_item_mask[test_items]), "Test items were not correctly removed from the train ICM."

    # Remove cold validation items interactions from the ICM
    validation_items = np.ediff1d(URM_validation.tocsc().indptr) != 0
    validation_items = np.arange(len(validation_items))[validation_items]
    ICM_train = remove_ICM_item_interactions(ICM_validation, validation_items)

    # Check if the removal was correct
    no_ICM_interaction_item_mask = np.ediff1d(ICM_train.indptr) == 0
    assert np.alltrue(
        no_ICM_interaction_item_mask[validation_items]), "Test items were not correctly removed from the train ICM."

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

    # Create the evaluator objects for validation and test
    # Train items are ignored during validation; train and validation items are ignored during testing
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], ignore_items=train_warm_item_mask)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20, 50],
                                      ignore_items=train_validation_warm_item_mask)

    evaluator_validation_wrapper = EvaluatorCFW_D_wrapper(evaluator_validation, ICM_validation)
    evaluator_validation_earlystopping = EvaluatorCFW_D_wrapper(evaluator_validation, ICM_validation,
                                                                model_to_use='last')
    evaluator_test_wrapper = EvaluatorCFW_D_wrapper(evaluator_test, original_ICM_train)

    ##################################################
    # CFW parameter tuning

    CFW_recommenders = [CFW_D_Similarity_Cython]  # , CFW_DVV_Similarity_Cython, FBSM_Rating_Cython]

    W_trains = {}
    for Recommender in CF_recommenders:
        # Get Collaborative Filtering best hyperparameters
        cf_recommender_name = Recommender.RECOMMENDER_NAME
        cf_path = f"../../results/{dataset_name}/{cf_recommender_name}/"
        cf_dataIO = DataIO(cf_path)
        cf_similarity = "cosine_" if Recommender is ItemKNNCFRecommender else ""
        cf_dict = cf_dataIO.load_data(f"{cf_recommender_name}_{cf_similarity}metadata.zip")
        cf_best_hyperparameters = cf_dict['hyperparameters_best']

        # Create Collaborative Filtering Recommender and fit with the best hyperparameters
        CF_recommender = Recommender(URM_train_validation)
        CF_recommender.fit(**cf_best_hyperparameters)

        # Get CF and CBF Similarity Matrices
        W_trains[cf_recommender_name] = CF_recommender.W_sparse.copy()

    for CFW_recommender in CFW_recommenders:
        CFW_name = CFW_recommender.RECOMMENDER_NAME

        if parallelize:
            cf_recommender_names = [Recommender.RECOMMENDER_NAME for Recommender in CF_recommenders]
            W_train_models = {cf_recommender_name: W_trains[cf_recommender_name]
                              for cf_recommender_name in cf_recommender_names}

            output_folder_paths = {cf_recommender_name:
                                       f"../../results/{dataset_name}/{ICM_name}/{CFW_name}/{cf_recommender_name}/"
                                   for cf_recommender_name in cf_recommender_names}

            args = [(CFW_recommender, ICM_name, output_folder_paths[cf_recommender_name],
                     W_train_models[cf_recommender_name], evaluator_validation_wrapper, evaluator_test_wrapper,
                     evaluator_validation_earlystopping, URM_train, URM_train_validation, ICM_train, original_ICM_train,
                     n_cases, n_random_starts) for cf_recommender_name in cf_recommender_names]
            parallelize_function(train_CFW, args, count_div=1, count_sub=0)

        else:
            for Recommender in CF_recommenders:
                cf_recommender_name = Recommender.RECOMMENDER_NAME
                W_train = W_trains[cf_recommender_name]

                output_folder_path = f"../../results/{dataset_name}/{ICM_name}/{CFW_name}/{cf_recommender_name}/"

                train_CFW(CFW_recommender, ICM_name, output_folder_path, W_train, evaluator_validation_wrapper,
                          evaluator_test_wrapper, evaluator_validation_earlystopping, URM_train, URM_train_validation,
                          ICM_train, original_ICM_train, n_cases, n_random_starts)
