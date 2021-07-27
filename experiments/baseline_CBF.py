import numpy as np

from data.DataLoader import DataLoader
from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout
from recsys.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from recsys.ParameterTuning.run_parameter_search import runParameterSearch_Content
from utils.sparse import merge_sparse_matrices


def baseline_CBF(data_loader: DataLoader, ICM_name, n_cases=50, n_random_starts=15, similarity_type_list=['cosine']):
    ##################################################
    # Data loading and splitting

    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the split
    URM_train, URM_validation, URM_test, URM_train_warm, URM_validation_warm = data_loader.get_both_splits()

    # Create the last test URM by merging the train and validation matrices
    URM_train_last_test = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    # Get the original ICM
    ICM_train, original_ICM_train = data_loader.get_ICM_train_from_name(ICM_name, return_original=True)
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    ##################################################
    # Evaluators instantiation

    # Obtain the array of train item indices, to be ignored during validation
    train_warm_item_mask = np.ediff1d(URM_train.tocsc().indptr) != 0
    train_warm_item_mask = np.arange(len(train_warm_item_mask))[train_warm_item_mask]

    # Obtain the array of train and validation item indices, to be ignored during testing
    train_validation_warm_item_mask = np.ediff1d(URM_train_last_test.tocsc().indptr) != 0
    train_validation_warm_item_mask = np.arange(len(train_validation_warm_item_mask))[train_validation_warm_item_mask]

    # Create the evaluator objects for validation and test
    # Train items are ignored during validation; train and validation items are ignored during testing
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], ignore_items=train_warm_item_mask)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20, 50],
                                      ignore_items=train_validation_warm_item_mask)

    ##################################################
    # Content-based Parameter Tuning

    output_folder_path = f"../../results/{dataset_name}/{ICM_name}/{ItemKNNCBFRecommender.RECOMMENDER_NAME}/"

    runParameterSearch_Content(ItemKNNCBFRecommender, URM_train, ICM_train, ICM_name,
                               URM_train_last_test=URM_train_last_test, ICM_last_test=original_ICM_train,
                               n_cases=n_cases, n_random_starts=n_random_starts, resume_from_saved=True,
                               save_model='best', evaluator_validation=evaluator_validation,
                               evaluator_test=evaluator_test, output_folder_path=output_folder_path,
                               similarity_type_list=similarity_type_list)
