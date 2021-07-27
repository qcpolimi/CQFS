import numpy as np

from recsys.Data_manager import XingChallenge2017Reader
from recsys.Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
from recsys.Data_manager.DataSplitter_Cold_items import DataSplitter_Cold_items
from utils.recsys import remove_ICM_item_interactions
from utils.sparse import merge_sparse_matrices


def main():
    CATEGORIES = ["career_level", "discipline", "industry", "country", "region", "is_paid", "employment"]

    ##################################################
    # Data loading and splitting

    data_reader = XingChallenge2017Reader()

    # Process the dataset to keep only users and items that have at least 5 interactions (5 cores)
    data_reader_5_cores = DataPostprocessing_K_Cores(data_reader, 5)

    # Split the data into train, validation and test (70, 10, 20) through the DataSplitter
    # The test and validation splits are cold items splits
    data_splitter = DataSplitter_Cold_items(data_reader_5_cores)
    data_splitter.load_data()

    # Get the split
    URM_train, URM_validation, URM_test, URM_train_warm, URM_validation_warm = data_splitter.get_both_splits()
    data_splitter.get_statistics_URM_warm()

    # Get the warm items URM
    URM_train_validation = merge_sparse_matrices(URM_train, URM_validation).tocsr()
    URM_train_validation_warm = merge_sparse_matrices(URM_train_warm, URM_validation_warm).tocsr()
    assert URM_train_validation.nnz == URM_train_validation_warm.nnz, "WRONG SPLIT!"
    assert np.equal(URM_train_validation.indptr, URM_train_validation_warm.indptr).all(), "WRONG SPLIT!"
    assert np.equal(URM_train_validation.indices, URM_train_validation_warm.indices).all(), "WRONG SPLIT!"

    # Get the original ICM
    ICM_name = 'ICM_all'
    original_ICM_train, mapper_ICM_train = data_splitter.get_filtered_ICM_from_name(ICM_name, CATEGORIES)

    ##################################################
    # ICM preparation

    # Remove cold items interactions from the ICM
    test_items = np.ediff1d(URM_test.tocsc().indptr) != 0
    test_items = np.arange(len(test_items))[test_items]
    ICM_train = remove_ICM_item_interactions(original_ICM_train, test_items)

    # Check if the removal was correct
    no_ICM_interaction_item_mask = np.ediff1d(ICM_train.indptr) == 0
    assert np.alltrue(
        no_ICM_interaction_item_mask[test_items]), "Test items were not correctly removed from the train ICM."

    n_items, n_features = ICM_train.shape

    print(f"Training ICM has {n_items} items and {n_features} features.")


if __name__ == "__main__":
    main()
