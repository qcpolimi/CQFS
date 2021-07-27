import numpy as np

from recsys.Data_manager import TheMoviesDatasetReader, XingChallenge2017Reader
from recsys.Data_manager.CiteULike.CiteULikeReader import CiteULike_aReader
from recsys.Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
from recsys.Data_manager.DataSplitter_Cold_items import DataSplitter_Cold_items
from utils.recsys import remove_ICM_item_interactions


class DataLoader:

    def __init__(self, data_reader, preprocessing=None, preprocessing_params=None):
        # self.data_reader = data_reader
        data_splitter = data_reader()

        if preprocessing is not None:
            assert preprocessing_params is not None and len(preprocessing) == len(preprocessing_params), \
                "Please, provide preprocessing parameters for each preprocessing reader."

            for pi in range(len(preprocessing)):
                p = preprocessing[pi]
                params = preprocessing_params[pi]
                data_splitter = p(data_splitter, **params)

        self.data_splitter = DataSplitter_Cold_items(data_splitter)

        self.available_ICMs = {}
        self.feature_mappers = {}

    def load_data(self, save_folder_path=None):
        self.data_splitter.load_data(save_folder_path)

    def get_dataset_name(self):
        return self.data_splitter._get_dataset_name()

    def get_splitter(self):
        return self.data_splitter

    def get_both_splits(self):
        return self.data_splitter.get_both_splits()

    def get_cold_split(self):
        return self.data_splitter.get_cold_split()

    def get_warm_split(self):
        return self.data_splitter.get_warm_split()

    def get_ICM_from_name(self, ICM_name):
        ICM = self.available_ICMs.get(ICM_name)

        if ICM is None:
            ICM = self.data_splitter.get_ICM_from_name(ICM_name)
            self.available_ICMs[ICM_name] = ICM.copy()
            self.feature_mappers[ICM_name] = self.data_splitter.SPLIT_ICM_MAPPER_DICT[ICM_name]

        return ICM

    def get_original_ICM_train_from_name(self, ICM_name):
        raise NotImplementedError("This method is not implemented in the parent class.")

    def get_ICM_train_from_name(self, ICM_name, return_original=False):
        original_ICM_train = self.get_original_ICM_train_from_name(ICM_name)

        # Remove cold items interactions from the ICM
        self.data_splitter._assert_is_initialized()
        URM_test = self.data_splitter.SPLIT_URM_DICT["URM_test"]

        test_items = np.ediff1d(URM_test.tocsc().indptr) != 0
        test_items = np.arange(len(test_items))[test_items]
        ICM_train = remove_ICM_item_interactions(original_ICM_train, test_items)

        # Check if the removal was correct
        no_ICM_interaction_item_mask = np.ediff1d(ICM_train.indptr) == 0
        assert np.alltrue(
            no_ICM_interaction_item_mask[test_items]), "Test items were not correctly removed from the train ICM."

        if return_original:
            return ICM_train, original_ICM_train

        return ICM_train

    def get_feature_to_index_mapper_from_name(self, ICM_name):
        return self.feature_mappers.get(ICM_name)

    def get_index_to_feature_mapper_from_name(self, ICM_name):
        mapper = self.feature_mappers.get(ICM_name)
        return {mapper[k]: k for k in mapper}


class TheMoviesDatasetLoader(DataLoader):

    def __init__(self, preprocessing=None, preprocessing_params=None):
        super(TheMoviesDatasetLoader, self).__init__(TheMoviesDatasetReader, preprocessing, preprocessing_params)

    def get_ICM_from_name(self, ICM_name='ICM_metadata'):
        return super(TheMoviesDatasetLoader, self).get_ICM_from_name(ICM_name)

    def get_original_ICM_train_from_name(self, ICM_name='ICM_metadata'):
        # Filter the ICM removing all the features with less than 5 interactions
        ICM = self.get_ICM_from_name(ICM_name)
        features_with_more_than_5_interactions = np.ediff1d(ICM.tocsc().indptr) >= 5

        original_features = np.array(list(self.data_splitter.SPLIT_ICM_MAPPER_DICT[ICM_name].keys()))
        filtered_features = original_features[features_with_more_than_5_interactions]
        filtered_features_index = np.arange(len(features_with_more_than_5_interactions))[
            features_with_more_than_5_interactions]

        # self.feature_mappers[ICM_name] = {filtered_features[i]: filtered_features_index[i] for i in
        #                                   range(len(filtered_features))}

        self.feature_mappers[ICM_name] = {filtered_features[i]: i for i in range(len(filtered_features))}

        return ICM[:, features_with_more_than_5_interactions]

    def get_ICM_train_from_name(self, ICM_name='ICM_metadata', return_original=False):
        return super(TheMoviesDatasetLoader, self).get_ICM_train_from_name(ICM_name, return_original)


class CiteULike_aLoader(DataLoader):

    def __init__(self, preprocessing=None, preprocessing_params=None):
        super(CiteULike_aLoader, self).__init__(CiteULike_aReader, preprocessing, preprocessing_params)

    def get_ICM_from_name(self, ICM_name='ICM_title_abstract'):
        return super(CiteULike_aLoader, self).get_ICM_from_name(ICM_name)

    def get_original_ICM_train_from_name(self, ICM_name='ICM_title_abstract'):
        return self.get_ICM_from_name(ICM_name)

    def get_ICM_train_from_name(self, ICM_name='ICM_title_abstract', return_original=False):
        return super(CiteULike_aLoader, self).get_ICM_train_from_name(ICM_name, return_original)


class XingChallenge2017Loader(DataLoader):
    CATEGORIES = ["career_level", "discipline", "industry", "country", "region", "is_paid", "employment"]
    EXCEPTIONS = []

    def __init__(self, preprocessing=None, preprocessing_params=None):
        if preprocessing is None:
            preprocessing = [DataPostprocessing_K_Cores]
            preprocessing_params = [{'k_cores_value': 5}]

        super(XingChallenge2017Loader, self).__init__(XingChallenge2017Reader, preprocessing, preprocessing_params)

    def get_ICM_from_name(self, ICM_name='ICM_all'):
        return super(XingChallenge2017Loader, self).get_ICM_from_name(ICM_name)

    def get_original_ICM_train_from_name(self, ICM_name='ICM_all'):
        original_ICM_train, mappers_ICM_train = self.data_splitter.get_filtered_ICM_from_name(ICM_name, self.CATEGORIES,
                                                                                              self.EXCEPTIONS)

        self.feature_mappers[ICM_name] = mappers_ICM_train[1]
        return original_ICM_train

    def get_ICM_train_from_name(self, ICM_name='ICM_all', return_original=False):
        return super(XingChallenge2017Loader, self).get_ICM_train_from_name(ICM_name, return_original)
