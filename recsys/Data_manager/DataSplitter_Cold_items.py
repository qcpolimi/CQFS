#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np

from recsys.Base.DataIO import DataIO
from recsys.Base.Recommender_utils import reshapeSparse
from recsys.Data_manager.DataReader import DataReader as _DataReader
from recsys.Data_manager.DataReader_utils import compute_density, reconcile_mapper_with_removed_tokens
from recsys.Data_manager.DataSplitter_Holdout import DataSplitter_Holdout as _DataSplitter_Holdout
from recsys.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_user_wise, split_train_in_two_percentage_global_sample, \
    split_train_in_two_percentage_cold_items


class DataSplitter_Cold_items(_DataSplitter_Holdout):
    """
    The splitter creates a random holdout of three split: train, validation and test
    The split is performed user-wise
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split


    """

    DATA_SPLITTER_NAME = "DataSplitter_Cold_items"

    SPLIT_URM_WARM_DICT = None

    def __init__(self, dataReader_object: _DataReader,
                 split_interaction_quota_list=None, user_wise=True, allow_cold_users=False,
                 forbid_new_split=False, force_new_split=False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        super(DataSplitter_Cold_items, self).__init__(dataReader_object,
                                                      split_interaction_quota_list=split_interaction_quota_list,
                                                      user_wise=user_wise, allow_cold_users=allow_cold_users,
                                                      forbid_new_split=forbid_new_split,
                                                      force_new_split=force_new_split)

    def _get_split_subfolder_name(self):
        """

        :return: warm_{n_folds}_fold/
        """

        if self.user_wise:
            user_wise_string = "user_wise"
        else:
            user_wise_string = "global_sample"

        return "cold_{}_{}/".format(
            '_'.join(str(split_quota) for split_quota in self.input_split_interaction_quota_list), user_wise_string)

    def _assert_is_initialized(self):
        assert self.SPLIT_URM_WARM_DICT is not None, "{}: Unable to load data split. The split has not been generated yet," \
                                                     " call the load_data function to do so.".format(
            self.DATA_SPLITTER_NAME)

        super(DataSplitter_Cold_items, self)._assert_is_initialized()

    def _split_data_from_original_dataset(self, save_folder_path):

        self.loaded_dataset = self.dataReader_object.load_data()
        self._load_from_DataReader_ICM_and_mappers(self.loaded_dataset)

        train_quota, validation_quota, test_quota = self.input_split_interaction_quota_list
        train_quota /= 100
        validation_quota /= 100
        test_quota /= 100

        URM_all = self.loaded_dataset.get_URM_all()

        URM_train_validation, URM_test = split_train_in_two_percentage_cold_items(URM_all.copy(),
                                                                                  train_percentage=train_quota + validation_quota)

        # Adjust train quota to account for the reduced size of the sample
        # URM_train_validation * adjusted_train_quota = URM_all * train quota
        adjusted_train_quota = URM_all.nnz * train_quota / URM_train_validation.nnz

        URM_train, URM_validation = split_train_in_two_percentage_cold_items(URM_train_validation.copy(),
                                                                             train_percentage=adjusted_train_quota)

        if self.user_wise:
            URM_train_warm, URM_validation_warm = split_train_in_two_percentage_user_wise(URM_train_validation.copy(),
                                                                                          train_percentage=adjusted_train_quota)
        else:
            URM_train_warm, URM_validation_warm = split_train_in_two_percentage_global_sample(
                URM_train_validation.copy(),
                train_percentage=adjusted_train_quota)

        if not self.allow_cold_users:

            user_interactions_cold = np.ediff1d(URM_train.indptr)
            user_to_preserve_cold = user_interactions_cold >= 1

            user_interactions_warm = np.ediff1d(URM_train_warm.indptr)
            user_to_preserve_warm = user_interactions_warm >= 1

            user_to_preserve = np.logical_and(user_to_preserve_cold, user_to_preserve_warm)
            user_to_remove = np.logical_not(user_to_preserve)

            n_users = URM_train.shape[0]

            if user_to_remove.sum() > 0:

                self._print(
                    "Removing {} ({:.2f} %) of {} users because they have no interactions in train data.".format(
                        user_to_remove.sum(), user_to_remove.sum() / n_users * 100, n_users))

                URM_train = URM_train[user_to_preserve, :]
                URM_validation = URM_validation[user_to_preserve, :]
                URM_test = URM_test[user_to_preserve, :]

                URM_train_warm = URM_train_warm[user_to_preserve, :]
                URM_validation_warm = URM_validation_warm[user_to_preserve, :]

                self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(
                    self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                    np.arange(0, len(user_to_remove), dtype=np.int)[user_to_remove])

                for UCM_name, UCM_object in self.SPLIT_UCM_DICT.items():
                    UCM_object = UCM_object[user_to_preserve, :]
                    self.SPLIT_UCM_DICT[UCM_name] = UCM_object

        # It can happen that removing users one or more of the URMs lose the last rows, so we reshape them
        URM_train, URM_validation, URM_test = self._consistency_reshape(URM_train, URM_validation, URM_test)
        URM_train_warm, URM_validation_warm, URM_test = self._consistency_reshape(URM_train_warm, URM_validation_warm,
                                                                                  URM_test)

        self.SPLIT_URM_DICT = {
            "URM_train": URM_train,
            "URM_validation": URM_validation,
            "URM_test": URM_test,
        }

        self.SPLIT_URM_WARM_DICT = {
            "URM_train_warm": URM_train_warm,
            "URM_validation_warm": URM_validation_warm,
        }

        self._compute_real_split_interaction_quota()

        self._save_split(save_folder_path)

        self._print("Split complete")

    @staticmethod
    def _consistency_reshape(URM_train, URM_validation, URM_test):

        n_users = max(URM_train.shape[0], URM_validation.shape[0], URM_test.shape[0])
        n_items = max(URM_train.shape[1], URM_validation.shape[1], URM_test.shape[1])
        new_shape = (n_users, n_items)

        URM_train = reshapeSparse(URM_train, new_shape)
        URM_validation = reshapeSparse(URM_validation, new_shape)
        URM_test = reshapeSparse(URM_test, new_shape)

        return URM_train, URM_validation, URM_test

    def get_both_splits(self):
        """
        :return: URM_train, URM_validation, URM_test, URM_train_warm, URM_validation_warm
        """

        self._assert_is_initialized()

        return self.SPLIT_URM_DICT["URM_train"].copy(), \
               self.SPLIT_URM_DICT["URM_validation"].copy(), \
               self.SPLIT_URM_DICT["URM_test"].copy(), \
               self.SPLIT_URM_WARM_DICT["URM_train_warm"].copy(), \
               self.SPLIT_URM_WARM_DICT["URM_validation_warm"].copy()

    def get_cold_split(self):
        """
        :return: URM_train, URM_validation, URM_test
        """

        return self.get_holdout_split()

    def get_warm_split(self):
        """
        :return: URM_train_warm, URM_validation_warm, URM_test
        """

        self._assert_is_initialized()

        return self.SPLIT_URM_WARM_DICT["URM_train_warm"].copy(), \
               self.SPLIT_URM_WARM_DICT["URM_validation_warm"].copy(), \
               self.SPLIT_URM_DICT["URM_test"].copy()

    def get_filtered_ICM_from_name(self, ICM_name, categories, exceptions=None):

        ICM = self.SPLIT_ICM_DICT[ICM_name].copy()
        mapper = self.SPLIT_ICM_MAPPER_DICT[ICM_name]
        categories = tuple(categories)

        if exceptions is None:
            exceptions = []

        inv_mapper = {mapper[k]: k for k in mapper}

        new_to_original_mapper = {}
        for key in mapper:
            if key.startswith(categories) and key not in exceptions:
                new_to_original_mapper[key] = mapper[key]

        new_mapper = {}
        i = 0
        for key in range(len(inv_mapper)):
            feature = inv_mapper[key]
            if feature.startswith(categories) and feature not in exceptions:
                new_mapper[feature] = i
                i += 1

        selected_features = np.array(list(new_to_original_mapper.values()))
        ICM = ICM[:, selected_features]

        return ICM, (new_to_original_mapper, new_mapper)

    def _save_split(self, save_folder_path):

        if save_folder_path:

            if self.allow_cold_users:
                allow_cold_users_suffix = "allow_cold_users"

            else:
                allow_cold_users_suffix = "only_warm_users"

            if self.user_wise:
                user_wise_string = "user_wise"
            else:
                user_wise_string = "global_sample"

            name_suffix = "_{}_{}".format(allow_cold_users_suffix, user_wise_string)

            dataIO = DataIO(folder_path=save_folder_path)

            dataIO.save_data(data_dict_to_save=self.SPLIT_URM_WARM_DICT,
                             file_name="split_URM_warm" + name_suffix)

            super(DataSplitter_Cold_items, self)._save_split(save_folder_path)

    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads warm split URMs
        :return:
        """

        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"

        if self.user_wise:
            user_wise_string = "user_wise"
        else:
            user_wise_string = "global_sample"

        name_suffix = "_{}_{}".format(allow_cold_users_suffix, user_wise_string)

        dataIO = DataIO(folder_path=save_folder_path)

        self.SPLIT_URM_WARM_DICT = dataIO.load_data(file_name="split_URM_warm" + name_suffix)

        super(DataSplitter_Cold_items, self)._load_previously_built_split_and_attributes(save_folder_path)

    def get_statistics_URM_warm(self):

        self._assert_is_initialized()

        n_users, n_items = self.SPLIT_URM_WARM_DICT["URM_train_warm"].shape

        statistics_string = "DataReader: {}\n" \
                            "\tNum items: {}\n" \
                            "\tNum users: {}\n" \
                            "\tTrain \t\tquota {:.2f} ({:.2f}), \tinteractions {}, \tdensity {:.2E}\n" \
                            "\tValidation \tquota {:.2f} ({:.2f}), \tinteractions {}, \tdensity {:.2E}\n" \
                            "\tTest \t\tquota {:.2f} ({:.2f}), \tinteractions {}, \tdensity {:.2E}\n".format(
            self.dataReader_object._get_dataset_name(),
            n_items,
            n_users,
            self.input_split_interaction_quota_list[0], self.actual_split_interaction_quota_list[0],
            self.SPLIT_URM_WARM_DICT["URM_train_warm"].nnz, compute_density(self.SPLIT_URM_WARM_DICT["URM_train_warm"]),
            self.input_split_interaction_quota_list[1], self.actual_split_interaction_quota_list[1],
            self.SPLIT_URM_WARM_DICT["URM_validation_warm"].nnz,
            compute_density(self.SPLIT_URM_WARM_DICT["URM_validation_warm"]),
            self.input_split_interaction_quota_list[2], self.actual_split_interaction_quota_list[2],
            self.SPLIT_URM_DICT["URM_test"].nnz, compute_density(self.SPLIT_URM_DICT["URM_test"]),
        )

        self._print(statistics_string)

        print("\n")
