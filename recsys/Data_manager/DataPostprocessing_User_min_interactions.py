#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from recsys.Base.Recommender_utils import check_matrix

from recsys.Data_manager.DataPostprocessing import DataPostprocessing
from recsys.Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens, remove_features, remove_empty_rows_and_cols
import numpy as np
import scipy.sparse as sps



class DataPostprocessing_User_min_interactions(DataPostprocessing):
    """
    This class selects a partition of URM such that all users have at least min_interactions.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """


    def __init__(self, dataReader_object, min_interactions):

        assert min_interactions >= 1,\
            "DataReaderPostprocessing_User_min_interactions: min_interactions must be a positive value >= 1, provided value was {}".format(min_interactions)

        super(DataPostprocessing_User_min_interactions, self).__init__(dataReader_object)

        self.min_interactions = min_interactions



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "{}_user_min_interactions/".format(self.min_interactions)

        inner_subfolder_name = self.dataReader_object._get_dataset_name_data_subfolder()

        # Avoid concatenating the original/ part
        if inner_subfolder_name != self.DATASET_SUBFOLDER_ORIGINAL:
            subfolder_name += inner_subfolder_name

        return subfolder_name



    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the k-cores
        :return:
        """

        loaded_dataset = self.dataReader_object.load_data()
        URM_all = loaded_dataset.AVAILABLE_URM["URM_all"]

        # Apply required min user interactions ORIGINAL split
        _, users_to_remove, items_to_remove = select_users_with_min_interactions(URM_all, min_interactions = self.min_interactions)

        loaded_dataset._remove_items_and_users(items_to_remove = items_to_remove, users_to_remove = users_to_remove)

        return loaded_dataset








def select_users_with_min_interactions(URM, min_interactions = 5, reshape = False):
    """

    :param URM:
    :param min_interactions:
    :param reshape:
    :return: URM, removedUsers, removedItems
    """

    print("DataReaderPostprocessing_User_min_interactions: min_interactions extraction will zero out some users and items without changing URM shape")

    URM.eliminate_zeros()

    n_users = URM.shape[0]
    n_items = URM.shape[1]


    print("DataReaderPostprocessing_User_min_interactions: Initial URM desity is {:.2E}".format(URM.nnz/(n_users*n_items)))

    n_users, n_items = URM.shape

    URM = sps.csr_matrix(URM)
    user_to_remove_mask = np.ediff1d(URM.indptr) < min_interactions
    removed_users = np.arange(0, n_users, dtype=np.int)[user_to_remove_mask]


    for user in removed_users:
        start_pos = URM.indptr[user]
        end_pos = URM.indptr[user + 1]

        URM.data[start_pos:end_pos] = np.zeros_like(URM.data[start_pos:end_pos])

    URM.eliminate_zeros()

    URM = sps.csc_matrix(URM)
    items_to_remove_mask = np.ediff1d(URM.indptr) == 0
    removed_items = np.arange(0, n_items, dtype=np.int)[items_to_remove_mask]


    if URM.data.sum() == 0:
        print("DataReaderPostprocessing_User_min_interactions: WARNING URM is empty.")

    else:
         print("DataReaderPostprocessing_User_min_interactions: URM desity without zeroed-out nodes is {:.2E}.\n"
              "Users with less than {} interactions are {} ( {:.2f}%), Items are {} ( {:.2f}%)".format(
            sum(URM.data)/((n_users-len(removed_users))*(n_items-len(removed_items))),
            min_interactions,
            len(removed_users), len(removed_users)/n_users*100,
            len(removed_items), len(removed_items)/n_items*100))


    print("DataReaderPostprocessing_User_min_interactions: split complete")

    URM = sps.csr_matrix(URM)

    if reshape:
        # Remove all columns and rows with no interactions
        return remove_empty_rows_and_cols(URM)


    return URM.copy(), removed_users, removed_items
