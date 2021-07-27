#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from recsys.Data_manager.DataPostprocessing import DataPostprocessing
from recsys.Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens, remove_features
from recsys.Data_manager.DataPostprocessing_K_Cores import select_k_cores
import numpy as np
import scipy.sparse as sps


class DataPostprocessing_User_sample(DataPostprocessing):
    """
    This class selects a partition of URM such that only some of the original users are present
    """


    def __init__(self, dataReader_object, user_quota = 1.0):

        assert user_quota > 0.0 and user_quota <= 1.0,\
            "DataReaderPostprocessing_User_sample: user_quota must be a positive value > 0.0 and <= 1.0, provided value was {}".format(user_quota)

        super(DataPostprocessing_User_sample, self).__init__(dataReader_object)

        self.user_quota = user_quota


    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "{}_user_sample/".format(self.user_quota)

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

        n_users, n_items = URM_all.shape

        num_users_to_select = int(n_users*self.user_quota)

        print("DataReaderPostprocessing_User_sample: Sampling {:.2f} % of all users, their number is {}".format(self.user_quota*100, num_users_to_select))

        user_id_list = np.arange(0, n_users, dtype=np.int)

        np.random.shuffle(user_id_list)

        # sampled_user_id_list = user_id_list[0:num_users_to_select]
        removed_users = user_id_list[num_users_to_select:]

        URM_all = sps.csr_matrix(URM_all)

        # Remove those user's interactions
        for user_id in removed_users:

            start_pos = URM_all.indptr[user_id]
            end_pos = URM_all.indptr[user_id + 1]

            URM_all.data[start_pos:end_pos] = 0

        URM_all.eliminate_zeros()

        # Apply K - core to remove items with no interactions
        _, users_to_remove, items_to_remove = select_k_cores(URM_all, k_value = 1)

        loaded_dataset._remove_items_and_users(items_to_remove = items_to_remove, users_to_remove = users_to_remove)

        return loaded_dataset



