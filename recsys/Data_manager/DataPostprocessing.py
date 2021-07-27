#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Maurizio Ferrari Dacrema
"""

from recsys.Data_manager.DataReader import DataReader



class DataPostprocessing(DataReader):
    """
    This class provides the interface for the DataReaderPostprocessing objects
    """


    def __init__(self, dataReader_object: DataReader):
        super(DataPostprocessing, self).__init__()

        self.dataReader_object = dataReader_object

        self.AVAILABLE_URM = dataReader_object.AVAILABLE_URM.copy()
        self.AVAILABLE_ICM = dataReader_object.AVAILABLE_ICM.copy()
        self.AVAILABLE_UCM = dataReader_object.AVAILABLE_UCM.copy()

    #
    # def get_all_available_ICM_names(self):
    #     return self.dataReader_object.get_all_available_ICM_names()
    #
    # def get_loaded_ICM_names(self):
    #     return self.dataReader_object.get_loaded_ICM_names()

    def _get_dataset_name(self):
        return self.dataReader_object._get_dataset_name()

    def _get_dataset_name_root(self):
        return self.dataReader_object._get_dataset_name_root()

    # def get_item_original_ID_to_index_mapper(self):
    #     return self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"].copy()
    #
    # def get_user_original_ID_to_index_mapper(self):
    #     return self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"].copy()

    def is_implicit(self):
        return self.dataReader_object.is_implicit()



    # def _load_from_DataReader_ICM_and_mappers(self):
    #
    #     self._LOADED_URM_DICT = {}
    #     self._LOADED_ICM_DICT = {}
    #     self._LOADED_ICM_MAPPER_DICT = {}
    #     self._LOADED_GLOBAL_MAPPER_DICT = {}
    #
    #     for URM_name in self.dataReader_object.get_loaded_URM_names():
    #         self._LOADED_URM_DICT[URM_name] = self.dataReader_object.get_URM_from_name(URM_name)
    #
    #     for ICM_name in self.dataReader_object.get_loaded_ICM_names():
    #         self._LOADED_ICM_DICT[ICM_name] = self.dataReader_object.get_ICM_from_name(ICM_name)
    #         self._LOADED_ICM_MAPPER_DICT[ICM_name] = self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name)
    #
    #     for mapper_name, mapper_object in self.dataReader_object._LOADED_GLOBAL_MAPPER_DICT.items():
    #         self._LOADED_GLOBAL_MAPPER_DICT[mapper_name] = mapper_object.copy()



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: Dataset_name/
        """
        raise NotImplementedError("DataReaderPostprocessing: The following method was not implemented for the required class.")



    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the postprocessing required
        :return:
        """
        raise NotImplementedError("DataReaderPostprocessing: The following method was not implemented for the required class.")
