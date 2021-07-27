#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Maurizio Ferrari Dacrema
"""

from recsys.Base.DataIO import DataIO

from recsys.Data_manager.DataSplitter import DataSplitter as _DataSplitter
from recsys.Data_manager.DataReader import DataReader as _DataReader


class DataSplitter_k_fold_random(_DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    
    """

    DATA_SPLITTER_NAME = "DataSplitter_k_fold_random"

    FOLD_DATA_SPLITTER_LIST = None



    def __init__(self, dataReader_object:_DataReader, dataSplitter_class,
                 dataSplitter_kwargs = None, preload_all = True,
                 n_folds = 5, forbid_new_split = False, force_new_split = False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        """


        assert n_folds>1, "{}: Number of folds must be  greater than 1".format(self.DATA_SPLITTER_NAME)

        super(DataSplitter_k_fold_random, self).__init__(dataReader_object,
                                                         forbid_new_split = forbid_new_split,
                                                         force_new_split = force_new_split)

        self.dataSplitter_class = dataSplitter_class
        self.n_folds = n_folds
        self.preload_all = preload_all

        if dataSplitter_kwargs is None:
            self.dataSplitter_kwargs = {}
        else:
            self.dataSplitter_kwargs = dataSplitter_kwargs.copy()

        # DataSplitter object without load data, to be used to get the subfolder paths
        self._dataSplitter_object_empty = self.dataSplitter_class(self.dataReader_object, **self.dataSplitter_kwargs)


    def _assert_is_initialized(self):
         assert self.FOLD_DATA_SPLITTER_LIST is not None, "{}: Unable to load data split. The split has not been generated yet, call the load_data function to do so.".format(self.DATA_SPLITTER_NAME)


    def get_statistics_URM(self):
        pass
        #
        # # This avoids the fixed bit representation of numpy preventing
        # # an overflow when computing the product
        # n_items = int(self.n_items)
        # n_users = int(self.n_users)
        #
        # print("DataSplitter_k_fold for DataReader: {}\n"
        #       "\t Num items: {}\n"
        #       "\t Num users: {}\n".format(self.dataReader_object._get_dataset_name(), n_items, n_users))
        #
        #
        # n_global_interactions = 0
        #
        # for fold_index in range(self.n_folds):
        #     URM_fold_object = self.fold_split[fold_index]["URM"]
        #     n_global_interactions += URM_fold_object.nnz
        #
        #
        # for fold_index in range(self.n_folds):
        #     URM_fold_object = self.fold_split[fold_index]["URM"]
        #     items_in_fold = self.fold_split[fold_index]["items_in_fold"]
        #
        #
        #     print("\t Statistics for fold {}: n_interactions {} ( {:.2f}%), n_items {} ( {:.2f}%), density: {:.2E}".format(
        #         fold_index,
        #         URM_fold_object.nnz, URM_fold_object.nnz/n_global_interactions*100,
        #         len(items_in_fold), len(items_in_fold)/n_items*100,
        #         URM_fold_object.nnz/(int(n_items)*int(n_users))
        #     ))
        #
        # print("\n")




    def get_statistics_ICM(self):
        pass
        #
        # for ICM_name in self.dataReader_object.get_loaded_ICM_names():
        #
        #     ICM_object = getattr(self, ICM_name)
        #     n_items = ICM_object.shape[0]
        #     n_features = ICM_object.shape[1]
        #
        #     print("\t Statistics for {}: n_features {}, feature occurrences {}, density: {:.2E}".format(
        #         ICM_name, n_features, ICM_object.nnz, ICM_object.nnz/(int(n_items)*int(n_features))
        #     ))
        #
        # print("\n")
        #

    #
    # def get_fold_split(self):
    #     self._assert_is_initialized()
    #     return self.FOLD_URM_LIST
    #
    #
    # def get_fold(self, n_fold):
    #     self._assert_is_initialized()
    #     return self.FOLD_URM_LIST[n_fold]["URM"].copy()



    # def get_URM_train_for_test_fold(self, n_test_fold):
    #     """
    #     The train set is defined as all data except the one of that fold, which is the test
    #     :param n_fold:
    #     :return:
    #     """
    #
    #     self._assert_is_initialized()
    #
    #     URM_test = self.FOLD_URM_LIST[n_test_fold]["URM"].copy()
    #
    #     URM_train = sps.csr_matrix(URM_test.shape)
    #
    #     for fold_index in range(self.n_folds):
    #
    #         if fold_index != n_test_fold:
    #             URM_fold_object = self.FOLD_URM_LIST[fold_index]["URM"]
    #
    #             URM_train+=URM_fold_object
    #
    #
    #     return URM_train, URM_test


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                ITERATOR                                             ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def __iter__(self):

        self._assert_is_initialized()

        self.__iterator_current_fold = 0
        return self


    def __next__(self):

        fold_to_return = self.__iterator_current_fold

        if self.__iterator_current_fold >= self.n_folds:
            raise StopIteration

        self.__iterator_current_fold += 1

        return self[fold_to_return]





    def __getitem__(self, n_fold):
        """
        :param index:
        :return:
        """

        self._assert_is_initialized()

        return self.FOLD_DATA_SPLITTER_LIST[n_fold]




    def __len__(self):

        return self.n_folds





########################################################################################################################
##############################################
##############################################          WARM ITEMS
##############################################



class DataSplitter_k_fold_random_fromDataSplitter(DataSplitter_k_fold_random):
    """
    This splitter performs a Holdout from the full URM splitting in train, test and validation
    Ensures that every user has at least an interaction in all splits
    """

    def _get_split_subfolder_name(self):
        """

        :return: random_{n_folds}_fold/
        """
        inner_subfolder_name = self._dataSplitter_object_empty._get_split_subfolder_name()

        return "random_{}_fold/{}".format(self.n_folds, inner_subfolder_name)



    def _split_data_from_original_dataset_fold(self, save_folder_path, fold_index):

        dataSplitter_object_fold = self.dataSplitter_class(self.dataReader_object, **self.dataSplitter_kwargs)

        # Either save or not
        if save_folder_path:
            save_folder_path_fold = save_folder_path + "fold_{}/".format(fold_index +1)
            self._print("Splitting Fold {}/{}. Saving in '{}'".format(fold_index+ 1, self.n_folds, save_folder_path_fold))

        else:
            save_folder_path_fold = False
            self._print("Splitting Fold {}/{}. Not saving.".format(fold_index+ 1, self.n_folds))


        dataSplitter_object_fold.load_data(save_folder_path = save_folder_path_fold)

        return dataSplitter_object_fold



    def _split_data_from_original_dataset(self, save_folder_path):

        # Perform k random splits.

        self.FOLD_DATA_SPLITTER_LIST = [None]*self.n_folds

        # If save_folder_path is None use default
        if save_folder_path is None and not save_folder_path == False:
            save_folder_path = self._get_default_save_path()


        for fold_index in range(self.n_folds):

            dataSplitter_object_fold = self._split_data_from_original_dataset_fold(save_folder_path, fold_index)

            if self.preload_all:
                self.FOLD_DATA_SPLITTER_LIST[fold_index] = dataSplitter_object_fold



        if save_folder_path:
            split_parameters_dict = {"n_folds": self.n_folds,
                                 }

            dataIO = DataIO(folder_path = save_folder_path)

            dataIO.save_data(data_dict_to_save = split_parameters_dict,
                             file_name = "split_parameters")


        self._print("Split complete")





    def __getitem__(self, n_fold):
        """
        :param index:
        :return:
        """

        self._assert_is_initialized()

        if not self.preload_all:

            # Check if fold was loaded, if not, reset list and load only the required one
            if self.FOLD_DATA_SPLITTER_LIST[n_fold] is None:
                self.FOLD_DATA_SPLITTER_LIST = [None]*self.n_folds
                self.FOLD_DATA_SPLITTER_LIST[n_fold] = self._load_previously_built_split_and_attributes_fold(self.__save_folder_path, n_fold)

        return self.FOLD_DATA_SPLITTER_LIST[n_fold]



    def _load_previously_built_split_and_attributes_fold(self, save_folder_path, fold_index):

        dataSplitter_object = self.dataSplitter_class(self.dataReader_object, **self.dataSplitter_kwargs)

        save_folder_path_fold = save_folder_path + "fold_{}/".format(fold_index +1)

        self._print("Splitting Fold {}/{}. Loading from '{}'".format(fold_index+ 1, self.n_folds, save_folder_path_fold))
        dataSplitter_object.load_data(save_folder_path = save_folder_path_fold)

        return dataSplitter_object


    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """

        self.__save_folder_path = save_folder_path

        dataIO = DataIO(folder_path = save_folder_path)

        split_parameters_dict = dataIO.load_data(file_name ="split_parameters")

        for attrib_name in split_parameters_dict.keys():
             self.__setattr__(attrib_name, split_parameters_dict[attrib_name])


        self.FOLD_DATA_SPLITTER_LIST = [None]*self.n_folds

        for fold_index in range(self.n_folds):

            dataSplitter_object = self._load_previously_built_split_and_attributes_fold(save_folder_path, fold_index)

            if self.preload_all:
                self.FOLD_DATA_SPLITTER_LIST[fold_index] = dataSplitter_object





