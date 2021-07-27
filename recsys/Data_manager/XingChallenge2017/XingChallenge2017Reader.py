#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import zipfile, os, shutil
from recsys.Data_manager.Dataset import Dataset
from recsys.Data_manager.DataReader import DataReader


class XingChallenge2017Reader(DataReader):
    DATASET_URL = "NOT AVAILABLE, PRIVATE DATASET"
    DATASET_SUBFOLDER = "XingChallenge2017/"
    AVAILABLE_URM = ["URM_all", "URM_positive", "URM_negative"]
    AVAILABLE_ICM = ["ICM_all"]

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "xing_challenge_data_2017.zip"

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            interactions_path = dataFile.extract("data/interactions_14.csv",
                                                 path=decompressed_zip_file_folder + "decompressed/")

            ICM_path = dataFile.extract("data/items.csv", path=decompressed_zip_file_folder + "decompressed/")
            # UCM_path = dataFile.extract("data/users.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(
                compressed_zip_file_folder))
            self._print(
                "Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        # print("XingChallenge2017Reader: Loading Impressions")
        # self.URM_impressions = self._load_impressions(impressions_path, if_new_user = "add", if_new_item = "add")

        self._print("Loading item content")
        ICM_all, tokenToFeatureMapper_ICM_all, self.item_original_ID_to_index = self._load_ICM(ICM_path,
                                                                                               if_new_item="add")

        self._print("Loading Interactions")
        URM_positive, URM_negative, _, _, self.user_original_ID_to_index = self._load_interactions(interactions_path,
                                                                                                   if_new_user="add",
                                                                                                   if_new_item="ignore")

        URM_all = URM_positive.copy()
        URM_all.data = np.ones_like(URM_all.data)

        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_positive": URM_positive,
                           "URM_negative": URM_negative,
                           # "URM_impression": URM_impression
                           }

        loaded_ICM_dict = {"ICM_all": ICM_all}
        loaded_ICM_mapper_dict = {"ICM_all": tokenToFeatureMapper_ICM_all}

        loaded_dataset = Dataset(dataset_name=self._get_dataset_name(),
                                 URM_dictionary=loaded_URM_dict,
                                 ICM_dictionary=loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary=loaded_ICM_mapper_dict,
                                 UCM_dictionary=None,
                                 UCM_feature_mapper_dictionary=None,
                                 user_original_ID_to_index=self.user_original_ID_to_index,
                                 item_original_ID_to_index=self.item_original_ID_to_index,
                                 is_implicit=self.IS_IMPLICIT,
                                 )

        self._print("cleaning temporary files")

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset

    def _load_interactions(self, impressions_path, if_new_user="add", if_new_item="ignore"):

        from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=self.item_original_ID_to_index,
                                                        on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        fileHandle = open(impressions_path, "r")

        numCells = 0

        # Remove header
        fileHandle.readline()

        for line in fileHandle:

            if numCells % 1000000 == 0 and numCells != 0:
                print("Processed {} cells".format(numCells))

            line = line.split("\t")

            """

            Interactions that the user performed on the job posting items. Fields:

            user_id ID          of the user who performed the interaction (points to users.id)
            item_id ID          of the item on which the interaction was performed (points to items.id)
            created_at          a unix time stamp timestamp representing the time when the interaction got created
            interaction_type    the type of interaction that was performed on the item:
                0 = XING showed this item to a user (= impression)
                1 = the user clicked on the item
                2 = the user bookmarked the item on XING
                3 = the user clicked on the reply button or application form button that is shown on some job postings
                4 = the user deleted a recommendation from his/her list of recommendation (clicking on "x") which has the effect that the recommendation will no longer been shown to the user and that a new recommendation item will be loaded and displayed to the user
                5 = (not used) a recruiter from the items company showed interest into the user. (e.g. clicked on the profile)

            """

            user_id = line[0]
            item_id = line[1]
            created_at = line[3]

            interaction_type = int(line[2])
            if interaction_type == 0:
                interaction_type = -1

            else:
                if interaction_type != 5:
                    URM_builder.add_data_lists([user_id], [item_id], [interaction_type])
                    numCells += 1

        fileHandle.close()

        URM_positive = URM_builder.get_SparseMatrix()

        # Negative interactions in a separate URM
        URM_negative = URM_positive.copy()
        URM_impression = URM_positive.copy()

        URM_negative.data[URM_negative.data != 4] = 0
        URM_negative.eliminate_zeros()
        URM_negative.data = np.ones_like(URM_negative.data)

        URM_impression.data[URM_impression.data != -1] = 0
        URM_impression.eliminate_zeros()
        URM_impression.data = np.ones_like(URM_impression.data)

        URM_positive.data[URM_positive.data == 4] = 0
        URM_positive.data[URM_positive.data == -1] = 0
        URM_positive.eliminate_zeros()

        return URM_positive, URM_negative, URM_impression, URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()

    def _load_ICM(self, ICM_path, if_new_item="ignore"):

        from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=None, on_new_row=if_new_item)

        fileHandle = open(ICM_path, "r")

        numCells = 0

        # Remove header
        # item_id	title	career_level	discipline_id	industry_id	country	is_payed	region	latitude	longitude	employment	tags	created_at
        fileHandle.readline()

        for line in fileHandle:

            if numCells % 1000000 == 0 and numCells != 0:
                print("Processed {} cells".format(numCells))

            line = line.split("\t")

            """
            ORDERING IN CSV FILE
            # item_id	
            title	
            career_level	
            discipline_id	
            industry_id	
            country	
            is_payed	
            region	
            latitude	
            longitude	
            employment
            tags	
            created_at


            id anonymized ID        of the item (referenced as item_id in the other datasets above)
            industry_id             anonymized IDs represent industries such as "Internet", "Automotive", "Finance", etc.
            discipline_id           anonymized IDs represent disciplines such as "Consulting", "HR", etc.
            is_paid (or is_payed)   indicates that the posting is a paid for by a compnay
            career_level            career level ID (e.g. beginner, experienced, manager)
                0 = unknown
                1 = Student/Intern
                2 = Entry Level (Beginner)
                3 = Professional/Experienced
                4 = Manager (Manager/Supervisor)
                5 = Executive (VP, SVP, etc.)
                6 = Senior Executive (CEO, CFO, President)
            country                 code of the country in which the job is offered
            latitude                latitude information (rounded to ca. 10km)
            longitude               longitude information (rounded to ca. 10km)
            region                  is specified for some users who have as country `de`. Meaning of the regions: see below.
            employment              the type of emploment
                0 = unknown
                1 = full-time
                2 = part-time
                3 = freelancer
                4 = intern
                5 = voluntary
            created_at              a unix time stamp timestamp representing the time when the interaction got created
            title                   concepts that have been extracted from the job title of the job posting (numeric IDs)
            tags                    concepts that have been extracted from the tags, skills or company name
            """

            item_id = line[0]

            title_id_list = line[1]
            title_id_list = ["title_" + str(ID) for ID in title_id_list.split(",")]

            career_level = "career_level_" + str(line[2])

            discipline_id_list = line[3]
            discipline_id_list = ["discipline_" + str(ID) for ID in discipline_id_list.split(",")]

            industry_id_list = line[4]
            industry_id_list = ["industry_" + str(ID) for ID in industry_id_list.split(",")]

            country = "country_" + str(line[5])
            is_paid = "is_paid_" + str(line[6])

            region = "region_" + str(line[7])

            latitude = "latitude_" + str(line[8])
            longitude = "longitude_" + str(line[9])

            employment = "employment_" + str(line[10])

            tags_list = line[11]
            tags_list = ["tags_" + str(ID) for ID in tags_list.split(",")]

            created_at = "created_at_" + str(line[12].strip())

            item_token_list = [*title_id_list, career_level, *industry_id_list, *discipline_id_list, country, is_paid,
                               region, employment, *tags_list]

            ICM_builder.add_single_row(item_id, item_token_list, data=1.0)

            numCells += 1

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

    def get_filtered_ICM(self, categories, exceptions=None, ICM_path=None, if_new_item="add"):

        assert categories is not None and len(categories) > 0, "You should pass at least one category (in a list)."

        from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=None, on_new_row=if_new_item)

        if ICM_path is None:
            compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
            decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER
            zipFile_name = "xing_challenge_data_2017.zip"
            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)
            ICM_path = dataFile.extract("data/items.csv", path=decompressed_zip_file_folder + "decompressed/")

        fileHandle = open(ICM_path, "r")

        numCells = 0

        # Remove header
        # item_id	title	career_level	discipline_id	industry_id	country	is_payed	region	latitude	longitude	employment	tags	created_at
        fileHandle.readline()

        for line in fileHandle:

            if numCells % 1000000 == 0 and numCells != 0:
                print("Processed {} cells".format(numCells))

            line = line.split("\t")

            item_id = line[0]

            title_id_list = line[1]
            title_id_list = ["title_" + str(ID) for ID in title_id_list.split(",")]

            career_level = "career_level_" + str(line[2])

            discipline_id_list = line[3]
            discipline_id_list = ["discipline_" + str(ID) for ID in discipline_id_list.split(",")]

            industry_id_list = line[4]
            industry_id_list = ["industry_" + str(ID) for ID in industry_id_list.split(",")]

            country = "country_" + str(line[5])
            is_paid = "is_paid_" + str(line[6])

            region = "region_" + str(line[7])

            latitude = "latitude_" + str(line[8])
            longitude = "longitude_" + str(line[9])

            employment = "employment_" + str(line[10])

            tags_list = line[11]
            tags_list = ["tags_" + str(ID) for ID in tags_list.split(",")]

            created_at = "created_at_" + str(line[12].strip())

            item_token_list = []

            if "title" in categories:
                item_token_list.extend(title_id_list)
            if "career_level" in categories:
                item_token_list.append(career_level)
            if "discipline" in categories:
                item_token_list.extend(discipline_id_list)
            if "industry" in categories:
                item_token_list.extend(industry_id_list)
            if "country" in categories:
                item_token_list.append(country)
            if "is_paid" in categories:
                item_token_list.append(is_paid)
            if "region" in categories:
                item_token_list.append(region)
            if "employment" in categories:
                item_token_list.append(employment)
            if "tags" in categories:
                item_token_list.extend(tags_list)

            if exceptions is not None:
                for feature in exceptions:
                    if feature in item_token_list:
                        item_token_list.remove(feature)

            ICM_builder.add_single_row(item_id, item_token_list, data=1.0)

            numCells += 1

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


