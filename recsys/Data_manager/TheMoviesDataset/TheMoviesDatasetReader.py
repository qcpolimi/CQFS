#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import zipfile
import numpy as np
import pandas as pd
import ast, csv, os, shutil, time
from recsys.Data_manager.Dataset import Dataset
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from recsys.Data_manager.DataReader_utils import remove_features, reconcile_mapper_with_removed_tokens, invert_dictionary, merge_ICM
from recsys.Data_manager.DataPostprocessing_K_Cores import select_k_cores
from recsys.Base.Recommender_utils import reshapeSparse
from recsys.Data_manager.Movielens.Movielens20MReader import _loadURM_preinitialized_item_id


class TheMoviesDatasetReader(DataReader):

    #DATASET_URL = "https://www.kaggle.com/rounakbanik/the-movies-dataset"
    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EQAIIMiVSTpIjZYmDqMyvukB8dur9LJ5cRT83CzXpLZ0TQ?e=lRNtWF"
    DATASET_SUBFOLDER = "TheMoviesDataset/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_all", "ICM_credits", "ICM_metadata"]
    DATASET_SPECIFIC_MAPPER = ["item_original_ID_to_title", "item_index_to_title"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "the-movies-dataset.zip"


        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            credits_path = dataFile.extract("credits.csv", path=decompressed_zip_file_folder + "decompressed/")
            metadata_path = dataFile.extract("movies_metadata.csv", path=decompressed_zip_file_folder + "decompressed/")
            movielens_tmdb_id_map_path = dataFile.extract("links.csv", path=decompressed_zip_file_folder + "decompressed/")

            URM_path = dataFile.extract("ratings.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            self._print("Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")




        self.item_original_ID_to_title = {}
        self.item_index_to_title = {}

        self._print("Loading ICM_credits")
        ICM_credits, tokenToFeatureMapper_ICM_credits, self.item_original_ID_to_index = self._loadICM_credits(credits_path, header=True, if_new_item = "add")

        self._print("Loading ICM_metadata")
        ICM_metadata, tokenToFeatureMapper_ICM_metadata, self.item_original_ID_to_index = self._loadICM_metadata(metadata_path, header=True, if_new_item = "add")



        # ICM_credits, _, tokenToFeatureMapper_ICM_credits = remove_features(ICM_credits, min_occurrence= 5, max_percentage_occurrence= 0.30,
        #                                                                    reconcile_mapper = tokenToFeatureMapper_ICM_credits)
        #
        # ICM_metadata, _, tokenToFeatureMapper_ICM_metadata = remove_features(ICM_metadata, min_occurrence= 5, max_percentage_occurrence= 0.30,
        #                                                                      reconcile_mapper = tokenToFeatureMapper_ICM_metadata)

        n_items = ICM_metadata.shape[0]

        ICM_credits = reshapeSparse(ICM_credits, (n_items, ICM_credits.shape[1]))




        # IMPORTANT: ICM uses TMDB indices, URM uses movielens indices
        # Load index mapper
        movielens_id_to_tmdb, tmdb_to_movielens_id = self._load_item_id_mappping(movielens_tmdb_id_map_path, header=True)

        # Modify saved mapper to accept movielens id instead of tmdb
        self._replace_tmdb_id_with_movielens(tmdb_to_movielens_id)


        self._print("Loading URM")
        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index, URM_timestamp = _loadURM_preinitialized_item_id(URM_path, separator=",",
                                                                                          header = True, if_new_user = "add", if_new_item = "ignore",
                                                                                          item_original_ID_to_index = self.item_original_ID_to_index)

        # Reconcile URM and ICM
        # Keep only items having ICM entries, remove all the others
        self.n_items = ICM_credits.shape[0]

        URM_all, removedUsers, removedItems = select_k_cores(URM_all, k_value = 1, reshape=True)
        URM_timestamp, _, _ = select_k_cores(URM_timestamp, k_value = 1, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        # Remove movie_ID discarded in previous step
        item_original_ID_to_title_old = self.item_original_ID_to_title.copy()

        for item_id in item_original_ID_to_title_old:

            if item_id not in self.item_original_ID_to_index:
                del self.item_original_ID_to_title[item_id]


        removed_item_mask = np.zeros(self.n_items, dtype=np.bool)
        removed_item_mask[removedItems] = True

        to_preserve_item_mask = np.logical_not(removed_item_mask)

        ICM_credits = ICM_credits[to_preserve_item_mask,:]
        ICM_metadata = ICM_metadata[to_preserve_item_mask,:]
        # URM is already clean

        self.n_items = ICM_credits.shape[0]


        ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(ICM_credits, ICM_metadata,
                                                          tokenToFeatureMapper_ICM_credits,
                                                          tokenToFeatureMapper_ICM_metadata)



        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_timestamp": URM_timestamp}

        loaded_ICM_dict = {"ICM_credits": ICM_credits,
                           "ICM_metadata": ICM_metadata,
                           "ICM_all": ICM_all,
                           }

        self.loaded_ICM_mapper_dict = {"ICM_credits": tokenToFeatureMapper_ICM_credits,
                                  "ICM_metadata": tokenToFeatureMapper_ICM_metadata,
                                  "ICM_all": tokenToFeatureMapper_ICM_all,
                                  }


        additional_data_mapper = {"item_original_ID_to_title": self.item_original_ID_to_title,
                                  "item_index_to_title": self.item_original_ID_to_title}



        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary = self.loaded_ICM_mapper_dict,
                                 UCM_dictionary = None,
                                 UCM_feature_mapper_dictionary = None,
                                 user_original_ID_to_index= self.user_original_ID_to_index,
                                 item_original_ID_to_index= self.item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 additional_data_mapper = additional_data_mapper,
                                 )



        self._print("cleaning temporary files")

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset



    def _load_item_id_mappping(self, movielens_tmdb_id_map_path, header=True):

        movielens_id_to_tmdb = {}
        tmdb_to_movielens_id = {}

        movielens_tmdb_id_map_file = open(movielens_tmdb_id_map_path, 'r', encoding="utf8")

        if header:
            movielens_tmdb_id_map_file.readline()


        for newMapping in movielens_tmdb_id_map_file:

            newMapping = newMapping.split(",")

            movielens_id = newMapping[0]
            tmdb_id = newMapping[2].replace("\n", "")

            movielens_id_to_tmdb[movielens_id] = tmdb_id
            tmdb_to_movielens_id[tmdb_id] = movielens_id


        return movielens_id_to_tmdb, tmdb_to_movielens_id



    def _replace_tmdb_id_with_movielens(self, tmdb_to_movielens_id):
        """
        Replace 'the original id' in such a way that it points to the same index
        :param tmdb_to_movielens_id:
        :return:
        """

        item_original_ID_to_index_movielens = {}
        item_index_to_original_ID_movielens = {}
        item_original_ID_to_title_movielens = {}

        # self.item_original_ID_to_index[item_id] = itemIndex
        # self.item_index_to_original_ID[itemIndex] = item_id

        self.item_index_to_original_ID = invert_dictionary(self.item_original_ID_to_index)

        for item_index in self.item_index_to_original_ID.keys():

            tmdb_id = self.item_index_to_original_ID[item_index]

            if tmdb_id in self.item_original_ID_to_title:
                movie_title = self.item_original_ID_to_title[tmdb_id]
            else:
                movie_title = ""

            movielens_id = tmdb_to_movielens_id[tmdb_id]

            item_index_to_original_ID_movielens[item_index] = movielens_id
            item_original_ID_to_index_movielens[movielens_id] = item_index
            item_original_ID_to_title_movielens[movielens_id] = movie_title


        # Replace the TMDB based mapper
        self.item_original_ID_to_index = item_original_ID_to_index_movielens
        self.item_index_to_original_ID = item_index_to_original_ID_movielens
        self.item_original_ID_to_title = item_original_ID_to_title_movielens







    def _loadICM_credits(self, credits_path, header=True, if_new_item = "add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = if_new_item)




        #parser_credits = parse_json(credits_path, header = header)
        numCells = 0

        credits_file = open(credits_path, 'r', encoding="utf8")

        if header:
            credits_file.readline()

        parser_credits = csv.reader(credits_file, delimiter=',', quotechar='"')


        for newCredits in parser_credits:

            # newCredits is a tuple of two strings, both are lists of dictionaries
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # NOTE: sometimes a dict value is ""Savannah 'Vannah' Jackson"", if the previous eval removes the commas "" "" then the parsing of the string will fail
            cast_list = []
            credits_list = []

            try:
                cast_list = ast.literal_eval(newCredits[0])
                credits_list = ast.literal_eval(newCredits[1])
            except Exception as e:
                print("TheMoviesDatasetReader: Exception while parsing: '{}', skipping".format(str(e)))


            movie_id = newCredits[2]

            cast_list.extend(credits_list)

            cast_list_name = [cast_member["name"] for cast_member in cast_list]

            ICM_builder.add_single_row(movie_id, cast_list_name, data=1.0)



        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()





    def _new_loadICM_credits(self, credits_path, header=True, if_new_item = "add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = if_new_item)




        #parser_credits = parse_json(credits_path, header = header)
        numCells = 0

        credits_file = open(credits_path, 'r', encoding="utf8")

        if header:
            credits_file.readline()

        parser_credits = csv.reader(credits_file, delimiter=',', quotechar='"')

        cast_dict = {}
        crew_dict = {}

        for credit in parser_credits:
            movie_id = credit[2]
            cast_dict[movie_id] = ast.literal_eval(credit[0])
            crew_dict[movie_id] = ast.literal_eval(credit[1])

        tot_cast = 0
        cast_list = []
        for k, cl in cast_dict.items():
            for c in cl:
                c.update({'movie_id': k, 'job': 'Actor'})
                cast_list.append(c)
                tot_cast += 1

        assert len(cast_list) == tot_cast

        tot_crew = 0
        crew_list = []
        for k, cl in crew_dict.items():
            for c in cl:
                c.update({'movie_id': k})
                crew_list.append(c)
                tot_crew += 1

        assert len(crew_list) == tot_crew

        original_cast_df = pd.DataFrame(cast_list)
        original_crew_df = pd.DataFrame(crew_list)
        original_cast_crew_df = pd.concat([original_cast_df, original_crew_df], sort=False, ignore_index=True)
        cast_crew_df = original_cast_crew_df[['name', 'job', 'movie_id']]

        name_job_group = cast_crew_df.groupby(['name', 'job'])
        name_job_count_df = name_job_group.count()
        name_job_cast_crew_df = cast_crew_df.set_index(['name', 'job'])
        more_than_5_items_df = name_job_cast_crew_df[name_job_count_df['movie_id'] > 5].reset_index()

        directors = more_than_5_items_df[more_than_5_items_df['job'] == 'Director']
        s_time = time.time()
        for d in range(directors.shape[0]):
            director = directors.iloc[d]
            row_list = [int(director['movie_id'])]
            col_list = [director['name']]
            ICM_builder.add_data_lists(row_list, col_list, data_list_to_add=[1.0])
        self._print(f"Building the ICM took {time.time() - s_time} seconds.")

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


    def get_filtered_ICM_credits(self, categories=None, k=0, credits_path=None, header=True, if_new_item="add", item_mapper=None):

        # assert categories is not None and len(categories) > 0, "You should pass at least one category (in a list)."

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER
        zipFile_name = "the-movies-dataset.zip"
        dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

        if credits_path is None:
            credits_path = dataFile.extract("credits.csv", path=decompressed_zip_file_folder + "decompressed/")

        credits_file = open(credits_path, 'r', encoding="utf8")

        if header:
            credits_file.readline()

        parser_credits = csv.reader(credits_file, delimiter=',', quotechar='"')

        cast_dict = {}
        crew_dict = {}

        for credit in parser_credits:
            movie_id = credit[2]
            cast_dict[movie_id] = ast.literal_eval(credit[0])
            crew_dict[movie_id] = ast.literal_eval(credit[1])

        tot_cast = 0
        cast_list = []
        for key, cl in cast_dict.items():
            for c in cl:
                c.update({'movie_id': key, 'job': 'Actor'})
                cast_list.append(c)
                tot_cast += 1

        assert len(cast_list) == tot_cast

        tot_crew = 0
        crew_list = []
        for key, cl in crew_dict.items():
            for c in cl:
                c.update({'movie_id': key})
                crew_list.append(c)
                tot_crew += 1

        assert len(crew_list) == tot_crew

        original_cast_df = pd.DataFrame(cast_list)
        original_crew_df = pd.DataFrame(crew_list)
        original_cast_crew_df = pd.concat([original_cast_df, original_crew_df], sort=False, ignore_index=True)
        cast_crew_df = original_cast_crew_df[['name', 'job', 'movie_id']]

        name_job_group = cast_crew_df.groupby(['name', 'job'])
        name_job_count_df = name_job_group.count()
        name_job_cast_crew_df = cast_crew_df.set_index(['name', 'job'])

        if k > 0:
            name_job_cast_crew_df = name_job_cast_crew_df[name_job_count_df['movie_id'] >= k].reset_index()

        if categories is not None and len(categories) > 0:
            name_job_cast_crew_df = name_job_cast_crew_df[name_job_cast_crew_df['job'].isin(categories)]

        # IMPORTANT: ICM uses TMDB indices, URM uses movielens indices
        # Load index mapper
        movielens_tmdb_id_map_path = dataFile.extract("links.csv", path=decompressed_zip_file_folder + "decompressed/")
        movielens_id_to_tmdb, tmdb_to_movielens_id = self._load_item_id_mappping(movielens_tmdb_id_map_path,
                                                                                 header=True)

        s_time = time.time()
        for d in range(name_job_cast_crew_df.shape[0]):
            item = name_job_cast_crew_df.iloc[d]
            movie_id = tmdb_to_movielens_id[item['movie_id']]
            row_list = [movie_id]
            col_list = [item['name']]
            ICM_builder.add_data_lists(row_list, col_list, data_list_to_add=[1.0])
        self._print(f"Building the ICM took {time.time() - s_time} seconds.")

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()



    def _loadICM_metadata(self, metadata_path, header=True, if_new_item = "add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = self.item_original_ID_to_index, on_new_row = if_new_item)


        numCells = 0

        metadata_file = open(metadata_path, 'r', encoding="utf8")

        if header:
            metadata_file.readline()

        parser_metadata = csv.reader(metadata_file, delimiter=',', quotechar='"')


        for newMetadata in parser_metadata:

            numCells += 1
            if numCells % 100000 == 0:
                print("Processed {} cells".format(numCells))

            token_list = []

            if len(newMetadata) < 22:
                #Sono 6, ragionevole
                print("TheMoviesDatasetReader: Line too short, possible unwanted new line character, skipping")
                continue

            movie_id = newMetadata[5]


            if newMetadata[0] == "True":
                token_list.append("ADULTS_YES")
            else:
                token_list.append("ADULTS_NO")

            if newMetadata[1]:
                collection = ast.literal_eval(newMetadata[1])
                token_list.append("collection_" + str(collection["id"]))

            #budget = int(rating[2])

            if newMetadata[3]:
                genres = ast.literal_eval(newMetadata[3])

                for genre in genres:
                    token_list.append("genre_" + str(genre["id"]))


            orig_lang = newMetadata[7]
            title = newMetadata[8]

            if movie_id not in self.item_original_ID_to_title:
                self.item_original_ID_to_title[movie_id] = title

            if orig_lang:
                token_list.append("original_language_"+orig_lang)

            if newMetadata[12]:
                prod_companies = ast.literal_eval(newMetadata[12])
                for prod_company in prod_companies:
                    token_list.append("production_company_" + str(prod_company['id']))


            if newMetadata[13]:
                prod_countries = ast.literal_eval(newMetadata[13])
                for prod_country in prod_countries:
                    token_list.append("production_country_" + prod_country['iso_3166_1'])


            try:
                release_date = int(newMetadata[14].split("-")[0])
                token_list.append("release_date_" + str(release_date))
            except Exception:
                pass


            if newMetadata[17]:
                spoken_langs = ast.literal_eval(newMetadata[17])
                for spoken_lang in spoken_langs:
                    token_list.append("spoken_lang_" + spoken_lang['iso_639_1'])


            if newMetadata[18]:
                status = newMetadata[18]
                if status:
                    token_list.append("status_" + status)

            if newMetadata[21] == "True":
                token_list.append("VIDEO_YES")
            else:
                token_list.append("VIDEO_NO")


            ICM_builder.add_single_row(movie_id, token_list, data=True)




        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

