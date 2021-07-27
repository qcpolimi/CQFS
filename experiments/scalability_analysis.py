import time

import dimod
import numpy as np
from neal import SimulatedAnnealingSampler
from recsys.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython

from core.CQFSSampler import CQFSEmbedding, CQFSSampler
from data.DataLoader import XingChallenge2017Loader
from recsys.Base.DataIO import DataIO
from recsys.Data_manager import TheMoviesDatasetReader
from recsys.Data_manager.DataSplitter_Cold_items import DataSplitter_Cold_items
from recsys.Recommender_import_list import ItemKNNCFRecommender
from utils.recsys import remove_ICM_item_interactions
from utils.samplers import get_hybrid_from_topology
from utils.sparse import merge_sparse_matrices

TMD_ALPHA = 1
TMD_BETA = 0.0001
TMD_STRENGTH = 100
TMD_PERCENTAGE = 0.6

XING_ALPHA = 1
XING_BETA = 0.001
XING_STRENGTH = 100
XING_PERCENTAGE = 0.8


def main():
    analysis_path = f"../"
    dataIO = DataIO(analysis_path)

    ##################################################
    # Analyse experiment scalability

    scalability_percentages = [1, 0.5, 0.25, 0.125]
    scalability_analysis_tmd = {}
    scalability_analysis_xing = {}

    #########################
    # Scale on items (TMD)
    # scalability_analysis_tmd['items'] = scalability_analysis_on_items_tmd(scalability_percentages)

    #########################
    # Scale on features (TMD)
    # scalability_analysis_tmd['features'] = scalability_analysis_on_features_tmd(scalability_percentages)

    #########################
    # Scale on items (Xing)
    # scalability_analysis_xing['items'] = scalability_analysis_on_items_xing(scalability_percentages)
    #
    # try:
    #     dataIO.save_data('scalability_analysis_xing', scalability_analysis_xing)
    # except:
    #     print(f"Cannot save the entire dictionary.")
    #     for k in scalability_analysis_xing:
    #         try:
    #             dataIO.save_data(f'scalability_analysis_xing_{k}', {k: scalability_analysis_xing[k]})
    #         except:
    #             print(f"Cannot save {k}.")

    #########################
    # Scale on features (Xing)
    scalability_analysis_xing['features'] = scalability_analysis_on_features_xing(scalability_percentages)

    #########################
    # Save analysis

    try:
        dataIO.save_data('scalability_analysis_xing', scalability_analysis_xing)
    except:
        print(f"Cannot save the entire dictionary.")
        for k in scalability_analysis_xing:
            try:
                dataIO.save_data(f'scalability_analysis_xing_{k}', {k: scalability_analysis_xing[k]})
            except:
                print(f"Cannot save {k}.")

    # dataIO.save_data('scalability_analysis_xing', scalability_analysis_xing)


def scalability_analysis_on_items_tmd(item_percentages):
    np.random.seed(56)

    print("Starting scalability analysis on items for The Movies Dataset.")

    ##################################################
    # Load essential data

    dataset_name, URM_train_last_test, ICM_train = load_essential_data_tmd()

    return scalability_analysis_on_items(item_percentages, dataset_name, URM_train_last_test, ICM_train, TMD_ALPHA,
                                         TMD_BETA, TMD_STRENGTH, TMD_PERCENTAGE, n_samples=10)


def scalability_analysis_on_features_tmd(feature_percentages):
    np.random.seed(56)

    print("Starting scalability analysis on features for The Movies Dataset.")

    ##################################################
    # Load essential data

    dataset_name, URM_train_last_test, ICM_train = load_essential_data_tmd()

    return scalability_analysis_on_features(feature_percentages, dataset_name, URM_train_last_test, ICM_train,
                                            TMD_ALPHA, TMD_BETA, TMD_STRENGTH, TMD_PERCENTAGE, n_samples=10)


def scalability_analysis_on_items_xing(item_percentages):
    np.random.seed(56)

    print("Starting scalability analysis on items for Xing Challenge 2017.")

    ##################################################
    # Load essential data

    dataset_name, URM_train_last_test, ICM_train = load_essential_data_xing()

    return scalability_analysis_on_items(item_percentages, dataset_name, URM_train_last_test, ICM_train, XING_ALPHA,
                                         XING_BETA, XING_STRENGTH, XING_PERCENTAGE, n_samples=10, hybrid=False)


def scalability_analysis_on_features_xing(feature_percentages):
    np.random.seed(56)

    print("Starting scalability analysis on features for Xing Challenge 2017.")

    ##################################################
    # Load essential data

    dataset_name, URM_train_last_test, ICM_train = load_essential_data_xing()

    return scalability_analysis_on_features(feature_percentages, dataset_name, URM_train_last_test, ICM_train,
                                            XING_ALPHA, XING_BETA, XING_STRENGTH, XING_PERCENTAGE, n_samples=10,
                                            hybrid=False)


def scalability_analysis_on_items(item_percentages, dataset_name, URM_train_last_test, ICM_train, alpha, beta,
                                  combination_strength, percentage, vartype='BINARY', n_samples=1, hybrid=True,
                                  topology='pegasus'):
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    ##################################################
    # Select items

    items = np.arange(n_items)
    np.random.shuffle(items)
    item_mapper = {i: items[i] for i in range(n_items)}

    timings = {
        'metadata': {
            'items': items,
            'item_mapper': item_mapper,
        },
    }

    for p in item_percentages:
        print(f"Percentage: {p * 100:.2f}.")

        k_items = int(p * n_items)
        k_indices = items[:k_items]

        ##################################################
        # Compute new data structures

        new_URM_train_last_test = URM_train_last_test[:, k_indices]
        new_ICM_train = ICM_train[k_indices]

        ##################################################
        # Scalability analysis

        timings[str(p)] = build_CQFS_model(dataset_name, new_ICM_train, new_URM_train_last_test, alpha, beta,
                                           combination_strength, percentage, vartype=vartype, n_samples=n_samples,
                                           hybrid=hybrid, topology=topology)

    return timings


def scalability_analysis_on_features(feature_percentages, dataset_name, URM_train_last_test, ICM_train, alpha, beta,
                                     combination_strength, percentage, vartype='BINARY', n_samples=1, hybrid=True,
                                     topology='pegasus'):
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    ##################################################
    # Select features

    features = np.arange(n_features)
    np.random.shuffle(features)
    feature_mapper = {i: features[i] for i in range(n_features)}

    timings = {
        'metadata': {
            'features': features,
            'feature_mapper': feature_mapper,
        },
    }

    for p in feature_percentages:
        print(f"Percentage: {p * 100:.2f}.")

        k_features = int(p * n_features)
        k_indices = features[:k_features]

        ##################################################
        # Compute new data structures

        new_ICM_train = ICM_train[:, k_indices]

        ##################################################
        # Scalability analysis

        timings[str(p)] = build_CQFS_model(dataset_name, new_ICM_train, URM_train_last_test, alpha, beta,
                                           combination_strength, percentage, vartype=vartype, n_samples=n_samples,
                                           hybrid=hybrid, topology=topology)

    return timings


def load_essential_data_tmd():
    ##################################################
    # Data loading and splitting

    # Instantiate the DataReader and load the data
    data_reader = TheMoviesDatasetReader()
    dataset = data_reader.load_data()
    dataset_name = dataset.get_dataset_name()

    # Split the data into train, validation and test (70, 10, 20) through the DataSplitter
    # The test and validation splits are cold items splits
    data_splitter = DataSplitter_Cold_items(data_reader)
    data_splitter.load_data()

    # Get the cold split
    URM_train, URM_validation, URM_test = data_splitter.get_cold_split()

    # Create the last test URM by merging the train and validation matrices
    URM_train_last_test = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    # Get the original ICM
    ICM_name = 'ICM_metadata'
    unfiltered_ICM_train = data_splitter.get_ICM_from_name(ICM_name)

    # Filter the ICM removing all the features with less than 5 interactions
    features_with_more_than_5_interactions = np.ediff1d(unfiltered_ICM_train.tocsc().indptr) >= 5
    original_ICM_train = unfiltered_ICM_train[:, features_with_more_than_5_interactions]

    # Remove cold items interactions from the ICM
    test_items = np.ediff1d(URM_test.tocsc().indptr) != 0
    test_items = np.arange(len(test_items))[test_items]
    ICM_train = remove_ICM_item_interactions(original_ICM_train, test_items)

    # Check if the removal was correct
    no_ICM_interaction_item_mask = np.ediff1d(ICM_train.indptr) == 0
    assert np.alltrue(
        no_ICM_interaction_item_mask[test_items]), "Test items were not correctly removed from the train ICM."

    return dataset_name, URM_train_last_test, ICM_train


def load_essential_data_xing():
    ##################################################
    # Data loading and splitting

    # Load data
    data_loader = XingChallenge2017Loader()
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the cold split
    URM_train, URM_validation, URM_test = data_loader.get_cold_split()

    # Create the last test URM by merging the train and validation matrices
    URM_train_last_test = merge_sparse_matrices(URM_train, URM_validation).tocsr()

    ##################################################
    # ICM preparation

    ICM_name = 'ICM_all'
    ICM_train, original_ICM_train = data_loader.get_ICM_train_from_name(ICM_name, return_original=True)
    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    return dataset_name, URM_train_last_test, ICM_train


def build_CQFS_model(dataset_name, ICM_train, URM_train_last_test, alpha, beta, combination_strength, percentage,
                     vartype='BINARY', n_samples=1, hybrid=True, topology='pegasus'):
    ##################################################
    # Content-based structure computation

    topK, _ = ICM_train.shape
    CBF_Similarity = Compute_Similarity_Cython(ICM_train.T, topK=topK, shrink=0, normalize=False,
                                               similarity='cosine')
    S_CBF = CBF_Similarity.compute_similarity()

    ##################################################
    # Collaborative filtering model training

    # Name of the experiment and output results folder path
    cf_recommender_name = ItemKNNCFRecommender.RECOMMENDER_NAME
    cf_path = f"../results/{dataset_name}/{cf_recommender_name}/"
    cf_dataIO = DataIO(cf_path)

    cf_metadata = f"{cf_recommender_name}_cosine_metadata.zip"
    cf_dict = cf_dataIO.load_data(cf_metadata)
    cf_best_hyperparameters = cf_dict['hyperparameters_best']

    cf_recommender = ItemKNNCFRecommender(URM_train_last_test)
    cf_recommender.fit(**cf_best_hyperparameters)

    S_CF = cf_recommender.W_sparse

    ##################################################
    # Check similarities

    n_items, n_features = ICM_train.shape
    print(f"Training ICM has {n_items} items and {n_features} features.")

    assert S_CF.shape == S_CBF.shape, "The two sparse matrices have different shapes!"
    assert S_CF.shape == (n_items, n_items), "The similarity matrices do not have the right shape."

    ##################################################
    # CQFS model building

    # Model parameters
    k = int(percentage * n_features)

    print("Starting preprocessing...")

    # Start measuring total preprocessing time
    preprocessing_time = time.time()

    #########################
    # Prepare similarities

    S_CBF_ones = S_CBF.copy()
    S_CBF_ones.data = np.ones_like(S_CBF.data)

    S_CF_bool = S_CF.astype(np.bool)
    S_CBF_bool = S_CBF.astype(np.bool, copy=True)

    #########################
    # Compute the bonus for similarities in common

    K_time = time.time()
    K = S_CBF_ones.multiply(S_CF_bool)
    K.data = -K.data
    K_time = time.time() - K_time

    #########################
    # Compute the penalization for similarities not in common

    E_time = time.time()
    S_intersection = S_CBF_bool.multiply(S_CF_bool)
    E = S_CBF_bool - S_intersection
    E = S_CBF_ones.multiply(E)
    E_time = time.time() - E_time

    assert K.nnz + E.nnz == S_CBF.nnz, "The number of items to keep and to penalize is not correct."

    #########################
    # Compute the QUBO model

    # Item Penalization Matrix
    # QUBO_time = time.time()
    IPM_time = time.time()
    IPM = alpha * K + beta * E
    IPM.eliminate_zeros()
    IPM_time = time.time() - IPM_time

    # Feature Penalization Matrix
    FPM_time = time.time()
    IFPM = IPM * ICM_train
    IFPM.eliminate_zeros()
    FPM = ICM_train.T * IFPM
    FPM.eliminate_zeros()
    FPM_time = time.time() - FPM_time

    BQM_conversion_time = time.time()
    BQM = dimod.as_bqm(FPM.toarray(), vartype)
    BQM_conversion_time = time.time() - BQM_conversion_time
    # QUBO_time = time.time() - QUBO_time

    # Add combinations constraint
    BQM_time = time.time()
    BQM_k = dimod.generators.combinations(n_features, k, strength=combination_strength, vartype=vartype)
    BQM_k = dimod.AdjVectorBQM(BQM_k)
    BQM_k.update(BQM)
    BQM_time = time.time() - BQM_time

    # BQM normalization
    BQM_normalize_time = time.time()
    BQM_k.normalize()
    BQM_normalize_time = time.time() - BQM_normalize_time

    preprocessing_time = time.time() - preprocessing_time

    # ##################################################
    # # Compute statistics
    #
    # S_union = S_CF_bool + S_CBF_bool
    #
    # statistics = similarity_statistics(S_CF, S_CBF, S_intersection, S_union)
    # statistics = error_statistics(S_CF, S_CBF, N=S_union.nnz, suffix="_dot", statistics=statistics)

    num_reads = 1
    if hybrid:
        ##################################################
        # Problem sampling with hybrid approach

        print("Sampling with hybrid approach.")

        solver = get_hybrid_from_topology(topology)

    else:
        ##################################################
        # Problem sampling with QPU

        print("Sampling with QPU only.")

        embedder = CQFSEmbedding(qpu_topology=topology, n_embedding_cases=5)
        embedding = embedder.get_embedding(BQM_k)

        solver = CQFSSampler(embedding, qpu_topology=topology)
        num_reads = 100

    properties = solver.properties

    # Setup structures
    responses = []
    response_times = []
    sampling_times = []
    network_times = []

    sa_responses = []
    sa_response_times = []

    for i in range(n_samples):
        # Measure total sampling time
        response_time = time.time()
        response = solver.sample(BQM_k)
        response_time = time.time() - response_time

        sampling_time = __get_sampling_time_from_response(response, hybrid) * pow(10, -6)
        network_time = response_time - sampling_time

        responses.append(response.to_serializable())
        response_times.append(response_time)
        sampling_times.append(sampling_time)
        network_times.append(network_time)

        time.sleep(60)

    ##################################################
    # Problem sampling with simulated annealing

    print("Sampling with simulated annealing.")

    sa_solver = SimulatedAnnealingSampler()

    for i in range(n_samples):
        # Measure total sampling time
        sa_response_time = time.time()
        sa_response = sa_solver.sample(BQM_k, num_reads=num_reads)
        sa_response_time = time.time() - sa_response_time

        sa_responses.append(sa_response.to_serializable())
        sa_response_times.append(sa_response_time)

    ##################################################
    # Return timings

    timings = {
        'preprocessing_time': preprocessing_time,
        'K_time': K_time,
        'E_time': E_time,

        'IPM_time': IPM_time,
        'FPM_time': FPM_time,
        'BQM_conversion_time': BQM_conversion_time,
        'BQM_time': BQM_time,
        'BQM_normalize_time': BQM_normalize_time,

        'response_time': response_times,
        'sampling_time': sampling_times,
        'network_time': network_times,

        'sa_response_time': sa_response_times,

        'responses': responses,
        'sa_responses': sa_responses,

        'properties': properties,
    }

    return timings


def __get_sampling_time_from_response(response, hybrid):
    if hybrid:
        return response.info['run_time']
    return response.info['timing']['qpu_access_time']


if __name__ == '__main__':
    main()
    # data = load_essential_data_xing()
