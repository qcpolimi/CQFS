import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import issparse
from scipy.stats import skew

from recsys.Base.DataIO import DataIO


def compare_similarities(S1, S2):
    S1_bool = S1.astype(bool, copy=True)
    S2_bool = S2.astype(bool, copy=True)
    intersection = S1_bool.multiply(S2_bool)

    S1_nnz = S1.nnz
    S2_nnz = S2.nnz
    intersection_nnz = intersection.nnz

    return intersection_nnz, \
           0 if S1_nnz == 0 else intersection_nnz / S1_nnz, \
           0 if S2_nnz == 0 else intersection_nnz / S2_nnz


def similarity_statistics(S_CF, S_CBF, intersection=None, union=None, statistics=None):
    assert S_CF.shape == S_CBF.shape, "The shapes of the two similarity matrices do not correspond."

    if statistics is None:
        statistics = {}

    n_items = S_CF.shape[0]
    area = n_items ** 2

    statistics['n_items'] = n_items

    cf_nnz = S_CF.nnz if issparse(S_CF) else np.count_nonzero(S_CF)
    statistics['CF_nnz'] = cf_nnz
    statistics['CF_density'] = cf_nnz / area

    cbf_nnz = S_CBF.nnz if issparse(S_CBF) else np.count_nonzero(S_CBF)
    statistics['CBF_nnz'] = cbf_nnz
    statistics['CBF_density'] = cbf_nnz / area

    intersection_nnz = 0
    if intersection is not None:
        intersection_nnz = intersection.nnz if issparse(intersection) else np.count_nonzero(intersection)
        statistics['intersection_nnz'] = intersection_nnz
        statistics['intersection_CF_perc'] = intersection_nnz / cf_nnz
        statistics['intersection_CBF_perc'] = intersection_nnz / cbf_nnz

    if union is not None:
        union_nnz = union.nnz if issparse(union) else np.count_nonzero(union)
        statistics['union_nnz'] = union_nnz
        statistics['union_CF_perc'] = cf_nnz / union_nnz
        statistics['union_CBF_perc'] = cbf_nnz / union_nnz

        if intersection is not None:
            statistics['intersection_over_union'] = intersection_nnz / union_nnz

    return statistics


def warm_similarity_statistics(S_CF, S_CBF, CF_warm_items=None, CBF_items_with_interactions=None, statistics=None):
    assert S_CF.shape == S_CBF.shape, "The shapes of the two similarity matrices do not correspond."

    if statistics is None:
        statistics = {}

    n_items = S_CF.shape[0]

    # Warm CF items density
    if CF_warm_items is not None:
        warm_S_CF = S_CF[CF_warm_items]
        warm_S_CF = warm_S_CF[:, CF_warm_items]

        assert warm_S_CF.shape[0] == warm_S_CF.shape[1], "Warm CF similarity matrix is incorrect."

        n_warm_CF_items = CF_warm_items.sum()
        statistics['CF_n_warm_items'] = n_warm_CF_items
        statistics['CF_warm_items_perc'] = n_warm_CF_items / n_items

        warm_nnz = warm_S_CF.nnz
        warm_area = warm_S_CF.shape[0] ** 2
        statistics['CF_warm_density'] = 0 if warm_area == 0 else warm_nnz / warm_area

    # Warm CBF items density
    if CBF_items_with_interactions is not None:
        warm_S_CBF = S_CBF[CBF_items_with_interactions]
        warm_S_CBF = warm_S_CBF[:, CBF_items_with_interactions]

        assert warm_S_CBF.shape[0] == warm_S_CBF.shape[1], "Warm CBF similarity matrix is incorrect."

        n_warm_CBF_items = CBF_items_with_interactions.sum()
        statistics['CBF_n_items_with_features'] = n_warm_CBF_items
        statistics['CBF_items_with_features_perc'] = n_warm_CBF_items / n_items

        warm_nnz = warm_S_CBF.nnz
        warm_area = warm_S_CBF.shape[0] ** 2
        statistics['CBF_items_with_features_density'] = warm_nnz / warm_area

    return statistics


def results_similarity_statistics(CQFS_S_CBF, S_CF, S_CBF, intersection, CQFS_intersection, CQFS_union, statistics=None,
                                  items_with_features=None):
    assert CQFS_S_CBF.shape == S_CF.shape, "The similarity matrices shapes do not correspond."
    assert CQFS_S_CBF.shape == S_CBF.shape, "The similarity matrices shapes do not correspond."

    if statistics is None:
        statistics = {}

    n_items, _ = CQFS_S_CBF.shape
    area = n_items ** 2

    CQFS_cbf_nnz = CQFS_S_CBF.nnz
    statistics['CQFS_CBF_nnz'] = CQFS_cbf_nnz
    statistics['CQFS_CBF_density'] = CQFS_cbf_nnz / area

    if items_with_features is not None:
        CQFS_S_CBF_with_features = CQFS_S_CBF[items_with_features]
        CQFS_S_CBF_with_features = CQFS_S_CBF_with_features[:, items_with_features]

        assert CQFS_S_CBF_with_features.shape[0] == CQFS_S_CBF_with_features.shape[
            1], "Warm CBF similarity matrix is incorrect."

        CQFS_CBF_n_items_with_features = items_with_features.sum()
        statistics['CQFS_CBF_n_items_with_features'] = CQFS_CBF_n_items_with_features
        statistics['CQFS_CBF_items_with_features_perc'] = CQFS_CBF_n_items_with_features / n_items

        CQFS_with_features_nnz = CQFS_S_CBF_with_features.nnz
        CQFS_with_features_area = CQFS_S_CBF_with_features.shape[0] ** 2
        statistics['CQFS_CBF_items_with_features_density'] = \
            0 if CQFS_with_features_area == 0 else CQFS_with_features_nnz / CQFS_with_features_area

    cf_nnz = S_CF.nnz
    cbf_nnz = S_CBF.nnz
    CQFS_intersection_nnz = CQFS_intersection.nnz
    CQFS_union_nnz = CQFS_union.nnz

    statistics['CQFS_intersection_nnz'] = CQFS_intersection_nnz
    statistics['CQFS_intersection_CF_perc'] = CQFS_intersection_nnz / cf_nnz
    statistics['CQFS_intersection_CBF_perc'] = CQFS_intersection_nnz / cbf_nnz

    statistics['CQFS_against_intersection_nnz'], \
    statistics['CQFS_against_intersection_CQFS_perc'], \
    statistics['CQFS_against_intersection_intersection_perc'] = compare_similarities(CQFS_S_CBF, intersection)

    statistics['CQFS_against_CF_nnz'], \
    statistics['CQFS_against_CF_CQFS_perc'], \
    statistics['CQFS_against_CF_CF_perc'] = compare_similarities(CQFS_S_CBF, S_CF)

    statistics['CQFS_union_nnz'] = CQFS_union_nnz
    statistics['CQFS_union_CF_perc'] = cf_nnz / CQFS_union_nnz
    statistics['CQFS_union_CBF_perc'] = CQFS_cbf_nnz / CQFS_union_nnz
    statistics['CQFS_intersection_over_union'] = CQFS_intersection_nnz / CQFS_union_nnz

    statistics['CQFS_against_CBF_nnz'], \
    statistics['CQFS_against_CBF_CQFS_perc'], \
    statistics['CQFS_against_CBF_CBF_perc'] = compare_similarities(CQFS_S_CBF, S_CBF)

    return statistics


def BQM_statistics(linear, quadratic, shape, prefix="", statistics=None, distplot_folder_path=None):
    if statistics is None:
        statistics = {}

    n_linear = len(linear)
    n_quadratic = len(quadratic)
    area = shape[0] * shape[1]
    statistics[f'{prefix}sparsity'] = (n_linear + n_quadratic) / area

    statistics[f'{prefix}linear_n'] = n_linear
    n_pos_linear = (linear > 0).sum()
    statistics[f'{prefix}linear_n_pos'] = n_pos_linear
    statistics[f'{prefix}linear_pos_perc'] = n_pos_linear / n_linear
    n_neg_linear = (linear < 0).sum()
    statistics[f'{prefix}linear_n_neg'] = n_neg_linear
    statistics[f'{prefix}linear_neg_perc'] = n_neg_linear / n_linear

    statistics[f'{prefix}linear_min'] = np.float64(linear.min())
    statistics[f'{prefix}linear_max'] = np.float64(linear.max())
    statistics[f'{prefix}linear_range'] = np.float64(linear.ptp())
    statistics[f'{prefix}linear_mean'] = np.float64(linear.mean())
    statistics[f'{prefix}linear_std'] = np.float64(linear.std())
    statistics[f'{prefix}linear_var'] = np.float64(linear.var())
    statistics[f'{prefix}linear_median'] = np.float64(np.median(linear))
    statistics[f'{prefix}linear_skewness'] = skew(linear)

    quadratic_area = area - n_linear
    statistics[f'{prefix}quadratic_n'] = n_quadratic
    n_pos_quadratic = (quadratic > 0).sum()
    statistics[f'{prefix}quadratic_n_pos'] = n_pos_quadratic
    statistics[f'{prefix}quadratic_pos_perc'] = n_pos_quadratic / n_quadratic
    statistics[f'{prefix}quadratic_pos_perc_on_tot'] = n_pos_quadratic / quadratic_area
    n_neg_quadratic = (quadratic < 0).sum()
    statistics[f'{prefix}quadratic_n_neg'] = n_neg_quadratic
    statistics[f'{prefix}quadratic_neg_perc'] = n_neg_quadratic / n_quadratic
    statistics[f'{prefix}quadratic_neg_perc_on_tot'] = n_neg_quadratic / quadratic_area

    statistics[f'{prefix}quadratic_min'] = np.float64(quadratic.min())
    statistics[f'{prefix}quadratic_max'] = np.float64(quadratic.max())
    statistics[f'{prefix}quadratic_range'] = np.float64(quadratic.ptp())
    statistics[f'{prefix}quadratic_mean'] = np.float64(quadratic.mean())
    statistics[f'{prefix}quadratic_std'] = np.float64(quadratic.std())
    statistics[f'{prefix}quadratic_var'] = np.float64(quadratic.var())
    statistics[f'{prefix}quadratic_median'] = np.float64(np.median(quadratic))
    statistics[f'{prefix}quadratic_skewness'] = skew(quadratic)

    if distplot_folder_path != None:
        print("Saving distribution plot...")
        plt.clf()

        dataIO = DataIO(distplot_folder_path)
        np_dict = {'linear': linear, 'quadratic': quadratic}
        dataIO.save_data('distribution', np_dict)

        sns.kdeplot(linear, shade=True)
        linear_path = f"{distplot_folder_path}/linear"
        plt.savefig(linear_path)
        plt.clf()

        sns.kdeplot(quadratic, shade=True)
        quadratic_path = f"{distplot_folder_path}/quadratic"
        plt.savefig(quadratic_path)
        plt.clf()

    return statistics


def similarity_RMSE(S_CF, S_CBF, N=None):
    assert S_CF.shape == S_CBF.shape, "The shapes of the two similarity matrices do not correspond."

    # n_items = S_CF.shape[0]
    # N = n_items * n_items

    if N is None:
        S_union = S_CF.astype(bool) + S_CBF.astype(bool)
        N = S_union.nnz

    S_RMSE = S_CF - S_CBF
    S_RMSE = S_RMSE.power(2)
    S_RMSE = S_RMSE.sum() / N

    return np.sqrt(S_RMSE)


def error_statistics(S_CF, S_CBF, N=None, suffix="", statistics=None):
    assert S_CF.shape == S_CBF.shape, "The shapes of the two similarity matrices do not correspond."

    if statistics is None:
        statistics = {}

    statistics[f'RMSE{suffix}'] = similarity_RMSE(S_CF, S_CBF, N)

    return statistics
