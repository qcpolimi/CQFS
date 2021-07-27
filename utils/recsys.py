import numpy as np


def remove_ICM_item_interactions(ICM, items_to_remove):
    n_items, _ = ICM.shape
    max_item = max(items_to_remove)
    assert max_item < n_items, f"Cannot remove these items from the ICM: {max_item} is out of bounds ({n_items})."

    new_ICM = ICM.tocoo().copy()

    item_rows = new_ICM.row
    items_to_remove_mask = np.in1d(item_rows, items_to_remove)

    new_ICM.data[items_to_remove_mask] = 0
    new_ICM.eliminate_zeros()

    return new_ICM.tocsr()


def test_ICM_feature_selection(ICM, selection):
    ICM_csc_indptr = ICM.tocsc().indptr
    col_lens = np.ediff1d(ICM_csc_indptr)

    zeros = np.zeros_like(col_lens)
    inverted_selection = np.logical_not(selection)

    # Check correctness of the new ICM
    np.testing.assert_array_equal(zeros[inverted_selection], col_lens[inverted_selection])
    np.testing.assert_array_less(zeros[selection], col_lens[selection])
