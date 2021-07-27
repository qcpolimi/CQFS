import numpy as np
import scipy.sparse as sps

from recsys.Base.Recommender_utils import reshapeSparse


def merge_sparse_matrices(matrix_a, matrix_b):

    assert matrix_a.shape == matrix_b.shape, "The two matrices have different shape, they should not be merged."

    matrix_a = matrix_a.tocoo()
    matrix_b = matrix_b.tocoo()

    data_a = matrix_a.data
    row_a = matrix_a.row
    col_a = matrix_a.col

    data_b = matrix_b.data
    row_b = matrix_b.row
    col_b = matrix_b.col

    data = np.concatenate((data_a, data_b))
    row = np.concatenate((row_a, row_b))
    col = np.concatenate((col_a, col_b))

    matrix = sps.coo_matrix((data, (row, col)))

    n_users = max(matrix_a.shape[0], matrix_b.shape[0])
    n_items = max(matrix_a.shape[1], matrix_b.shape[1])
    new_shape = (n_users, n_items)

    matrix = reshapeSparse(matrix, new_shape)

    return matrix


def select_columns(matrix, selection):

    m = matrix.tocoo()

    if selection.dtype == bool:
        indices = np.arange(len(selection))
        selection = indices[selection]

    for i in range(m.nnz):
        if m.col[i] not in selection:
            m.data[i] = 0

    m.eliminate_zeros()

    return m.tocsr()
