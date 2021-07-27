def get_experiment_id(alpha, beta, p=None, combination_strength=None):
    expID = f'a{alpha}b{beta}'

    if combination_strength is not None:
        expID = f'{expID}s{combination_strength}'
        if p is not None:
            expID = f'{expID}p{p:03d}'

    expID = expID.replace('.', '')
    return expID
