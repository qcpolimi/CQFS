from dwave.system import LeapHybridSampler


def get_hybrid_from_topology(topology):
    AVAILABLE_TOPOLOGIES = ['chimera', 'pegasus']

    assert topology in AVAILABLE_TOPOLOGIES, f"No hybrid solver available for the requested topology ({topology})." \
                                             f"Available topologies are {AVAILABLE_TOPOLOGIES}."

    if topology == 'pegasus':
        return LeapHybridSampler(solver={'name__regex': '.*version2'})
    if topology == 'chimera':
        return LeapHybridSampler(solver={'name__regex': '.*v1'})
    else:
        return LeapHybridSampler()


def maximum_energy_delta(bqm):
    """Compute conservative bound on maximum change in energy when flipping a single variable"""
    return max(abs(bqm.get_linear(i))
               + sum(abs(bqm.get_quadratic(i, j))
                     for j in bqm.iter_neighbors(i))
               for i in bqm.iter_variables())
