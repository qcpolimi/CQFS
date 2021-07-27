import shutil

import dimod
import dwave_qbsolv
import neal
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from core.CountSampler import CountSampler
from core.Embedding import EmbeddingGroup

ACCEPTED_TOPOLOGIES = ['pegasus', 'chimera']

DEFAULT_QPU_TOPOLOGY = 'pegasus'
DEFAULT_NUM_READS = 100
DEFAULT_N_EMBEDDING_CASES = 10

DEFAULT_SOLVER_LIMIT = {
    'pegasus': 120,
    'chimera': 60,
}


class CQFSEmbedding():
    TEMP_EMBEDDING_FOLDER_PATH = '../embeddings/.temp/'

    def __init__(self, target_sampler=None, qpu_topology=DEFAULT_QPU_TOPOLOGY,
                 n_embedding_cases=DEFAULT_N_EMBEDDING_CASES, embedding_folder_path=None, clique_embedding=False):
        assert qpu_topology in ACCEPTED_TOPOLOGIES, f"QPU topology should be one of {ACCEPTED_TOPOLOGIES}."

        if target_sampler is None:
            target_sampler = DWaveSampler(solver={'topology__type': qpu_topology})
        self.qpu_topology = qpu_topology
        self.target = target_sampler.edgelist

        self.embeddings = {}
        self.n_embedding_cases = n_embedding_cases
        if embedding_folder_path is None:
            self.embedding_folder_path = self.TEMP_EMBEDDING_FOLDER_PATH
            self.save_embeddings = False
        else:
            self.embedding_folder_path = embedding_folder_path
            self.save_embeddings = True

        self.clique_embedding = clique_embedding

    def get_embedding(self, bqm, exp_id='', resume_from_saved=False):
        bqm_graph = bqm.to_networkx_graph()
        # bqm_graph_hash = nx.weisfeiler_lehman_graph_hash(bqm_graph)
        # embedding_dict = self.embeddings.get(bqm_graph_hash)

        # if embedding_dict is not None:
        #     if _check_embedding_validity(embedding_dict, self.qpu_topology):
        #         return embedding_dict

        print(f"Searching for an embedding of the problem onto the {self.qpu_topology} graph...")

        output_folder_path = f"{self.embedding_folder_path}/{exp_id}/embedding/"
        # embedding_group = EmbeddingGroup(bqm_graph, self.target, qpu_graph=self.qpu_topology,
        #                                  source_hash=bqm_graph_hash)
        embedding_group = EmbeddingGroup(bqm_graph, self.target, qpu_graph=self.qpu_topology,
                                         clique_embedding=self.clique_embedding)
        embedding_group.find_embedding(self.n_embedding_cases, output_folder_path=output_folder_path,
                                       resume_from_saved=resume_from_saved)

        found_embedding = embedding_group.get_embedding()
        embedding_dict = found_embedding.embedding
        # self.embeddings[bqm_graph_hash] = embedding_dict

        if not self.save_embeddings:
            shutil.rmtree(output_folder_path)
        else:
            found_embedding.save_embedding(output_folder_path, 'best_embedding')

        return embedding_dict


class CQFSSampler(dimod.Sampler):

    def __init__(self, embedding, target_sampler=None, qpu_topology=DEFAULT_QPU_TOPOLOGY, num_reads=DEFAULT_NUM_READS,
                 return_embedding=True):
        assert qpu_topology in ACCEPTED_TOPOLOGIES, f"QPU topology should be one of {ACCEPTED_TOPOLOGIES}."

        if target_sampler is None:
            target_sampler = DWaveSampler(solver={'topology__type': qpu_topology})
        self.sampler = FixedEmbeddingComposite(target_sampler, embedding=embedding)
        self.num_reads = num_reads
        self.return_embedding = return_embedding

    @property
    def properties(self):
        return self.sampler.properties

    @property
    def parameters(self):
        return self.sampler.parameters

    def sample(self, bqm, **parameters):

        if parameters.get('num_reads') is None:
            parameters['num_reads'] = self.num_reads

        if parameters.get('return_embedding') is None:
            parameters['return_embedding'] = self.return_embedding

        return self.sampler.sample(bqm, **parameters)


class CQFSSimulatedAnnealingSampler(neal.SimulatedAnnealingSampler):

    def __init__(self, num_reads=DEFAULT_NUM_READS):
        super(CQFSSimulatedAnnealingSampler, self).__init__()
        self.num_reads = num_reads

    def sample(self, bqm, **parameters):
        if parameters.get('num_reads') is None:
            parameters['num_reads'] = self.num_reads

        return super(CQFSSimulatedAnnealingSampler, self).sample(bqm, **parameters)


class CQFSQBSolvTabuSampler(dwave_qbsolv.QBSolv):

    def __init__(self, num_reads=50, solver_limit=DEFAULT_SOLVER_LIMIT[DEFAULT_QPU_TOPOLOGY]):
        super(CQFSQBSolvTabuSampler, self).__init__()

        self.num_reads = num_reads
        self.solver_limit = solver_limit

    def sample(self, bqm, **parameters):

        if parameters.get('num_repeats') is None:
            parameters['num_repeats'] = self.num_reads

        if parameters.get('solver_limit') is None:
            parameters['solver_limit'] = self.solver_limit

        if parameters.get('solver') is not None:
            assert parameters['solver'] == 'tabu', \
                "Should not use a different solver than 'tabu' with CQFSQBSolvTabuSampler."

        sampleset = super(CQFSQBSolvTabuSampler, self).sample(bqm, **parameters)
        return sampleset


class CQFSQBSolvSampler(dwave_qbsolv.QBSolv):

    def __init__(self, child_sampler=None, num_reads=DEFAULT_NUM_READS,
                 solver_limit=DEFAULT_SOLVER_LIMIT[DEFAULT_QPU_TOPOLOGY]):
        super(CQFSQBSolvSampler, self).__init__()

        self.num_reads = num_reads

        if child_sampler is None:
            child_sampler = neal.SimulatedAnnealingSampler()
        self.child_sampler = CountSampler(child_sampler)

        self.solver_limit = solver_limit

    def sample(self, bqm, **parameters):

        if parameters.get('num_repeats') is None:
            parameters['num_repeats'] = self.num_reads

        if parameters.get('solver') is None:
            parameters['solver'] = self.child_sampler

        if parameters.get('solver_limit') is None:
            parameters['solver_limit'] = self.solver_limit

        sampleset = super(CQFSQBSolvSampler, self).sample(bqm, **parameters)
        sampleset.info['subproblem_sampler_calls'] = self.child_sampler.get_count()
        self.child_sampler.reset_count()

        return sampleset
