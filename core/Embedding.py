import random
import time

import minorminer
from dwave.embedding.chimera import find_clique_embedding as find_clique_embedding_chimera
from dwave.embedding.pegasus import find_clique_embedding as find_clique_embedding_pegasus

from recsys.Base.DataIO import DataIO
from utils.graph import get_nodes, get_edges, _graph_hash, get_nodes_from_embedding, draw_embedding


class Embedding(object):
    EMBEDDING_FUNCTIONS = {
        'pegasus': find_clique_embedding_pegasus,
        'chimera': find_clique_embedding_chimera,
    }
    TARGET_SIZE = {
        'pegasus': 16,
        'chimera': 16,
    }

    def __init__(self, source, target, embedding_name='embedding', qpu_graph='pegasus', source_hash=None):

        self.embedding_name = embedding_name

        self.source = get_edges(source)
        self.target = get_edges(target)

        self.embedding = None
        self.nodelist = []
        self.node_count = 0
        self.elapsed_time = 0

        self.qpu_graph = qpu_graph

        if source_hash is None:
            self.source_hash = _graph_hash(source)
        else:
            self.source_hash = source_hash

    def __lt__(self, other):

        if self.node_count > 0 and other.node_count > 0:
            return self.node_count < other.node_count
        return self.node_count > 0

    def find_embedding(self, find_embedding_function=minorminer.find_embedding, **embedding_parameters):

        # print("Finding possible embeddings...")
        s_time = time.time()

        self.embedding = find_embedding_function(self.source, self.target, **embedding_parameters)
        self.nodelist = sorted(get_nodes_from_embedding(self.embedding))
        self.node_count = len(self.nodelist)

        self.elapsed_time = time.time() - s_time

        return self.embedding

    def find_clique_embedding(self, **embedding_parameters):

        find_clique_embedding_function = self.EMBEDDING_FUNCTIONS[self.qpu_graph]
        clique_graph = get_nodes(self.source)
        target_clique_size = self.TARGET_SIZE[self.qpu_graph]

        s_time = time.time()

        self.embedding = find_clique_embedding_function(clique_graph, target_clique_size, **embedding_parameters)
        self.nodelist = sorted(get_nodes_from_embedding(self.embedding))
        self.node_count = len(self.nodelist)

        self.elapsed_time = time.time() - s_time

        return self.embedding

    def find_embedding_from_source_target(self, S, T,
                                          find_embedding_function=minorminer.find_embedding, **embedding_parameters):

        # print("Finding possible embeddings...")
        s_time = time.time()

        self.embedding = find_embedding_function(S, T, **embedding_parameters)
        self.nodelist = sorted(get_nodes_from_embedding(self.embedding))
        self.node_count = len(self.nodelist)

        self.elapsed_time = time.time() - s_time

        return self.embedding

    def draw_embedding(self, highlight_variables=None, full_drawing=True, show=True):

        active_nodes = [node for edge in self.target for node in edge]
        active_nodes = list(set(active_nodes))

        draw_embedding(self.embedding, active_nodes=active_nodes, highlight_variables=highlight_variables,
                       full_drawing=full_drawing, show=show)

    def save_embedding(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.embedding_name

        print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = self.__dict__

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        print("Saving complete")

    @classmethod
    def load_embedding(cls, folder_path, file_name):

        print("Loading embedding from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        embedding = cls(None, None)
        for attrib_name in data_dict.keys():
            embedding.__setattr__(attrib_name, data_dict[attrib_name])

        print("Loading complete")

        return embedding


class EmbeddingGroup(object):
    """Class used to explore and analyse many embeddings of the same problem."""

    def __init__(self, source, target, embedding_name='embedding', qpu_graph='pegasus', clique_embedding=False,
                 source_hash=None):

        self.embedding_name = embedding_name

        self.source = get_edges(source)
        self.target = get_edges(target)

        self.embeddings = []

        self.n_cases = 0
        self.total_time = 0
        self.average_time = 0

        self.qpu_graph = qpu_graph
        self.clique_embedding = clique_embedding

        if source_hash is None:
            self.source_hash = _graph_hash(source)
        else:
            self.source_hash = source_hash

    def find_embedding(self, n_cases=10, find_embedding_function=minorminer.find_embedding,
                       reset_previous_search=True, resume_from_saved=False, output_folder_path=None,
                       **embedding_parameters):

        if output_folder_path is None:
            output_folder_path = f"./{self.embedding_name}_embedding_group/"

        if reset_previous_search:
            self.embeddings.clear()
            self.n_cases = 0
            self.total_time = 0
            self.average_time = 0

        embedding_check_name = "group_metadata"
        dataIO = DataIO(folder_path=output_folder_path)

        if resume_from_saved:
            try:
                metadata_dict = dataIO.load_data(file_name=embedding_check_name)
                done_cases = metadata_dict['n_cases']
                self.total_time = metadata_dict['total_time']

                done_embeddings = []
                for i in range(done_cases):
                    i_filename = f"e{i:05d}"
                    embedding = Embedding.load_embedding(folder_path=output_folder_path, file_name=i_filename)
                    done_embeddings.append(embedding)

                self.embeddings.clear()
                self.embeddings.extend(done_embeddings)
                self.n_cases = done_cases + 1

                print(f"Found an already existing EmbeddingGroup with {done_cases + 1} done cases.")

            except FileNotFoundError:
                print("Could not resume from a saved run. Starting a new one...")

        assert n_cases >= self.n_cases, "Expected more cases than the ones already done. Choose a larger n_cases."

        for i in range(self.n_cases, n_cases):
            embedding = Embedding(self.source, self.target, embedding_name=self.embedding_name,
                                  qpu_graph=self.qpu_graph, source_hash=self.source_hash)

            if self.clique_embedding:
                embedding.find_clique_embedding(**embedding_parameters)
            else:
                embedding.find_embedding(find_embedding_function, **embedding_parameters)

            i_filename = f"e{i:05d}"
            embedding.save_embedding(output_folder_path, i_filename)

            self.total_time += embedding.elapsed_time
            saved_cases = {'n_cases': i, 'total_time': self.total_time}
            dataIO.save_data(embedding_check_name, saved_cases)

            self.embeddings.append(embedding)

        self.average_time = self.total_time / n_cases
        self.n_cases = n_cases

    def get_embedding(self, type='best'):

        if len(self.embeddings) == 0:
            return None

        if type == 'best':
            sorted_embeddings = sorted(self.embeddings)
            return sorted_embeddings[0]
        elif type == 'average':
            sorted_embeddings = sorted(self.embeddings)
            return sorted_embeddings[self.n_cases // 2]
        elif type == 'random':
            return random.choice(self.embeddings)
        elif type == 'all':
            return self.embeddings.copy()
