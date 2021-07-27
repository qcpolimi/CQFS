import dimod
from core.Counter import Counter


class CountSampler(dimod.Sampler):

    def __init__(self, child_sampler: dimod.Sampler):
        self.child_sampler: dimod.Sampler = child_sampler
        self.counter: Counter = Counter.get_instance()

    @property
    def properties(self):
        return self.child_sampler.properties

    @property
    def parameters(self):
        return self.child_sampler.properties

    def sample(self, bqm: dimod.BinaryQuadraticModel, **parameters):
        self.counter.count()
        return self.child_sampler.sample(bqm, **parameters)

    def get_count(self):
        return self.counter.get_count()

    def reset_count(self):
        self.counter.reset_count()
