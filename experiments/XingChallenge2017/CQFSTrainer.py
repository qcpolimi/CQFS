from data.DataLoader import XingChallenge2017Loader
from experiments.train_CQFS import train_CQFS
from recsys.Recommender_import_list import ItemKNNCFRecommender, PureSVDItemRecommender, \
    RP3betaRecommender


from dwave.system import DWaveSampler
# from neal import SimulatedAnnealingSampler


def main():
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'

    percentages = [40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    solver_class = DWaveSampler
    # solver_class = SimulatedAnnealingSampler

    CF_recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]

    cpu_count_div = 1
    cpu_count_sub = 0

    train_CQFS(data_loader, ICM_name, percentages, alphas, betas, combination_strengths, solver_class,
               CF_recommender_classes, cpu_count_div=cpu_count_div, cpu_count_sub=cpu_count_sub)


if __name__ == '__main__':
    main()
