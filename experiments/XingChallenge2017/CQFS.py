from data.DataLoader import XingChallenge2017Loader
from experiments.run_CQFS import run_CQFS
from recsys.Recommender_import_list import ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender


from dwave.system import DWaveSampler
# from neal import SimulatedAnnealingSampler
# from core.CQFSSampler import CQFSSimulatedAnnealingSampler, CQFSQBSolvSampler


def main():
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'

    ##################################################
    # CQFS hyperparameters and settings

    percentages = [40, 60, 80, 95]
    alphas = [1]
    betas = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    combination_strengths = [1, 10, 100, 1000, 10000]

    ##################################################
    # Samplers

    solver_class = DWaveSampler
    # solver_class = SimulatedAnnealingSampler
    # solver_class = CQFSSimulatedAnnealingSampler
    # solver_class = CQFSQBSolvSampler

    CF_recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]

    save_FPMs = True
    save_BQMs = True

    run_CQFS(data_loader, ICM_name, percentages, alphas, betas, combination_strengths, solver_class,
             CF_recommender_classes, save_FPMs, save_BQMs)


if __name__ == '__main__':
    main()
