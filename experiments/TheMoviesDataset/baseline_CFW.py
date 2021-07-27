from data.DataLoader import TheMoviesDatasetLoader
from experiments.baseline_CFW import baseline_CFW
from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender


def main():
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'
    CF_recommenders = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    baseline_CFW(data_loader, ICM_name, CF_recommenders)


if __name__ == "__main__":
    main()
