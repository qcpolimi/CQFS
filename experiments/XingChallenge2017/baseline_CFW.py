from data.DataLoader import XingChallenge2017Loader
from experiments.baseline_CFW import baseline_CFW
from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender


def main():
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    CF_recommenders = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    baseline_CFW(data_loader, ICM_name, CF_recommenders)


if __name__ == "__main__":
    main()
