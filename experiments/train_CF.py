from data.DataLoader import DataLoader
from recsys.Base.Evaluation.Evaluator import EvaluatorHoldout
from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from recsys.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


def train_CF(data_loader: DataLoader, n_cases=50, n_random_starts=15):
    # Load data
    data_loader.load_data()
    dataset_name = data_loader.get_dataset_name()

    # Get the warm split
    URM_train, URM_validation, URM_test = data_loader.get_warm_split()

    # Instantiate the validation evaluator needed by the parameter search algorithm
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    recommender_classes = [ItemKNNCFRecommender, PureSVDItemRecommender, RP3betaRecommender]
    for Recommender in recommender_classes:
        # Name of the experiment and output results folder path
        recommendation_folder = f"{dataset_name}/{Recommender.RECOMMENDER_NAME}"
        output_folder_path = f"../../results/{recommendation_folder}/"

        runParameterSearch_Collaborative(Recommender, URM_train, evaluator_validation=evaluator_validation,
                                         output_folder_path=output_folder_path, n_cases=n_cases,
                                         n_random_starts=n_random_starts, resume_from_saved=True)
