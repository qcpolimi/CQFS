from data.DataLoader import TheMoviesDatasetLoader
from experiments.baseline_CBF import baseline_CBF


def main():
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'
    baseline_CBF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
