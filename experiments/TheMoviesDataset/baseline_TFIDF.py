from data.DataLoader import TheMoviesDatasetLoader
from experiments.baseline_TFIDF import baseline_TFIDF


def main():
    data_loader = TheMoviesDatasetLoader()
    ICM_name = 'ICM_metadata'
    baseline_TFIDF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
