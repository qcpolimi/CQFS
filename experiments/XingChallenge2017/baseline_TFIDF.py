from data.DataLoader import XingChallenge2017Loader
from experiments.baseline_TFIDF import baseline_TFIDF


def main():
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    baseline_TFIDF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
