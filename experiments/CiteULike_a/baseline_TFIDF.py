from data.DataLoader import CiteULike_aLoader
from experiments.baseline_TFIDF import baseline_TFIDF


def main():
    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'
    baseline_TFIDF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
