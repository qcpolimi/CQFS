from data.DataLoader import CiteULike_aLoader
from experiments.baseline_CBF import baseline_CBF


def main():
    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'
    baseline_CBF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
