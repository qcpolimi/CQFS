from data.DataLoader import XingChallenge2017Loader
from experiments.baseline_CBF import baseline_CBF


def main():
    data_loader = XingChallenge2017Loader()
    ICM_name = 'ICM_all'
    baseline_CBF(data_loader, ICM_name)


if __name__ == "__main__":
    main()
