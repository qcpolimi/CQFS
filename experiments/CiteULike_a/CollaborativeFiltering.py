from data.DataLoader import CiteULike_aLoader
from experiments.train_CF import train_CF


def main():
    data_loader = CiteULike_aLoader()
    train_CF(data_loader)


if __name__ == '__main__':
    main()
