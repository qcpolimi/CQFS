from data.DataLoader import XingChallenge2017Loader
from experiments.train_CF import train_CF


def main():
    data_loader = XingChallenge2017Loader()
    train_CF(data_loader)


if __name__ == '__main__':
    main()
