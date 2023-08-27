from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Argumentos para o cálculo da métrica de distância entre grafos bipartidos')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
