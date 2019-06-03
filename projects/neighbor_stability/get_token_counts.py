import sys


def count_tokens(path):

    with open(path) as infile:
        text = infile.read().strip()
        tokens = text.split(' ')
    n_tokens = len(tokens)
    return n_tokens


def main():

    year = sys.argv[1]

    path = f'../../Data/coha_decades/{year}.txt'
    n_tokens = count_tokens(path)
    print(f'Token count for the decade {year}: {n_tokens}')


if __name__ == '__main__':
    main()
