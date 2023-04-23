import os


def clean():
    for f in os.listdir(os.getcwd()):
        if '_' in f and f.endswith('.csv'):
            os.remove(f)


if __name__ == '__main__':
    clean()

