import argparse
import random
from generator import random_dataset
from sklearn.model_selection import train_test_split
import hickle as hkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=1000, help="the number of signals")
    ap.add_argument("-m1", "--mass1", type=int, default=40, help="First mass of black hole")
    ap.add_argument("-m2", "--mass2", type=int, default=70, help="Second mass of black hole")
    ap.add_argument("-time_steps", "--ts", type=int, default=800, help="Time steps for single signal")

    args = vars(ap.parse_args())
    n = args["n"]
    random.seed(int(n))

    m1 = args["mass1"]
    m2 = args["mass2"]
    time_steps = args["ts"]


    X, y = random_dataset(m1=m1, m2=m2, n_steps=time_steps, batch_size=n, channels=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    data = {'xtrain': X_train, 'xtest' : X_test, 'ytrain': y_train, 'ytest': y_test}

    hkl.dump(data, 'D-SET(n={0},time_steps={1},mass1={2},mass2={3}).hkl'.format(n, time_steps, m1, m2))


if __name__ == '__main__':
    main()
