import argparse
import random
from generator_dataset import * 
from sklearn.model_selection import train_test_split
import hickle as hkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=100, help="the number of signals")
    ap.add_argument("-m1", "--mass1", type=int, default=10, help="First mass of black hole")
    ap.add_argument("-m2", "--mass2", type=int, default=40, help="Second mass of black hole")
    ap.add_argument("-time_steps", "--ts", type=int, default=1400, help="Time steps for single signal")

    args = vars(ap.parse_args())
    n = args["n"]
    random.seed(int(n))

    m1 = args["mass1"]
    m2 = args["mass2"]
    time_steps = args["ts"]

    X_norm, y_norm, _, _ = random_dataset(m1=m1, m2=m2, n_steps=time_steps, iteraction=n, quark=False, max_model=True)
    len(X_norm)
    len(y_norm)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.33, random_state=42)

    data = {'xtrain': X_train, 'xtest' : X_test, 'ytrain': y_train, 'ytest': y_test}


    hkl.dump(data, 'D-SET({0},{1}){2}.hkl'.format(n,time_steps, "max-model"))


if __name__ == '__main__':
    main()
