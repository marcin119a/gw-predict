import argparse
import random
from generator_dataset import random_dataset
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", required=True, default=100, help="the number of signals")
    ap.add_argument("-m1", "--mass1", required=True, type=int, default=10, help="First mass of black hole")
    ap.add_argument("-m2", "--mass2", required=True, type=int, default=40, help="Second mass of black hole")
    ap.add_argument("-time_steps", "--ts", required=True, type=int, default=1400, help="Time steps for single signal")


    args = vars(ap.parse_args())
    n = args["n"]
    random.seed(int(n))

    m1 = args["m1"]
    m2 = args["m2"]
    time_steps = args["time_steps"]
    
    X, y = random_dataset(m1=m1, m2=m2, n_steps=n, iteraction=time_steps)
    
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    data = {'xtrain': X_train, 'xtest' : X_test, 'ytrain': y_train, 'ytest': y_test}
    hkl.dump(data, 'D-SET({0},{1}).hkl'.format(n,time_steps))


if __name__ == '__main__':
    main()
