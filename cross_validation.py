import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Nodes.SymbolicRegressionNodes import FeatureNode


class CrossValidation:
    def __init__(self, GA: SimpleGP, terminals, X=None, y=None):
        self.GA = GA

        # Set the default Boston dataset
        if X is None and y is None:
            X, y = datasets.load_boston(return_X_y=True)
            self.X = X
            self.y = y
        else:
            self.X = X
            self.y = y

        self.ksplits = 10

        # Set the feature terminals as it is based on the size of the dataset
        for i in range(X.shape[1]):
            terminals.append(FeatureNode(i))  # add a feature node for each feature

        self.GA.terminals = terminals

    def get_splits(self):
        kf = KFold(n_splits=self.ksplits, random_state=42)
        X = self.X
        y = self.y

        for trn_idx, tst_idx in kf.split(X):
            # Get the data for the split
            X_train, X_test = X[trn_idx], X[tst_idx]
            y_train, y_test = y[trn_idx], y[tst_idx]

            # Fit the scaler on the training set
            X_scaler = StandardScaler()
            # y_scaler = StandardScaler()

            X_train = X_scaler.fit_transform(X_train)
            # y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))

            # Transform the test data
            X_test = X_scaler.transform(X_test, copy=True)
            # y_test = y_scaler.transform(y_test.reshape(-1, 1), copy=True)
            # TODO: we dont scale the targets

            yield X_train, X_test, X_scaler, y_train, y_test, None

    def validate(self):
        train_mses = []
        train_Rs = []

        test_mses = []
        test_Rs = []

        run = 0
        for X_train, X_test, X_scaler, y_train, y_test, _ in self.get_splits():
            # Set the data in the fitness function
            self.GA.fitness_function = SymbolicRegressionFitness(X_train, y_train)

            # Run the GA and get the best function
            self.GA.run()
            best_function = self.GA.fitness_function.elite
            best_fitness = best_function.fitness

            # TODO: return this?
            # Compute the training metrics
            train_mse = best_fitness
            train_R = 1.0 - best_fitness / np.var(y_train)

            # Make predictions and transform them
            y_pred = best_function.get_output(X_test)
            # y_pred = y_scaler.inverse_transform(y_pred)

            # Compute the testing metrics
            test_mse = np.mean(np.square(y_test - y_pred))
            test_R = 1.0 - test_mse / np.var(y_test)

            train_mses.append(train_mse)
            train_Rs.append(train_R)

            test_mses.append(test_mse)
            test_Rs.append(test_Rs)

            run += 1
            print('KFold Run: ', run,
                  '\nTraining\n\tMSE:', np.round(train_mse, 3),
                  '\n\tRsquared:', np.round(train_R, 3),
                  '\nTesting\n\tMSE:', np.round(test_mse, 3),
                  '\n\tRsquared:', np.round(test_R, 3)
                  )

        print('KFold Result ',
              '\nTesting\n\tMSE:', np.mean(test_mses),
              '\nTraining\n\tMSE:', np.mean(train_mses),
              )
        # Return the average and the deviation over the runs
        return (np.mean(test_mses), np.std(test_mses)), (np.mean(test_Rs), np.std(test_Rs))
