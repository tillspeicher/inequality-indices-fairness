import sys
import numpy as np
from itertools import product, repeat
import shelve
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
sys.path.insert(0, '../fairness_optimization/') # the code for fair classification is in this directory
import utils as ut
sys.path.insert(0, '../util')
import persistence

class LearningEnv(object):
    """
    Learning environment baseclass that holds the data and model for each experiment.
    It should be subclassed for the specific experiment
    """
    def __init__(self):
        pass

    def setup_data(self, X, y, x_control, train_split=.7,
            val_split=0.):
        print('Loaded {} dataset with dimension {}'.format(
            self.ds_name, X.shape))

        # Split data into training and testing
        self.x_train, self.y_train, self.x_control_train, \
                self.x_test, self.y_test, self.x_control_test = \
                ut.split_into_train_test(X, y, x_control, train_split)

        # create a validation set if specified
        if val_split > 0.:
            self.x_train, self.y_train, self.x_control_train, \
                    self.x_val, self.y_val, self.x_control_val = \
                    ut.split_into_train_test(self.x_train,
                            self.y_train, self.x_control_train, val_split)
        else:
            self.x_val, self.y_val, self.x_control_val = None, None, None

    def find_hyperparams_cv(self, model_constructor, hyperparam_ranges):
        params = []
        val_ranges = []

        for param, val_range in hyperparam_ranges.items():
            params.append(param)
            val_ranges.append(val_range)

        if self.x_val is None:
            eval_func = evaluate_params_cv
            data = repeat((self.x_train, self.y_train))
        else:
            eval_func = evaluate_params_val
            data = repeat((self.x_train, self.y_train, self.x_val, self.y_val))

        pool = mp.Pool()
        val_combs = list(product(*val_ranges))
        models = [model_constructor(**dict(zip(params, val_comb))) for val_comb in val_combs]
        param_scores = pool.map(eval_func, zip(models, data))
        pool.close()
        pool.join()
        
        best_param_index = np.argmax(param_scores)
        best_vals = val_combs[best_param_index]

        return dict(zip(params, best_vals))

    def setup_model(self, model_constructor, model_name, hyperparam_ranges):
        best_params = self.load_hyperparams(model_name, hyperparam_ranges)
        if best_params is None:
            best_params = self.find_hyperparams_cv(model_constructor,
                    hyperparam_ranges)
            self.save_hyperparams(model_name, hyperparam_ranges,
                    best_params)
        print('best parameters:', best_params)

        self.model = model_constructor(**best_params)
        return best_params

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def get_key(self, model_name):
        return self.ds_name + '_' + model_name

    def load_hyperparams(self, model_name, hyperparam_ranges):
        return persistence.load_params(self.hyperparam_store_loc,
                self.get_key(model_name), hyperparam_ranges)

    def save_hyperparams(self, model_name, hyperparam_ranges,
            best_hyperparams):
        persistence.save_params(self.hyperparam_store_loc,
                self.get_key(model_name), hyperparam_ranges,
                best_hyperparams)


def evaluate_params_cv(arg):
    # no validation set, use cross-validation
    NUM_FOLDS = 5
    model, (x_train, y_train) = arg
    cv_scores = cross_val_score(model, x_train,
            y_train, cv=NUM_FOLDS, n_jobs=1)
    return cv_scores.mean()

def evaluate_params_val(arg):
    # validation set, use it to estimate parameters
    model, (x_train, y_train, x_val, y_val) = arg
    model.fit(x_train, y_train)
    return model.score(x_val, y_val)

