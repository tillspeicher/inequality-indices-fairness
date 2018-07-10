import sys
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from learning_env import LearningEnv
from collections import defaultdict
import inequality_util as iu
sys.path.insert(0, '../shared')
import indices
sys.path.insert(0, '../util')
import datasets.load_adult_data as lad
import datasets.load_compas_data as lcd
import output
import parallelization as para

import matplotlib.pyplot as plt


regression_methods_info = {'Logistic': ('Logistic Regression', 'http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
        'SVM': ('Support Vector classifier', 'http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'),
        'GaussianNB': ('Gausian Naive Bayes', 'http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html'),
        'RandomForest': ('Random Forest classifier', 'http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
        'Oracle': ('Oracle classifier', '')}

method_colors = {'Logistic': u'#1f77b4', 'SVM': u'#ff7f0e', 'RandomForest': u'#2ca02c', 'GaussianNB': 'y', 'Oracle': 'r'}

seeds = [4194, 343, 29032, 8520, 20285,
        20, 6874, 953, 2094, 108134]

def filter_pos_frac(X, y, x_control, target_pos_frac, seed):
    np.random.seed(seed)
    act_pos_frac = np.sum(y) / len(y)
    removal_amount = int((act_pos_frac - target_pos_frac) * len(y))
    if removal_amount > 0:
        print("removing {} pos users".format(removal_amount))
        removal_indices = np.argwhere(y == 1)
    else:
        print("removing {} neg users".format(-removal_amount))
        removal_indices = np.argwhere(y == 0)
    np.random.shuffle(removal_indices)
    removal_indices = removal_indices[:abs(removal_amount) + 1]
    mask = np.ones(len(y), dtype=bool)
    mask[removal_indices] = False
    X = X[mask]
    y = y[mask]
    x_control = {g_name: group[mask] for g_name, group in x_control.items()}
    return X, y, x_control

class OracleClassifier(object):
    """
    Classifier that remembers the ground truth labels for the test set and uses them for prediction.
    Used for reference purposes.
    """
    def __init__(self, ground_truth_labels, X_test):
        self.labels = ground_truth_labels
        self.X_test = X_test
        self.classes_ = np.array([0., 1.])
    def predict(self, X):
        assert X is self.X_test
        return self.labels
    def predict_proba(self, X):
        assert X is self.X_test
        order = np.argsort(self.labels)
        inv_order = np.argsort(order)
        stepsize = 1 / len(order)
        scores = np.arange(0, 1, stepsize)
        scores = scores[inv_order]
        return np.array([1 - scores, scores]).T

class ProbClassEnv(LearningEnv):
    def __init__(self, dataset, val_split=0.):
        self.hyperparam_store_loc = 'hyperparams/prob_class_params'
        output.create_dir('hyperparams')

        self.ds_name = dataset
        self.val_split = val_split
        self.X = None

    def load_data(self, seed=4194, target_pos_frac=None):
        """
        Load the dataset, shuffle the data and split the data into train, test
        and potentially validation
        """
        if self.X is None:
            if self.ds_name == 'Adult':
                self.X, self.y, self.x_control, self.feature_names = lad.load_adult_data()
                self.feature_names[self.feature_names.index('education_num')] = 'ed_num'
                print(self.feature_names)
            elif self.ds_name == 'Compas':
                self.X, self.y, self.x_control, self.feature_names = lcd.load_compas_data()
                # invert y to make 1 correspond to desirable outcome
                self.y = -self.y
            else:
                raise ValueError('Invalid dataset name "{}"'.format(self.ds_name))

            self.y = (self.y + 1.) / 2. # convert to {0, 1}

        np.random.seed(seed)
        order = np.random.permutation(len(self.X))
        self.X = self.X[order]
        self.y = self.y[order]
        self.x_control = {g_name: group[order] for g_name, group in self.x_control.items()}

        if target_pos_frac is not None:
            self.X, self.y, self.x_control = filter_pos_frac(self.X, self.y, self.x_control, target_pos_frac, seed)

        # Todo, since the loaded data objects are members not, we don't need to pass them
        self.setup_data(self.X, self.y, self.x_control, val_split=self.val_split)

    def setup_model(self, classifier):
        """
        Specify which model to use for prediction.
        Supports Logistic Regression, SVM, Gaussian Naive Bayes, Decision Trees,
        Random Forests and the Oracle classifier.
        """
        self.method = classifier
        default_penalty_range = [.01, .05, .1, .2, .5, 1., 2., 5., 10., \
                20., 50., 100.]
        max_depth_values = [i+1 for i in range(7)]
        if classifier == 'Logistic':
            model = linear_model.LogisticRegression
            hyperparam_ranges = {}
        elif classifier == 'SVM':
            model = SVC
            hyperparam_ranges = {'C': default_penalty_range, 'probability': [True]}
        elif classifier == 'GaussianNB':
            model = GaussianNB
            hyperparam_ranges = {}
        elif classifier == 'Tree':
            model = tree.DecisionTreeClassifier
            hyperparam_ranges = {'max_depth': max_depth_values}
        elif classifier == 'RandomForest':
            model = RFC
            hyperparam_ranges = {'max_depth': max_depth_values}
        elif classifier == 'Oracle':
            self.model = OracleClassifier(self.y_test, self.x_test)
            return
        else:
            raise ValueError('Invalid classifier "{}"'.format(classifier))

        super(ProbClassEnv, self).setup_model(model, classifier, hyperparam_ranges)

    def train_model(self):
        """Fit the model to the training data"""
        if self.method == 'Oracle':
            self.model.labels = self.y_test
            self.model.X_test = self.x_test
        else:
            super(ProbClassEnv, self).train_model()

    def calibrate_probabilities(self, calibration_method):
        self.calibration_method = calibration_method
        if calibration_method == 'none' or self.method == 'Oracle':
            return
        #print('Calibrating probabilities with method "{}"'.format(
        #    calibration_method))
        if self.x_val is None:
            self.model = CalibratedClassifierCV(self.model, cv=5,
                    method=calibration_method)
            self.model.fit(self.x_train, self.y_train)
        else:
            self.model = CalibratedClassifierCV(self.model, cv='prefit',
                    method=calibration_method)
            self.model.fit(self.x_val, self.y_val)

def evaluate_inequality_decomp(self, sens_feature_comb):
    pos_class_index, = np.where(self.model.classes_ == 1.)[0]
    scores = self.model.predict_proba(self.x_test)[:,pos_class_index]
    labels = self.y_test

    sens_groups = iu.compute_sensitive_groups(
            sens_feature_comb,
            self.x_control_test)

    fig_name = "{}_{}".format("_".join(sens_feature_comb),
            self.method)
    decomp_res = iu.plot_inequality_decomposition(scores,
            labels, sens_groups, fig_name)

    return decomp_res


calibration_methods = ['sigmoid']

def get_method_name(method, calibration, error_type):
    return method
    method_name = method
    if calibration == 'sigmoid':
        calibration = 'Platt'
    method_name += ' ('
    if len(calibration_methods) > 1:
        method += calibration + ', '
    method_name += error_type + ')'
    return method_name


def eval_prob_class_kernel(class_env, calibration_method, method, sens_feature_combs, seed):
    class_env.load_data(seed)
    class_env.train_model()

    # TODO: add calibration back in if we use the absolute values of probabilities, not just their ordering
    #class_env.calibrate_probabilities(calibration_method)

    model = class_env.model
    preds = model.predict(class_env.x_test)
    pos_class_index, = np.where(model.classes_ == 1.)[0]
    probas = model.predict_proba(class_env.x_test)[:,pos_class_index]

    # compute values to show in table
    labels = class_env.y_test
    accuracy = metrics.accuracy_score(labels, preds)
    
    #diffs = probas - labels
    diffs = preds - labels
    ineq_res, col_names = indices.evaluate_inequality(
            diffs,
            ['Gini', 'Theil', 'GE_2'],
            ['shift', 'sigmoid', 'none'],
            -1)

    # Compute accuracy fairness tradeoff curves
    rejection_metrics = {"Accuracy": lambda labels, pred_labels, _: metrics.accuracy_score(labels, pred_labels),
            "Unfairness": lambda labels, pred_labels, _: indices.ge_2_index(indices.compute_benefits(pred_labels, labels))}
    overall_fairness_res = iu.compute_rejection_curves(probas,
            labels, {}, rejection_metrics)
    rejection_res = {'overall': overall_fairness_res}

    def intergroup_fairness(labels, pred_labels, groups):
        benefits = indices.compute_benefits(pred_labels, labels)
        decomp = iu.get_inequality_decomp(benefits, groups)
        return decomp['intergroup_ineq']

    for feature_comb in sens_feature_combs:
        sens_groups = iu.compute_sensitive_groups(
                feature_comb, class_env.x_control_test)
        intergroup_res = iu.compute_rejection_curves(probas,
                labels, sens_groups,
                {"Overall": rejection_metrics["Unfairness"],
                    "Between-group": intergroup_fairness})
        #intergroup_res[""] = overall_fairness_res["Accuracy"]
        rejection_res[",".join(feature_comb)] = intergroup_res

    # TODO: maybe add Lorenz curves back in
    #util_diffs = preds - labels
    #lorenz_curve_unshifted = iu.compute_lorenz_curve(util_diffs)
    #lorenz_curve_shifted = iu.compute_lorenz_curve(util_diffs + 1)

    return accuracy, ineq_res, col_names, rejection_res

def evaluate_prob_classification(datasets, methods):
    for dataset, sens_features in datasets:
        print("Evaluating dataset {}".format(dataset))

        out_dir = 'results/prob_class/' + dataset + '/'
        output.create_dir(out_dir)

        class_env = ProbClassEnv(dataset, val_split=0.7)
        class_env.load_data()
        sens_feature_combs = iu.powerset(sens_features)

        method_results = {}
        hyperparams = {}
        col_names = []
        for method in methods:
            print('\nEvaluating {}'.format(method))

            params = class_env.setup_model(method)
            hyperparams[method] = params

            for calibration_method in calibration_methods:

                eval_kernel = para.wrap_function(eval_prob_class_kernel)
                results = para.map_parallel(eval_kernel, seeds, invariant_data=(class_env,
                    calibration_method, method, sens_feature_combs), run_parallel=True)

                accuracies, ineqs, col_names, rejection_res = \
                        para.extract_positions(results, range(4))

                avg_ineqs = para.aggregate_results(ineqs, axis=0)
                table_row = [para.aggregate_results(accuracies, axis=0)] + \
                        list(avg_ineqs)
                col_names = ["Acc"] + col_names[0]

                def rejection_curve_aggregator(curves):
                    agg_curves = defaultdict(list)
                    for rejection_curves in curves:
                        for metric, metric_res in rejection_curves.items():
                            agg_curves[metric].append(metric_res)
                    agg_curves = {metric: para.aggregate_results(metric_res) \
                            for metric, metric_res in agg_curves.items()}
                    return agg_curves

                curve_types = ['overall'] + [','.join(feature_comb) for feature_comb in sens_feature_combs]
                rejection_curves = {curve_type: curves for curve_type, curves in \
                        zip(curve_types, para.extract_positions(
                            rejection_res, curve_types))}
                avg_rejection_curves = {curve_type: rejection_curve_aggregator(rejection_res) for \
                                curve_type, rejection_res in \
                                rejection_curves.items()}
                for curve_type, curves in avg_rejection_curves.items():
                    color = method_colors[method]
                    iu.plot_rejection_curves(curves, method,
                            fig_name=curve_type, color=color)

                method_name = method #+ '_' + calibration_method
                method_results[method_name] = table_row

        iu.plot_results(out_dir, dataset)
        wiki_file_loc = iu.get_wiki_file(out_dir)
        with open(wiki_file_loc, 'a') as wiki_file:
            col_format = ['3'] * len(col_names)
            col_format[0] = '2'
            iu.write_wiki_results(
                    wiki_file, col_names,
                    method_results, col_format, hyperparams,
                    regression_methods_info)

            iu.emit_acc_fairness_curves(wiki_file, out_dir, dataset)

            lorenz_desc = "error (y_hat - y)"
            iu.emit_lorenz_curves(wiki_file, out_dir, dataset, lorenz_desc)

        iu.clear_figures()


def evaluate_inequality_decomposition(datasets, methods):
    for dataset, sens_features in datasets:
        print("Evaluating dataset {}".format(dataset))

        out_dir = 'results/class_ineq_decomp/' + dataset + '/'
        output.create_dir(out_dir)

        class_env = ProbClassEnv(dataset, val_split=.7)
        class_env.load_data()

        x_range = np.arange(len(class_env.y_test) + 1) / len(class_env.y_test)
        x_label = "Fraction of rejected users ($\\tau$)"
        y1_label = "Between-group\nunfairness ($\mathcal{E}^2_\\beta$)"
        y2_label = "Accuracy"

        sens_feature_combs = iu.powerset(sens_features)
        method_plots = {",".join(sens_feature_comb): {'intergroup_ineq': {}, 'accuracy': {}} \
                for sens_feature_comb in sens_feature_combs}

        for method in methods:
            print("Evaluating method", method)

            class_env.setup_model(method)
            class_env.train_model()
            #class_env.calibrate_probabilities(calibration_method)

            sens_feature_plots = {'intergroup_ineq': {}}
            for sens_feature_comb in sens_feature_combs:
                print("Evaluating sens feature combination", sens_feature_comb)

                decomp_res = class_env.evaluate_inequality_decomp(sens_feature_comb)
                 
                # method plots
                method_key = ','.join(sens_feature_comb)
                method_plots[method_key]['intergroup_ineq'][(method, '(Unfairness)')] = decomp_res['intergroup_ineq']
                method_plots[method_key]['accuracy'][(method, '(Accuracy)')] = decomp_res['accuracy']

                # sens_feature plots
                sens_feature_label = ', '.join(sens_feature_comb) + " inequality"
                sens_feature_plots['intergroup_ineq'][sens_feature_label] = decomp_res['intergroup_ineq']
                if 'accuracy' not in sens_feature_plots:
                    sens_feature_plots['accuracy'] = decomp_res['accuracy']

            iu.plot_curves(iu.FIG_TYPE_INTERGROUP_INEQ,
                "method_{}".format(method), x_range, x_label,
                sens_feature_plots['intergroup_ineq'], y1_label,
                {"Accuracy": sens_feature_plots['accuracy']}, y2_label)

        linestyles = {'(Unfairness)': '-', '(Accuracy)': ':'}
        for sens_feature_comb, method_results in method_plots.items():
            iu.plot_curves(iu.FIG_TYPE_INTERGROUP_INEQ,
                "feature_{}".format(sens_feature_comb), x_range, x_label,
                method_results['intergroup_ineq'], y1_label,
                method_results['accuracy'], y2_label,
                colors=method_colors, linestyles=linestyles)

        iu.plot_results(out_dir, dataset)

        wiki_file_loc = iu.get_wiki_file(out_dir)
        with open(wiki_file_loc, 'a') as wiki_file:
            iu.emit_curves(wiki_file, out_dir, dataset,
                    iu.FIG_TYPE_INEQ_DECOMP,
                    "Inequality decomposition of the overall GE_2 into intergroup- and intragroup-inequality")
            iu.emit_curves(wiki_file, out_dir, dataset,
                    iu.FIG_TYPE_INTERGROUP_INEQ,
                    "== Intergroup inequalities for methods and feature combinations ==")

        iu.clear_figures()

# parallel kernel
def eval_seed_benefits(class_env, split_features, seed):
    class_env.load_data(seed=seed)
    class_env.train_model()
    predictions = class_env.model.predict(class_env.x_test)

    ds_features_groups = iu.create_feature_groups(
            split_features, class_env.feature_names,
            class_env.x_test, class_env.x_control_test)
    benefits = indices.compute_benefits(predictions,
            class_env.y_test)

    x_labels, intergroup_inequality_fracs, intragroup_inequality_fracs = \
            iu.compute_intersectional_inequalities(split_features,
                    ds_features_groups, benefits)

    # TODO: try at least not to return the x_labels
    return x_labels, intergroup_inequality_fracs, intragroup_inequality_fracs

def evaluate_population_splits(datasets, methods):
    feature_split_order = {'Compas': ['sex', 'race', 'age']}
    #feature_split_order = {'Compas': ['sex', 'race', 'age', 'c'],
    #        #'Adult': ['marital', 'relationship', 'workclass', 'education']}
    #        'Adult': ['marital', 'relationship']}#, 'workclass']}#, 'education']}

    methods.remove('Oracle')

    for dataset, sens_features in datasets:
        print("Evaluating dataset {}".format(dataset))

        out_dir = 'results/intergroup_splits/' + dataset + '/'
        output.create_dir(out_dir)

        split_features = feature_split_order[dataset]
        class_env = ProbClassEnv(dataset, val_split=0.7)
        class_env.load_data()

        x_labels = None
        intergroup_inequality_fracs = {}
        intragroup_inequality_fracs = {}
        for method in methods:
            print("Training method", method)
            class_env.setup_model(method)

            eval_func = para.wrap_function(eval_seed_benefits)
            results = para.map_parallel(eval_func, seeds,
                    invariant_data=(class_env, split_features))

            x_labels = results[0][0]
            intergroup_fracs = []
            intragroup_fracs = defaultdict(lambda: defaultdict(list))
            for _, intergroup_frac, intragroup_frac in results:
                intergroup_fracs.append(intergroup_frac)
                # TODO: relies on ordered dicts
                for sens_comb, frac in zip(x_labels, intragroup_frac):
                    for group, group_share in frac.items():
                        intragroup_fracs[sens_comb][group].append(group_share)
            #intergroup_inequality_fracs[method] = para.mean_with_conf(intergroup_fracs, axis=0)
            intergroup_inequality_fracs[method] = para.aggregate_results(intergroup_fracs, axis=0)
            intragroup_inequality_fracs[method] = {sens_comb: {group: para.aggregate_results(group_fracs, np.mean, axis=0) for group, group_fracs in fracs.items()} for sens_comb, fracs in intragroup_fracs.items()}
           
        iu.plot_curves(iu.FIG_TYPE_INTERGROUP_SPLITS,
                "comparison", x_labels, "Feature combinations",
                intergroup_inequality_fracs,
                "Contribution (%)", bars=True, colors=method_colors)

        # Plot the intragroup inequalities for the various groups
        for method in methods:
            for feature_comb, intergroup_ineq, intragroup_ineqs in zip(x_labels, intergroup_inequality_fracs[method], intragroup_inequality_fracs[method].values()):
                if isinstance(intergroup_ineq, tuple):
                    intergroup_ineq = intergroup_ineq[0]
                # TODO: relies on dict order
                iu.plot_pie(iu.FIG_TYPE_INTERGROUP_SPLITS,
                        "{}_{}_breakdown".format(method, feature_comb),
                        [intergroup_ineq] + list(intragroup_ineqs.values()),
                        ["between-group"] + list(intragroup_ineqs.keys()))

        iu.plot_results(out_dir, dataset)#, output_channel="show")

        wiki_file_loc = iu.get_wiki_file(out_dir)
        with open(wiki_file_loc, 'a') as wiki_file:
            iu.emit_curves(wiki_file, out_dir, dataset,
                    iu.FIG_TYPE_INTERGROUP_SPLITS,
                    "Contribution of between-group unfairness to the overall individual unfairness. The numbers in parentheses after the feature combinations denote the number of population subgroups obtained from splitting the population on all the features.")


def main():
    datasets = [
        ('Compas', ['race', 'sex']),
        #('Adult', ['race', 'sex']),
    ]

    methods = [
        'Logistic',
        'SVM',
        #'GaussianNB',
        'RandomForest',
        'Oracle'
    ]

    output.set_paper_style()

    evaluate_prob_classification(datasets, methods)
    #evaluate_inequality_decomposition(datasets, methods)
    evaluate_population_splits(datasets, methods)

if __name__ == '__main__':
    main()
