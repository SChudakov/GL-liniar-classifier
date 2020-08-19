import numpy as np
from scipy import interp

from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure

from sklearn.metrics import plot_roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
from sklearn import datasets
import os

_plots_dir = 'plots/'


class InHouseGaussianNB:
    _estimator_type = "classifier"
    classes_ = [0, 1]

    def __init__(self, num_of_buckets, threshold) -> None:
        self._num_of_buckets = num_of_buckets
        self._threshold = threshold

        self._unique_sates = None

        self._observations_by_feature_and_state_map = None
        self._feature_rage_map = None
        self._total_state_observations_map = None
        self._occurrence_probability = None

    def fit(self, X, y):
        self._observations_by_feature_and_state_map = dict()
        self._feature_rage_map = dict()
        self._total_state_observations_map = dict()
        self._occurrence_probability = dict()

        self._unique_states = set(y)

        # _feature_rage_map
        for feature_id in range(X.shape[1]):
            self._feature_rage_map[feature_id] = (X[:, feature_id].min(), X[:, feature_id].max())

        # _total_state_observations_map
        for state in self._unique_states:
            total_observations_of_state = 0
            for i, row in enumerate(X):
                if y[i] == state:
                    total_observations_of_state += 1
            self._total_state_observations_map[state] = total_observations_of_state

        # _observations_by_feature_and_state_map
        for feature_id in range(len(X[0])):
            for state in self._unique_states:
                entry = (feature_id, state)
                observations_by_bucket = np.array([0] * self._num_of_buckets)
                range_start, range_end = self._feature_rage_map[feature_id]
                for i, row in enumerate(X):
                    if y[i] == state:
                        bucket_id = self._get_bucket_id(range_start, range_end, row[feature_id])
                        observations_by_bucket[bucket_id] += 1
                self._observations_by_feature_and_state_map[entry] = observations_by_bucket

        # _occurrence_probability
        for feature_id in range(len(X[0])):
            for state in self._unique_states:
                entry = (feature_id, state)
                result = self._observations_by_feature_and_state_map[entry] / self._total_state_observations_map[state]
                self._occurrence_probability[entry] = result

    def _get_bucket_id(self, range_start, range_end, feature_value):
        delta = 0.00001
        if feature_value < range_start:
            return 0
        if feature_value > range_end:
            return self._num_of_buckets - 1
        for bucket_id in range(self._num_of_buckets):
            bucket_start = self._get_bucket_start(range_start, range_end, bucket_id)
            bucket_end = self._get_bucket_end(range_start, range_end, bucket_id)
            if bucket_start - delta <= feature_value <= bucket_end + delta:
                return bucket_id

    def _get_bucket_start(self, range_start, range_end, bucket_id):
        return range_start + ((range_end - range_start) * bucket_id) / self._num_of_buckets

    def _get_bucket_end(self, range_start, range_end, bucket_id):
        return range_start + ((range_end - range_start) * (bucket_id + 1)) / self._num_of_buckets

    def score(self, X_test, y_test):
        if self._feature_rage_map is None:
            raise RuntimeError('fit() method has not been called!')

        total_rows = len(X_test)
        correct_rows = 0
        for i, row in enumerate(X_test):
            prediction = self._predict_binary_feature(row)
            if prediction == y_test[i]:
                correct_rows += 1
        return correct_rows / total_rows

    def _predict_binary_feature(self, X_row):
        probabilities = self._predict_probability_row(X_row)
        if probabilities[1] == 0:
            return 0
        if probabilities[0] / probabilities[1] > self._threshold:
            return 0
        else:
            return 1

    def predict_proba(self, X):
        if self._feature_rage_map is None:
            raise RuntimeError('fit() method has not been called!')

        result = []
        for row in X:
            result.append(self._predict_probability_row(row))

        return np.array(result)

    def _predict_probability_row(self, X_row):
        return [self._predict_probability_row_state(X_row, 0), self._predict_probability_row_state(X_row, 1)]

    def _predict_probability_row_state(self, X_row, state):
        result = 1.0
        for feature_id, feature_value in enumerate(X_row):
            entry = (feature_id, state)
            range_start, range_end = self._feature_rage_map[feature_id]
            bucket_id = self._get_bucket_id(range_start, range_end, feature_value)

            feature_probability = self._occurrence_probability[entry][bucket_id]
            result *= feature_probability
        return result


def plot_features(dataset, name_1, pos_1, name_2, pos_2):
    X1 = dataset[0][:, pos_1]
    X2 = dataset[0][:, pos_2]
    y = dataset[1]

    x_min, x_max = X1.min() - .5, X1.max() + .5
    y_min, y_max = X2.min() - .5, X2.max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X1, X2, c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(name_1)
    plt.ylabel(name_2)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.savefig(os.path.join(_plots_dir, f'features_plot_{name_1}_{name_2}.png'))


def test_classifiers(names, classifiers, data_sets, h, name_1, name_2):
    plt.figure(figsize=(27, 9))
    i = 1
    for data_set_count, data_set in enumerate(data_sets):
        X, y = data_set
        # X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # plot the dataset
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(data_sets), len(classifiers) + 1, i)
        if data_set_count == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, classifier in zip(names, classifiers):
            ax = plt.subplot(len(data_sets), len(classifiers) + 1, i)
            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)

            # Plot the decision boundary.
            # For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(classifier, "decision_function"):
                Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if data_set_count == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(_plots_dir, f'probability_density_{name_1}_{name_2}.png'))


def roc_curve_and_auc(names, classifiers, data_sets, name_1, name_2):
    for data_set_count, data_set in enumerate(data_sets):
        X, y = data_set

        # Classification and ROC analysis
        cv = StratifiedKFold(n_splits=6)
        for name, classifier in zip(names, classifiers):
            fig = figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1, 1, 1)

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for i, (train, test) in enumerate(cv.split(X, y)):
                classifier.fit(X[train], y[train])
                viz = plot_roc_curve(classifier, X[test], y[test],
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax)
                interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title=f"ROC curve for {name}")
            ax.legend(loc="lower right")
            plt.savefig(os.path.join(_plots_dir, f'roc_auc_{name}_{name_1}_{name_2}.png'))
            plt.close(fig)


def execute_for_features(dataset, name_1, pos_1, name_2, pos_2):
    plot_features(dataset, name_1, pos_1, name_2, pos_2)

    h = .02  # step size in the mesh

    names = [
        'Out-of-box classifier',
        'In-house classifier'
    ]
    classifiers = [
        GaussianNB(),
        InHouseGaussianNB(10, 1.0)
    ]
    X = dataset[0]
    X = X[:, (pos_1, pos_2)]
    y = dataset[1]

    data_sets = [(X, y)]

    test_classifiers(names, classifiers, data_sets, h, name_1, name_2)

    roc_curve_and_auc(names, classifiers, data_sets, name_1, name_2)


def to_binary_feature_dataset(dataset):
    X = list()
    y = list()
    for i in range(len(dataset.data)):
        if dataset.target[i] in (0, 1):
            X.append(dataset.data[i])
            y.append(dataset.target[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    iris = datasets.load_iris()
    dataset = to_binary_feature_dataset(iris)

    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for pos_1, name_1 in enumerate(feature_names):
        for pos_2 in range(pos_1 + 1, len(feature_names)):
            name_2 = feature_names[pos_2]
            execute_for_features(dataset, name_1, pos_1, name_2, pos_2)


if __name__ == '__main__':
    main()
