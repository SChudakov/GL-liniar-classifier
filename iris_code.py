import matplotlib.pyplot as plt
from sklearn import datasets


def plot_features(iris, name_1, pos_1, name_2, pos_2):
    X1 = iris.data[:, pos_1]
    X2 = iris.data[:, pos_2]
    y = iris.target

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

    plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for pos_1, name_1 in enumerate(feature_names):
        for pos_2 in range(pos_1 + 1, len(feature_names)):
            name_2 = feature_names[pos_2]
            plot_features(iris, name_1, pos_1, name_2, pos_2)
