from skewboost import SkewBoost
from sklearn.ensemble import AdaBoostClassifier
import sklearn.datasets
from sklearn.model_selection import train_test_split, GridSearchCV
import dimod


bc_data = sklearn.datasets.load_breast_cancer(return_X_y=False)

X_train, X_test, y_train, y_test = train_test_split(
    bc_data.data,
    bc_data.target,
    test_size=0.20,
    stratify=bc_data.target,
)

# {0,1} => {-1,1}
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

ab_clf = AdaBoostClassifier(n_estimators=20)
ab_clf.fit(X_train, y_train)


sampler = dimod.SimulatedAnnealingSampler()
skb = SkewBoost(ab_clf.estimators_)

skb.fit(X_train, y_train, sampler, alpha=0.2, gamma=10)

# Do something cool with the new weights:
print(skb.estimator_weights)
