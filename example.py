from skewboost import SkewBoost
from sklearn.ensemble import AdaBoostClassifier
import sklearn.datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dimod
import sys

bc_data = sklearn.datasets.load_breast_cancer(return_X_y=False)

X_train, X_test, y_train, y_test = train_test_split(
    bc_data.data,
    bc_data.target,
    test_size=0.20,
    stratify=bc_data.target,
)

# Transform {0,1} labels into {-1,1} labels.
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

# Train classical model
ab_clf = AdaBoostClassifier(n_estimators=20)
ab_clf.fit(X_train, y_train)

sampler = {}

if len(sys.argv) < 2:
    sampler['sampler'] = dimod.SimulatedAnnealingSampler()
    sampler['params'] = {}
else:
    token = sys.argv[1]
    sampler['sampler'] = EmbeddingComposite(
        DWaveSampler(token=token, solver={'qpu': True}))
    sampler['params'] = {
        'num_reads': 1000,
        'auto_scale': True,
        'num_spin_reversal_transforms': 10,
        'postprocess': 'optimization',
    }

skb = SkewBoost(ab_clf.estimators_)
# Train SkewBoost on a D-Wave QPU or using SimulatedAnnealingSampler
skb.fit(
    X_train,
    y_train,
    sampler['sampler'],
    alpha=0.2,
    gamma=10,
    **sampler['params'])

# Do something cool with the new weights:
print(skb.estimator_weights)
