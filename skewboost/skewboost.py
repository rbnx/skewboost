from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import dimod


RANK_FUNCTIONS = {
    'auc': roc_auc_score,
    'f1': f1_score
}


def _check_spin_labels(labels):
    assert list(np.unique(labels)) == [-1, 1]
    return labels


class SkewBoost(object):
    def __init__(self, weak_classifiers):
        assert isinstance(weak_classifiers, list) and len(weak_classifiers) > 1
        self.estimators_ = weak_classifiers
        self.n_estimators = len(self.estimators_)
        self.estimator_weights = np.zeros(self.n_estimators)

    def fit(
            self,
            X,
            y,
            sampler,
            alpha=0.2,
            gamma=1,
            rank_function='auc',
            **kwargs):
        assert isinstance(sampler, dimod.Sampler)
        y = _check_spin_labels(y)
        preds = np.float64([_check_spin_labels(m.predict(X))
                            for m in self.estimators_])
        preds *= 1. / self.n_estimators

        rank_cb = RANK_FUNCTIONS[rank_function]

        qij = np.dot(preds, preds.T)
        qij[np.diag_indices_from(qij)] = self._diagonal_terms(
            X, y, preds, alpha, gamma, rank_cb)

        Q = {k: qij[k] for k in zip(*np.triu_indices(len(qij)))}

        res = sampler.sample_qubo(Q, **kwargs)
        samples = np.array(
            [[samp[k] for k in range(self.n_estimators)] for samp in res])

        self.estimator_weights = samples[0]

        return self

    def predict(self, X):
        y = np.zeros(len(X))
        for i in np.nditer(np.nonzero(self.estimator_weights)):
            y += self.estimators_[i].predict(X)

        return np.sign(2 * y - 1)

    def _get_r_scores(self, X, y, rank_fn):
        scores = []
        for m in self.estimators_:
            y_p = [round(x) for x in m.predict(X)]
            _check_spin_labels(y_p)
            scores.append(rank_fn(y, y_p))

        return np.array(scores)

    def _diagonal_terms(self, X, y, y_hat, alpha, gamma, rank_fn):
        scores = self._get_r_scores(X, y, rank_fn)
        s_mean = scores.mean()
        min_amp = scores.std() / 2
        lbl_term = np.dot(y_hat, y)

        for i in range(len(scores)):
            f_0 = abs(s_mean - scores[i])
            f_0 = f_0 if f_0 >= min_amp else min_amp
            f = f_0 if scores[i] >= s_mean else -1 * f_0
            lbl_term[i] += abs(lbl_term[i]) * f * gamma

        return len(X) * 1. / (self.n_estimators ** 2) + alpha - 2 * lbl_term
