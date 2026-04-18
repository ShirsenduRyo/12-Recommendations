import numpy as np

class HybridOPE:

    def __init__(self, encoder, temporal_model, target_policy):
        self.encoder = encoder
        self.temporal_model = temporal_model
        self.target_policy = target_policy

    def estimate(self, X, A, R, logging_prob):

        z = self.encoder.predict(X)

        r_hat = self.temporal_model.predict(z)

        pi = self.target_policy.predict_proba(z)

        target_prob = pi[np.arange(len(A)), A]

        weights = target_prob / logging_prob

        weights = np.clip(weights, 0, 10)

        value = np.mean(weights * r_hat)

        return value