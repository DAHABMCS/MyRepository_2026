class AdaptiveWeightManager:
    def __init__(self, alpha=0.2, min_w=0.2, max_w=0.8):
        self.alpha = alpha
        self.min_w = min_w
        self.max_w = max_w
        self.weights = {"indicators": 0.5, "ml": 0.5}
        self.performance = {
            "indicators": {"wins": 0, "losses": 0},
            "ml": {"wins": 0, "losses": 0}
        }

    def record_result(self, source, correct=True):
        """Update historical performance (call after trades)."""
        if correct:
            self.performance[source]["wins"] += 1
        else:
            self.performance[source]["losses"] += 1

    def win_rate(self, stats):
        total = stats["wins"] + stats["losses"]
        return stats["wins"] / total if total > 0 else 0.5

    def update_weights(self):
        ind_wr = self.win_rate(self.performance["indicators"])
        ml_wr = self.win_rate(self.performance["ml"])
        total_wr = ind_wr + ml_wr
        if total_wr == 0:
            return self.weights

        # Target proportional weights
        target_ind = ind_wr / total_wr
        target_ml = ml_wr / total_wr

        # EWA update
        self.weights["indicators"] = (1 - self.alpha) * self.weights["indicators"] + self.alpha * target_ind
        self.weights["ml"] = (1 - self.alpha) * self.weights["ml"] + self.alpha * target_ml

        # Apply floor/ceiling
        self.weights["indicators"] = max(self.min_w, min(self.max_w, self.weights["indicators"]))
        self.weights["ml"] = max(self.min_w, min(self.max_w, self.weights["ml"]))

        # Re-normalize
        total = self.weights["indicators"] + self.weights["ml"]
        self.weights["indicators"] /= total
        self.weights["ml"] /= total

        return self.weights

    def combine_confidence(self, indicator_conf, ml_conf):
        """Return weighted combined confidence."""
        self.update_weights()
        return (
                indicator_conf * self.weights["indicators"] +
                ml_conf * self.weights["ml"]
        )
