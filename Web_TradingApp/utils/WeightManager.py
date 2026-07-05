class WeightManager:
    """
    Minimal adapter to match existing usage:
      self.weight_manager.combine_confidence(base_confidence, ml_confidence_pct)
    - Inputs are expected in 0..100
    - Returns a combined confidence in 0..100
    """
    def __init__(self, w_base=0.6, w_ml=0.4, cap=95.0):
        self.w_base = float(w_base)
        self.w_ml = float(w_ml)
        self.cap = float(cap)

    def combine_confidence(self, base_confidence_pct: float, ml_confidence_pct: float) -> float:
        b = max(0.0, min(100.0, float(base_confidence_pct)))
        m = max(0.0, min(100.0, float(ml_confidence_pct)))
        # Confidence blending with diminishing returns on very high ML values:
        m_adj = 100.0 * (m / 100.0) ** 0.9
        combined = self.w_base * b + self.w_ml * m_adj
        return min(self.cap, combined)