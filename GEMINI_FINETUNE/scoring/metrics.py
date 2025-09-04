from sklearn.metrics import cohen_kappa_score

def calculate_weighted_kappa(y_true, y_pred):
    try:
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception as e:
        print(f"Error calculating kappa: {e}")
        return 0.0
