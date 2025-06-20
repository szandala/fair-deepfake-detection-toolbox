from .common import DataFrame, _compute_global_parity


def equality_of_odds_parity(expected=None, predicted=None, sensitive_features=None, data=None, one=True):
    """
    one - return values for whole dataset
    """
    if data is None:
        data = DataFrame(expected, predicted, sensitive_features)

    tpr_dict = {}
    fpr_dict = {}

    for group in data.groups():
        # Also Recall
        # P(C=1 | Y=1, A=a)
        tpr = (
            data.tp(group) / (data.tp(group) + data.fn(group))
            if (data.tp(group) + data.fn(group)) != 0
            else 0
        )
        tpr_dict[group] = tpr

        # P(C=1 | Y=0, A=a)
        fpr = (
            data.fp(group) / (data.fp(group) + data.tn(group))
            if (data.fp(group) + data.tn(group)) != 0
            else 0
        )
        fpr_dict[group] = fpr
    if one:
        tpr_parity = _compute_global_parity(tpr_dict)
        fpr_parity = _compute_global_parity(fpr_dict)

        return tpr_parity, fpr_parity
    return tpr_dict, fpr_dict

def predictive_value_parity(expected=None, predicted=None, sensitive_features=None, data=None, one=True):
    """
    one - return values for whole dataset
    """
    if data is None:
        data = DataFrame(expected, predicted, sensitive_features)

    ppv_dict = {}
    npv_dict = {}

    for group in data.groups():

        # PPV (Positive Predictive Value): P(Y=1 | C=1, A=a)
        # PPV = TP / (TP + FP)
        ppv = (
            data.tp(group) / (data.tp(group) + data.fp(group))
            if (data.tp(group) + data.fp(group)) > 0
            else 0
        )
        ppv_dict[group] = ppv

        # NPV (Negative Predictive Value): P(Y=0 | C=0, A=a)
        # NPV = TN / (TN + FN)
        npv = (
            data.tn(group) / (data.tn(group) + data.fn(group))
            if (data.tn(group) + data.fn(group)) > 0
            else 0
        )
        npv_dict[group] = npv

    if one:
        ppv_parity = _compute_global_parity(ppv_dict)
        npv_parity = _compute_global_parity(npv_dict)

        return ppv_parity, npv_parity

    return ppv_dict, npv_dict
