from sklearn.feature_selection import SelectFromModel


def match_C_to_number_of_features(estimator_class, number_of_features, X, y, estimator_params={}, parameter='C'):
    """
    Match the parameter C to the expected number of features selected by the estimator.

    Parameters
    ----------
    estimator_class : class
    number_of_features : int
    X, y
        Dataset.
    estimator_params : dict
        Invariant estimator parameters.
    parameter
        Name of the control parameter, usually C.
    Returns
    -------
    C
        The value of a control parameter to obtain a set number of features on a given set of data.
    """
    C = 0.5
    C_below = 0.0001
    C_above = 0.9999
    while True:
        estimator_params[parameter] = C
        selector = SelectFromModel(
            estimator = estimator_class(**estimator_params),
            threshold = 1e-8,
            importance_getter = "auto"
        )
        selector.fit(X, y)
        nof = len(selector.get_support(indices=True))
        #print(nof, C)
        if nof > number_of_features:
            if C < C_above:
                C_above = C
                C = (C_above + C_below) / 2
            else:
                break
        elif nof < number_of_features:
            if C > C_below:
                C_below = C
                C = (C_above + C_below) / 2
            else:
                break
        else:
            break
    return C