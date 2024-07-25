from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


def match_C_to_number_of_features(estimator_class, number_of_features, X, y, estimator_params={}, parameter='C', max_c=0.9999):
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
        Name of the control parameter, default C.
    max_c : float
        Maximum value of C parameter, default 0.9999.

    Returns
    -------
    C
        The value of a control parameter to obtain a set number of features on a given set of data.
    """
    c = 0.5
    c_below = 0.0001
    c_above = max_c
    while True:
        estimator_params[parameter] = c
        selector = SelectFromModel(
            estimator = estimator_class(**estimator_params),
            threshold = 1e-8,
            importance_getter = "auto"
        )
        selector.fit(X, y)
        nof = len(selector.get_support(indices=True))
        #print(nof, C)
        if nof > number_of_features:
            if c < c_above:
                c_above = c
                c = (c_above + c_below) / 2
            else:
                break
        elif nof < number_of_features:
            if c > c_below:
                c_below = c
                c = (c_above + c_below) / 2
            else:
                break
        else:
            break
    return c


def create_selectors(estimators_and_params, X, y, features_nums, train_size=0.8):
    """
    Create selectors for use to given dataset X,y to select given number of features.

    Parameters
    ----------
    estimators_and_params : dict[str:(class,dict)]
        Estimators base classes with parameters.
    X,y
        Dataset.
    features_nums : list[int]
        Number of features expected in the selection.
    train_size : float, default 0.8
        The assumed size (fraction) of the training set.
        Needed to determine the value of the parameter C to select a given number of features.
    """
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    selectors = {}
    for estimator_name, (estimator_class, params) in estimators_and_params.items():
        for features_num in features_nums:
            c = match_C_to_number_of_features(estimator_class, features_num, X_train, y_train, params, max_c=1.0)
            if c == 1.0:
                c = match_C_to_number_of_features(estimator_class, features_num, X_train, y_train, params, max_c=10.0)
            params['C'] = c
            estimator = estimator_class(**params)
            selectors[f"{estimator_name}_f{features_num}"] = SelectFromModel(estimator = estimator, threshold = 1e-8, importance_getter = "auto")
    return selectors