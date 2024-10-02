from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


def match_C_to_number_of_features(estimator_class, number_of_features, X, y, estimator_params={}, parameter='C', nof_tolerance=0,
                                  max_c=0.9999, verbose=False):
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
    nof_tolerance : int
        The tolerance relating to the number of features selected by the estimator.
        A value between 0 and 100. Default is 0. The value of parameter C will be chosen
        so that the estimator selects the given number of features +/- tolerance.
    max_c : float
        Maximum value of C parameter, default 0.9999.

    Returns
    -------
    C
        The value of a control parameter to obtain a set number of features on a given set of data.
    """

    def fit_selector_and_get_nof(c):
        estimator_params[parameter] = c
        selector = SelectFromModel(
            estimator = estimator_class(**estimator_params),
            threshold = 1e-8,
            importance_getter = "auto"
        )
        selector.fit(X, y)
        return len(selector.get_support(indices=True))
    
    c_below = 0.0001
    c_above = max_c

    nof_above = fit_selector_and_get_nof(c_above)
    if nof_above < number_of_features:
        return c_above
    
    nof_below = fit_selector_and_get_nof(c_below)
    if nof_below > number_of_features:
        return c_below

    while True:
        if verbose:
            print(f"{nof_below}:{c_below} - {nof_above}:{c_above}")
        c = (c_above-c_below) / (nof_above-nof_below) * (number_of_features-nof_below) + c_below
        nof = fit_selector_and_get_nof(c)
        if nof > round(number_of_features * (1.+nof_tolerance/100)):
            if c < c_above:
                c_above = c
                nof_above = nof
            else:
                break
        elif nof < round(number_of_features * (1.-nof_tolerance/100)):
            if c > c_below:
                c_below = c
                nof_below = nof
            else:
                break
        else:
            break
        if c_above - c_below < 1e-10:
            break
    return c


def create_selectors(estimators_and_params, X, y, features_nums, train_size=0.8, nof_tolerance=0):
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
    nof_tolerance : int
        The tolerance relating to the number of features selected by the estimator.
        A value between 0 and 100. Default is 0. The value of parameter C will be chosen
        so that the estimator selects the given number of features +/- tolerance.

    Returns
    -------
    selectors : dict[str:SelectFromModel]
        Group of selectors, single selector for each features_num.
    """
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    selectors = {}
    for estimator_name, (estimator_class, params) in estimators_and_params.items():
        for features_num in features_nums:
            c = match_C_to_number_of_features(estimator_class, features_num, X_train, y_train, params, nof_tolerance=nof_tolerance, max_c=1.0)
            if c > 0.999995:
                c = match_C_to_number_of_features(estimator_class, features_num, X_train, y_train, params, nof_tolerance=nof_tolerance, max_c=100.0)
            params['C'] = c
            estimator = estimator_class(**params)
            selectors[f"{estimator_name}_f{features_num}"] = SelectFromModel(estimator = estimator, threshold = 1e-8, importance_getter = "auto")
    return selectors