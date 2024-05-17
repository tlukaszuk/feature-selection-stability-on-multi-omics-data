# feature-selection-stability-on-multi-omics-data
Article about stability of feature selection on multi-omics data

# Our experiments:
embedded feature selection: 4 methods

Exp 1:
    - find C per method per dataset to select similar (low=10,midium=25,high=100) number of features

Exp 2: 
    - 5-fold cross-validation vs 10-folds
    - stability,accuracy,selected features per method and dataset 4x15 x3C

Exp 3: (omics)
    - per subset: 4xmethods/3xC/1xcv/


# Based on article:
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04962-x
15 cancer multi-omics datasets
four filter methods, two embedded methods, and two wrapper methods
two classifiers: support vector machines and random forests

The accuracy, the AUC, and the Brier were used to evaluate the predictive performance. As an evaluation scheme, we used fivefold cross-validation repeated three times to measure the performance of each method on each dataset.

 p-values show the results of the Friedman tests

Best FS methods:
 - Minimum Redundancy Maximum Relevance method (mRMR)
 - permutation importance of random forests (RF-VI)

 Including the clinical information did not improve the predictive performance. 
 However, we did not prioritize the clinical information.

 The results of the Friedman test for performance differences between the methods were significant with the exception of the results for SVM....


 ‘nvar’ (10,100,1000,5000) denotes the number of selected features, ‘selsep’ whether the features were selected separately by data type, and ‘clivar’ whether clinical variables were included or not


 For all algorithms, the default parameter values in the respective R implementations were used if not indicated otherwise.

 In the cases of the wrapper methods and the embedded methods, the computation time becomes very large if the number of features is large, which is the case for multi-omics data. Therefore, before applying these methods, we used t-test based filtering to select the top 10% of features to reduce the computational consumption.