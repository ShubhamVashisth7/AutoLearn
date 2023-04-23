import os
import warnings
import numpy as np
import pandas as pd
import pandas.errors
from sklearn.tree import tree
from sklearn.metrics import f1_score
from scipy.stats.stats import pearsonr
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform, pdist
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Ridge, RandomizedLasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from clean import clean
warnings.filterwarnings('ignore')
RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)
N_FOLDS = 5

clf = Ridge(alpha=1.0)
cnt = 0
ans = []


def original_ig(ress, test, labels):  # ress is training data
    x, y = ress.shape
    names = np.arange(y)

    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress, labels)

    # print "Features sorted by their scores according to the scoring function - mutual information gain:"
    original_features = sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
                                   names), reverse=True)

    finale = []
    for i in range(0, len(original_features)):
        r, s = original_features[i]
        if r > 0:  # This is eta-o
            finale.append(s)

        # finale.append(s)

    print("Selected features after O + IG:")
    global len_orig_ig
    len_orig_ig += len(finale)
    print(len(finale))
    dataset1 = np.zeros((len(ress), len(finale)), dtype=float)
    dataset3 = np.zeros((len(test), len(finale)), dtype=float)
    dataset1 = ress[:, finale]
    dataset3 = test[:, finale]
    # dataset3=test.iloc[:,finale]

    if os.path.exists("sonar_original_ig_testfeatures.csv"):  # Name of Ouput file generated
        os.remove("sonar_original_ig_testfeatures.csv")
    if os.path.exists("sonar_original_ig_trainfeatures.csv"):  # Name of Ouput file generated
        os.remove("sonar_original_ig_trainfeatures.csv")

    with open("sonar_original_ig_testfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset3, delimiter=",", fmt="%s")
    with open("sonar_original_ig_trainfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset1, delimiter=",", fmt="%s")


def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def dependent(x, th1, fold):
    ans = []
    ans1 = []
    m, n = x.shape
    cnt = 0
    cnt1 = 0
    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                a, b = pearsonr(x[:, i][:, np.newaxis], x[:, j][:, np.newaxis])
                if (distcorr(np.array(x[:, i]), np.array(x[:, j])) >= th1):
                    a1 = i, j
                    ans.append(a1)
                    cnt = cnt + 1
                elif (distcorr(np.array(x[:, i]), np.array(x[:, j])) > 0 and distcorr(np.array(x[:, i]),
                                                                                      np.array(x[:, j])) < 0.7):
                    zz = i, j
                    ans1.append(zz)
                    cnt1 = cnt1 + 1

        # print(i)
    if os.path.exists('sonar_linear_correlated_{}.csv'.format(fold)):  # Name of Ouput file generated
        os.remove('sonar_linear_correlated_{}.csv'.format(fold))
    if os.path.exists('sonar_nonlinear_correlated_{}.csv'.format(fold)):  # Name of Ouput file generated
        os.remove('sonar_nonlinear_correlated_{}.csv'.format(fold))

    np.savetxt("sonar_linear_correlated_{}.csv".format(fold), ans, delimiter=",", fmt="%.5f")
    np.savetxt("sonar_nonlinear_correlated_{}.csv".format(fold), ans1, delimiter=",", fmt="%s")

    print("This is fold no - {}".format(fold))
    print("Number of linear correlated features are:")
    print(cnt)
    print("Number of non linear correlated features are:")
    print(cnt1)


def linear(TR, TST, fold):
    a, b = TR.shape
    c, d = TST.shape
    try:
        dataset = pd.read_csv('sonar_linear_correlated_{}.csv'.format(fold), header=None)
    except pandas.errors.EmptyDataError:
        print 'no linearly correlated features'
        return

    val = dataset.as_matrix(columns=None)
    aa, bb = val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((a, len(ans)), dtype=float)
    predicted_test = np.zeros((c, len(ans)), dtype=float)
    predicted_train_final = np.zeros(2 * (a, len(ans)), dtype=float)
    predicted_test_final = np.zeros(2 * (c, len(ans)), dtype=float)
    predicted_train_error = np.zeros((a, len(ans)), dtype=float)
    predicted_test_error = np.zeros((c, len(ans)), dtype=float)

    for j in range(0, aa):
        rr, ss = np.array(TR[:, (int)(val[j][0])][:, np.newaxis]), np.array(TR[:, (int)(val[j][1])])
        tt, uu = np.array(TST[:, (int)(val[j][0])][:, np.newaxis]), np.array(TST[:, (int)(val[j][1])])
        y_train = clf.fit(rr, ss).predict(rr)[:, np.newaxis]
        y_test = clf.fit(rr, ss).predict(tt)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

        dd = ss[:, np.newaxis]
        ee = uu[:, np.newaxis]
        diff_train = (dd - y_train)
        diff_test = (ee - y_test)
        predicted_train_error = np.hstack([predicted_train_error, diff_train])
        predicted_test_error = np.hstack([predicted_test_error, diff_test])
        # predicted_test=np.hstack([predicted_test,clf.coef_*TST[:,val[j][1]][:, np.newaxis]+clf.intercept_])
        # predicted_train=np.hstack([predicted_train,clf.coef_*TR[:,val[j][1]][:, np.newaxis]+clf.intercept_])
    predicted_train_final = np.hstack([predicted_train, predicted_train_error])
    predicted_test_final = np.hstack([predicted_test, predicted_test_error])
    # Saving constructed features finally to a file

    if os.path.exists("sonar_related_lineartest_{}.csv".format(fold)):  # Name of Ouput file generated
        os.remove("sonar_related_lineartest_{}.csv".format(fold))

    if os.path.exists('sonar_related_lineartrain_{}.csv'.format(fold)):  # Name of Ouput file generated
        os.remove('sonar_related_lineartrain_{}.csv'.format(fold))

    with open("sonar_related_lineartest_{}.csv".format(fold), "wb") as myfile:
        np.savetxt(myfile, predicted_test_final, delimiter=",", fmt="%s")
    with open("sonar_related_lineartrain_{}.csv".format(fold), "wb") as myfile:
        np.savetxt(myfile, predicted_train_final, delimiter=",", fmt="%s")


def nonlinear(TR, TST, fold):
    a, b = TR.shape
    c, d = TST.shape
    dataset = pd.read_csv('sonar_nonlinear_correlated_{}.csv'.format(fold), header=None)
    val = dataset.as_matrix(columns=None)
    aa, bb = val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train = np.zeros((a, len(ans)), dtype=float)
    predicted_test = np.zeros((c, len(ans)), dtype=float)
    predicted_train_final = np.zeros(2 * (a, len(ans)), dtype=float)
    predicted_test_final = np.zeros(2 * (c, len(ans)), dtype=float)

    predicted_train_error = np.zeros((a, len(ans)), dtype=float)
    predicted_test_error = np.zeros((c, len(ans)), dtype=float)

    svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                          kernel_params=None)

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    for j in range(0, aa):
        rr, ss = np.array(TR[:, (int)(val[j][0])][:, np.newaxis]), np.array(TR[:, (int)(val[j][1])])
        tt, uu = np.array(TST[:, (int)(val[j][0])][:, np.newaxis]), np.array(TST[:, (int)(val[j][1])])

        y_train = svr_rbf.fit(rr, ss).predict(rr)[:, np.newaxis]
        y_test = svr_rbf.fit(rr, ss).predict(tt)[:, np.newaxis]
        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

        dd = ss[:, np.newaxis]
        ee = uu[:, np.newaxis]
        diff_train = (dd - y_train)
        diff_test = (ee - y_test)

        predicted_train_error = np.hstack([predicted_train_error, diff_train])
        predicted_test_error = np.hstack([predicted_test_error, diff_test])
        '''
        popt, pcov = curve_fit(curve,rr,ss)
        predicted_test=np.hstack([predicted_test,float(popt[0])*(TST[:,val[j][1]][:, np.newaxis]**2)+float(popt[1])*TST[:,val[j][1]][:,    np.newaxis]+float(popt[2])])
        predicted_train=np.hstack([predicted_train,float(popt[0])*(TR[:,val[j][1]][:, np.newaxis]**2)+float(popt[1])*TR[:,val[j][1]][:, np.newaxis]+float(popt[2])])
        '''
    predicted_train_final = np.hstack([predicted_train, predicted_train_error])
    predicted_test_final = np.hstack([predicted_test, predicted_test_error])

    if os.path.exists("sonar_related_nonlineartest_{}.csv".format(fold)):  # Name of Ouput file generated
        os.remove("sonar_related_nonlineartest_{}.csv".format(fold))

    if os.path.exists('sonar_related_nonlineartrain_{}.csv'.format(fold)):  # Name of Ouput file generated
        os.remove('sonar_related_nonlineartrain_{}.csv'.format(fold))

    # Saving constructed features finally to a file
    with open("sonar_related_nonlineartest_{}.csv".format(fold), "wb") as myfile:
        np.savetxt(myfile, predicted_test_final, delimiter=",")
    with open("sonar_related_nonlineartrain_{}.csv".format(fold), "wb") as myfile:
        np.savetxt(myfile, predicted_train_final, delimiter=",")


def stable(ress, test, labels):  # ress is training data
    x, y = ress.shape
    names = np.arange(y)
    rlasso = RandomizedLasso()
    rlasso.fit(ress, labels)

    # print "Features sorted by their scores according to the stability scoring function"
    val = sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                     names), reverse=True)

    print("len of val")  # newly constructed features
    print(len(val))
    global nc_val
    nc_val += len(val)

    finale = []
    for i in range(0, len(val)):
        r, s = val[i]  # 'r' represents scores, 's' represents column name
        if r > 0.1:  # This is eta for stability selection
            finale.append(s)

    print("Total features after stability selection:")
    print(len(finale))  # finale stores col names - 2nd, 4th etc of stable features.
    global stable_val
    stable_val += len(finale)

    dataset1 = np.zeros((len(ress), len(finale)), dtype=float)
    dataset3 = np.zeros((len(test), len(finale)), dtype=float)
    dataset1 = ress[:, finale]
    dataset3 = test[:, finale]
    # dataset3=test.iloc[:,finale]

    if os.path.exists("sonar_stable_testfeatures.csv"):  # Name of Ouput file generated
        os.remove("sonar_stable_testfeatures.csv")
    if os.path.exists("sonar_stable_trainfeatures.csv"):  # Name of Ouput file generated
        os.remove("sonar_stable_trainfeatures.csv")

    with open("sonar_stable_testfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset3, delimiter=",", fmt="%s")
    with open("sonar_stable_trainfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset1, delimiter=",", fmt="%s")

    # -----------------------------------------------------------------------------------
    # check the inter-feature dependence - 2nd phase of ensemble

    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress[:, finale], labels)

    # print "Features sorted by their scores according to the scoring function - mutual information gain:"
    feats = sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
                       names), reverse=True)

    ensemble_finale = []
    for i in range(0, len(feats)):
        r, s = feats[i]
        if (r > 0):  # This is eta-o
            ensemble_finale.append(s)

    print("Total features after 2 phase selection:")
    print(len(ensemble_finale))  # ensemble_finale stores col names further pruned in the 2nd phase of feature selection
    global ensemble_val
    ensemble_val += len(ensemble_finale)
    # print(ensemble_select)

    dataset2 = np.zeros((len(ress), len(ensemble_finale)), dtype=float)
    dataset4 = np.zeros((len(test), len(ensemble_finale)), dtype=float)
    dataset2 = ress[:, ensemble_finale]
    dataset4 = test[:, ensemble_finale]

    if os.path.exists("sonar_ensemble_testfeatures.csv"):  # Name of Ouput file generated
        os.remove("sonar_ensemble_testfeatures.csv")
    if os.path.exists("sonar_ensemble_trainfeatures.csv"):  # Name of Ouput file generated
        os.remove("sonar_ensemble_trainfeatures.csv")

    with open("sonar_ensemble_testfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset4, delimiter=",", fmt="%s")
    with open("sonar_ensemble_trainfeatures.csv", "wb") as myfile:
        np.savetxt(myfile, dataset2, delimiter=",", fmt="%s")


def rank(X1, y):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X1, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X1.shape[1]):
        #  print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        pass


if __name__ == '__main__':
    clean()
    df = pd.read_csv('lfe_datasets/Sonar.csv')
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    len_orig_ig = 0
    nc_val = 0
    stable_val = 0
    ensemble_val = 0

    # names = ['KNN', 'LR', 'SVM-L', 'SVM-P', 'RF', 'AB', 'NN', 'DT']
    # models = [KNeighborsClassifier(), LogisticRegression(), LinearSVC(), SVC(C=1.0, kernel='poly'),
    #           RandomForestClassifier(n_estimators=100), AdaBoostClassifier(), MLPClassifier(), tree.DecisionTreeClassifier()]
    # accuracy = {'KNN': 0, 'LR': 0, 'SVM-L': 0, 'SVM-P': 0, 'RF': 0, 'AB': 0, 'NN': 0, 'DT': 0}
    # f1_scores = {'KNN': 0, 'LR': 0, 'SVM-L': 0, 'SVM-P': 0, 'RF': 0, 'AB': 0, 'NN': 0, 'DT': 0}

    names = ['RF']
    models = [RandomForestClassifier(n_estimators=100)]
    f1_scores = {'RF': 0}

    for fold, (train_index, test_index) in enumerate(StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)):
        fold = fold + 1
        print 'fold-{}'.format(fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = X_train.as_matrix()
        X_test = X_test.as_matrix()
        y_train = y_train.as_matrix()
        y_test = y_test.as_matrix()

        original_ig(ress=X_train, test=X_test, labels=y_train)
        original_ig_X_train = pd.read_csv('sonar_original_ig_trainfeatures.csv', header=None).as_matrix()
        original_ig_X_test = pd.read_csv('sonar_original_ig_testfeatures.csv', header=None).as_matrix()

        dependent(x=original_ig_X_train, th1=0.7, fold=fold)
        linear(TR=original_ig_X_train, TST=original_ig_X_test, fold=fold)
        nonlinear(TR=original_ig_X_train, TST=original_ig_X_test, fold=fold)

        try:
            a1 = pd.read_csv('sonar_related_lineartest_{}.csv'.format(fold), header=None)
            print 'a1: {}'.format(np.shape(a1))
        except IOError:
            a1 = None
        try:
            a2 = pd.read_csv('sonar_related_lineartrain_{}.csv'.format(fold), header=None)
            print 'a2: {}'.format(np.shape(a2))
        except IOError:
            a2 = None
        try:
            a3 = pd.read_csv('sonar_related_nonlineartest_{}.csv'.format(fold), header=None)
            print 'a3: {}'.format(np.shape(a3))
        except IOError:
            a3 = None
        try:
            a4 = pd.read_csv('sonar_related_nonlineartrain_{}.csv'.format(fold), header=None)
            print 'a4: {}'.format(np.shape(a4))
        except IOError:
            a3 = None

        if a1 is None and a2 is None:
            r4 = a4.values
            r3 = a3.values
        else:
            r4 = np.hstack([a2, a4])
            r3 = np.hstack([a1, a3])

        print 'r4: {}'.format(np.shape(r4))
        print 'r3: {}'.format(np.shape(r3))

        scaler = StandardScaler().fit(r4)
        p2 = scaler.transform(r4)
        p1 = scaler.transform(r3)

        stable(ress=p2, test=p1, labels=y_train)

        f1 = pd.read_csv('sonar_ensemble_trainfeatures.csv', header=None)
        f2 = pd.read_csv('sonar_ensemble_testfeatures.csv', header=None)

        scaler = StandardScaler().fit(f1)
        e_f1 = scaler.transform(f1)
        e_f2 = scaler.transform(f2)

        x1X = np.hstack(
            [X_test, f2])  # original test features, selected by IG, f2 is feature space after ensemble selection.
        x2X = np.hstack([X_train, f1])

        scaler = StandardScaler().fit(x2X)  # Again normalization of the complete combined feature pool
        x2 = scaler.transform(x2X)  # note - when features need to be merged with R2R, we need to do normalization.
        x1 = scaler.transform(x1X)

        """
        y1Y = np.hstack([X_test, f2])
        y2Y = np.hstack([X_train, f1])

        scaler = StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
        y2 = scaler.transform(y2Y)  # note - when features need to be merged with R2R, we need to do normalization.
        y1 = scaler.transform(y1Y)

        st_f1 = pd.read_csv('sonar_stable_trainfeatures.csv', header=None)
        st_f2 = pd.read_csv('sonar_stable_testfeatures.csv', header=None)

        st_x1X = np.hstack([original_ig_X_test,
                            st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
        st_x2X = np.hstack([original_ig_X_train, st_f1])

        scaler = StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
        st_x2 = scaler.transform(
            st_x2X)  # note - when features need to be merged with R2R, we need to do normalization.
        st_x1 = scaler.transform(st_x1X)
        """

        for i in range(0, len(models)):
            models[i].fit(x2, y_train)
            y_out = models[i].predict(x1)
            f1 = f1_score(y_true=y_test, y_pred=y_out)
            print '{}: {}'.format(names[i], f1)
            f1_scores[names[i]] += f1
            # acc = models[i].score(x1, y_test)
            # accuracy[names[i]] += acc

        rank(x2, y_train)

        print 'fold-{} done.'.format(fold)
        print '{}'.format('_' * 80)

    # accuracy = {k: v / N_FOLDS for k, v in accuracy.items()}
    # print accuracy
    f1_scores = {k: v / N_FOLDS for k, v in f1_scores.items()}
    print f1_scores
    clean()

# TODO: 1. measure accuracy
# 2. use same models as kgfarm
# 3. save results as dataframes
# 4. test and drop categorical features