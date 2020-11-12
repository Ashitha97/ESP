"""
Classes and methods for Getting various metrics
"""
# Base Imports
import pandas as pd
import numpy as np
# Card Imports
from imblearn.over_sampling import SMOTE
# Sklearn
from sklearn.metrics import jaccard_score, precision_recall_fscore_support, f1_score, recall_score, precision_score, \
    hamming_loss, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from skmultilearn.problem_transform import LabelPowerset


# Functions for Some quick Metrics

def quick_metrics(y_true, y_predicted, classes):
    # Weighted
    sc = np.array(precision_recall_fscore_support(y_true, y_predicted, average='weighted'), dtype=np.float32).round(
        4) * 100
    print('Weighted Metrics')
    print('Precision : {:.2f}\nRecall: {:.2f}\nF-score: {:.2f}'.format(sc[0], sc[1], sc[2]))

    # macro
    print('\nMacro Metrics')
    sc_macro = np.array(precision_recall_fscore_support(y_true, y_predicted, average='macro'), dtype=np.float32).round(4) * 100
    print('Precision : {:.2f}\nRecall: {:.2f}\nF-score: {:.2f}'.format(sc_macro[0], sc_macro[1], sc_macro[2]))

    # Classification Report
    print("\nClassification Report")
    print(classification_report(y_true,
                                y_predicted,
                                target_names=classes))
    return None


class MultiClassMetrics:

    @staticmethod
    def baseline_metrics(x, y, model):
        """
        This function will give a classification report in addition to a class based metrics
        :param x: Feature Vector
        :param y: Label Vector (Output Vector)
        :param model: Model
        :return: None
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.3,
                                                            stratify=y,
                                                            random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        sc = np.array(precision_recall_fscore_support(y_test, y_pred, average='weighted'), dtype=np.float32).round(
            4) * 100
        print('Weighted Metrics')
        print('Precision : {:.2f}\nRecall: {:.2f}\nF-score: {:.2f}'.format(sc[0], sc[1], sc[2]))

        print('\nMacro Metrics')
        sc_macro = np.array(precision_recall_fscore_support(y_test, y_pred, average='macro'), dtype=np.float32).round(
            4) * 100
        print('Precision : {:.2f}\nRecall: {:.2f}\nF-score: {:.2f}'.format(sc_macro[0], sc_macro[1], sc_macro[2]))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        # print("\nConfusion Matrix:")
        # print(confusion_matrix(y_test, pred))
        # print()

        return None

    @staticmethod
    def cv_validation(x, y, pipeline, n=3):
        """
        Cross-Validate The algorithm
        :param x: Features
        :param y: Output
        :param pipeline: Algorithm
        :param n: No of Iterations (int)
        :return: DataFrame with CV results
        """

        # Scoring
        scorer = {
            'F-Score_wt': 'f1_weighted',
            'Precision_wt': 'precision_weighted',
            'Recall_wt': 'recall_weighted',
            'F-Score_macro': 'f1_macro',
            'Precision_macro': 'precision_macro',
            'Recall_macro': 'recall_macro',
        }

        # Cross Validate
        scores = cross_validate(pipeline, x, y, scoring=scorer, cv=n)

        # Metrics
        cv_df = pd.DataFrame(index=range(n), data=scores)
        cv_df = cv_df.drop(columns=['fit_time', 'score_time']).round(4)
        cv_df.loc['Mean'] = cv_df.mean()
        cv_df.loc['STD'] = cv_df.std()
        cv_df.columns = list(scorer)
        cv_df = cv_df * 100

        return cv_df

    @staticmethod
    def kfold_validation(x, y, model, splits=5, verbose=0):
        verbosePrint = print if verbose != 0 else lambda *a, **k: None  # Have a Verbose Option

        skf = StratifiedKFold(n_splits=splits, shuffle=True)

        scores = []
        i = 1

        for train_index, test_index in skf.split(x, y):
            verbosePrint('Starting Iteration {}'.format(i))
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # get metrics
            verbosePrint('Weighted')
            sc1 = np.array(precision_recall_fscore_support(y_test, y_pred, average='weighted'), dtype=np.float32).round(
                4) * 100
            verbosePrint('\nPrecision : {:.2f}\nRecall: {:.2f}\nF-score: {:.2f}'.format(sc1[0], sc1[1], sc1[2]))

            verbosePrint('\nMacro')
            sc2 = np.array(precision_recall_fscore_support(y_test, y_pred, average='macro'), dtype=np.float32).round(
                4) * 100
            verbosePrint('\nPrecision : {:.2f}\nRecall: {:.2f}\nF-score: {:.2f}'.format(sc2[0], sc2[1], sc2[2]))

            scores.append(np.hstack([sc1, sc2]))

            i = i + 1
            verbosePrint('---------------------------\n')

        # K-fold Metrics
        kfold_df = pd.DataFrame(scores).dropna(axis=1)

        kfold_df.loc['Mean'] = kfold_df.mean()
        kfold_df.loc['STD'] = kfold_df.std()

        kfold_df.columns = ['Precision_wt', 'Recall_wt', 'F-score_wt', 'Precision_macro', 'Recall_macro',
                            'F-score_macro']

        verbosePrint('Overall K-Fold')
        verbosePrint(kfold_df.round(2))

        return kfold_df


class MultiLabelMetrics:
    """
    List of Static methods which provide metrics
    """

    @staticmethod
    def label_specific_metrics(y_true, y_pred, classes=None, count_df=None):
        """
        Gives label Specific Metrics
        :param y_true: True Labels
        :param y_pred: Predicted Labels
        :param classes: List of classes. Default=None. if provided an index of Labels will be present
        :param count_df: A DataFrame(Or Series) with the value counts of each class.
                         If classes are provided, counts_df should have the same indices as the classes
        """

        metrics = pd.DataFrame(data={
            'Jaccard Score': jaccard_score(y_true, y_pred, average=None),
            'Precision': precision_score(y_true, y_pred, average=None),
            'Recall': recall_score(y_true, y_pred, average=None),
            'F1-Score': f1_score(y_true, y_pred, average=None)
        }).round(2)

        if classes is not None:
            metrics.loc[:, 'Labels'] = classes
            metrics.set_index('Labels', inplace=True)

        if count_df is not None and classes is not None:
            metrics = pd.concat([metrics, count_df], axis=1)  # classes and count_df should match with indixes
        elif count_df is not None and classes is None:
            print("Count Values will not be used as class values in order, not provided")

        return metrics

    @staticmethod
    def averaged_scores(y_true, y_pred):
        """
        Gives averaged out scores
        :param y_true: True Labels
        :param y_pred: Predicted Labels
        """
        metric_types = ['weighted', 'macro', 'micro', 'samples']
        metric_df = pd.DataFrame(index=['Precision', 'Recall', 'F-Score', 'Jaccard-Score'], columns=metric_types)

        for types in metric_types:
            mt = list(precision_recall_fscore_support(y_true, y_pred, average=types))
            mt = [i for i in mt if i]  # Remove nan
            mt.append(jaccard_score(y_true, y_pred, average=types))

            metric_df.loc[:, types] = mt

        metric_df = metric_df * 100
        metric_df = metric_df.round(2)
        return metric_df


class MultiLabelAlgoTesting(MultiLabelMetrics):

    def __init__(self, x, y, model, counts_df=None, cl=None):
        self.X = x
        self.Y = y
        self.model = model
        self.counts_df = counts_df
        self.cl = cl

    @staticmethod
    def oversample(X, Y, thresh):
        """
        Oversample those group of multi labels that fall below  threshold. They are oversampled to the threshold.
        Note: Allow for a versbose option which will show df with resampled values
        :param X: Features
        :param Y: MultiLabels (One Hot encoded). Can be some other way was well, not tested yet.
        :param thresh: Threshold (A percentage value)
        :return: X_res, y_res
        """
        lp = LabelPowerset()  # To get singular labels from multi label gps
        yt = lp.transform(Y)
        count_info = pd.DataFrame(data={
            'gps': np.unique(yt, return_counts=True)[0],
            'totalVal': np.unique(yt, return_counts=True)[1],
            'pctVal': np.round(np.unique(yt, return_counts=True)[1] / np.shape(yt)[0] * 100, 2)
        })

        gps = count_info.loc[count_info.pctVal <= thresh, 'gps']  # Gps which are below the threshold
        n = np.round(np.shape(yt)[0] * thresh / 100, 0)  # The number to which they will be oversampled to
        smote_dict = {gp: int(n) for gp in gps}  # Smote dict to oversample

        sm = SMOTE(random_state=55, sampling_strategy=smote_dict)
        X_res, yt_res = sm.fit_resample(X, yt)
        y_res = lp.inverse_transform(yt_res).A  # '.A' to convert the sparse matrix into a binaraized form

        return X_res, y_res

    def kfold_validation(self, splits, oversample_thresh=0, verbose=0):
        """
        Run a stratified Split and get validation metrics
        :param oversample_thresh: Threshhold upto which label groups need to be resampled. Default: 0 (No Resampling)
        :param splits: No of Splits (int)
        :param verbose: 0/1 (Default). Anything other than zero will give additional metrics while running the function
        :return: Metrics DataFrame for each spit
        """

        vprint = print if verbose != 0 else lambda *args, **kwargs: None

        skf = StratifiedKFold(n_splits=splits, shuffle=True)
        lp = LabelPowerset()  # We need to implement a labelPowerset to stratify our classes
        yt = lp.transform(self.Y)  # Transform it, Will hae to re-transform it going ahead

        scores = []
        i = 1

        for train_index, test_index in skf.split(self.X, yt):
            vprint("\nIteration: {}".format(i))

            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = yt[train_index], yt[test_index]
            y_train = lp.inverse_transform(y_train).A
            y_test = lp.inverse_transform(y_test).A

            if oversample_thresh > 0:
                x_train, y_train = self.oversample(x_train, y_train, oversample_thresh)

            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            vprint(self.averaged_scores(y_test, y_pred))
            #         vprint(label_specifc_metr)
            # get metrics
            sc = list(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
            sc = [i for i in sc if i]  # Remove nan
            sc.append(jaccard_score(y_test, y_pred, average='weighted'))
            sc.append(hamming_loss(y_test, y_pred))
            scores.append(sc)
            i = i + 1
            vprint('------------------------------------------')

        kfold_df = pd.DataFrame(scores).dropna(axis=1)

        kfold_df.loc['Mean'] = kfold_df.mean()
        kfold_df.loc['STD'] = kfold_df.std()

        kfold_df.columns = ['Precision', 'Recall', 'F-score', 'Jaccard Score', 'Hamming Loss']

        vprint('Overall K-Fold')
        vprint(kfold_df.round(4))

        return kfold_df

    def quick_test(self, split, oversample_thresh=0, verbose=0):
        """
        Using a basic train-test split get metrics
        :param split: The Test size while splitting
        :param oversample_thresh: Using Smote to oversamle labels (Default: 0 , int indicates threshold)
        :param verbose: Print out details
        :return: DF with Averaged out Scores, DF with class specific scores
        """
        vprint = print if verbose != 0 else lambda *args, **kwargs: None

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=split,
                                                            shuffle=True,
                                                            stratify=self.Y)

        if oversample_thresh > 0:
            x_train, y_train = self.oversample(x_train, y_train, thresh=oversample_thresh)
            vprint("Data Oversampled")

        self.model.fit(x_train, y_train)  # Train the model
        y_pred = self.model.predict(x_test)

        avg_scores = self.averaged_scores(y_test, y_pred)
        all_scores = self.label_specific_metrics(y_test, y_pred, classes=self.cl, count_df=self.counts_df)

        vprint("Averaged out Scores are: ")
        vprint(avg_scores)
        vprint("Class Specific Scores:")
        vprint(all_scores)

        return avg_scores, all_scores
