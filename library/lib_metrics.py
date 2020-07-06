"""
Classes and methods for Getting various metrics
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import cross_validate, train_test_split


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
