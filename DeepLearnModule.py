import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

def ModelHist_plot(hist,plot1,plot2,leg1,leg2):
    plt.figure()
    plt.plot(hist.history[plot1])
    plt.plot(hist.history[plot2])
    plt.legend([leg1, leg2])
    plt.xlabel('epochs')
    plt.show()

def Model_Analysis(y_true,y_pred):
    cr = classification_report(y_true,y_pred)
    print(cr)
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred)
    plt.show()


def boxplot(df,con_col,nrows=1,ncols=1, size1=(30,40)):
    fig, ax = plt.subplots(nrows, ncols, figsize=size1)
    df[con_col].plot.box(layout=(nrows, ncols), 
                subplots=True, 
                ax=ax, 
                vert=False, 
                sharex=False)
    plt.tight_layout()
    plt.show()

