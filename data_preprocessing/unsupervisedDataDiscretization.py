import pandas as pd
import numpy as np
from scipy.stats import iqr


########################################### statistical discretization

def get_labels(bins):
    labels = []
    for bin_index in range(len(bins)+1):
        if bin_index == 0:
            label = '<=' + str(np.round(bins[bin_index],2))
        elif bin_index == len(bins):
            label = '>=' + str(np.round(bins[bin_index-1],2))
        else:
            label = str(str(np.round(bins[bin_index-1],2))) + ' - ' + str(np.round(bins[bin_index],2))
        labels.append(label)
    return labels

def get_bin_edges(cont_data,method='auto',IQRange = (0.1,0.99)):
    cols = cont_data.columns
    cols_bins = []
    cols_labels = []

    for col in cols:
        q1 = cont_data[col].quantile(IQRange[0])
        q2 = cont_data[col].quantile(IQRange[1])
        mask = cont_data[col].between(q1, q2, inclusive=True)
        values = cont_data.loc[mask, col].dropna().values
        _, bins = np.histogram(values,bins=method)

        labels = get_labels(bins)
        
        bins = [-np.Inf] + list(bins) + [np.Inf]
        
        cols_bins.append(bins)
        cols_labels.append(labels)
    return cols_bins,cols_labels

def get_descritized_data(cont_data,method='auto',IQRange = (0.1,0.99)):
    cols = cont_data.columns
    col_bins, cols_labels = get_bin_edges(cont_data,method=method,IQRange = IQRange)
    

class UnsupervisedDescritizer:
    def __init__(self):
        self.cols = []
        self.bins = []
        self.labels = []

    def fit(self, cont_data, method='auto',IQRange = (0.1,0.99)):
        self.cols = cont_data.columns
        self.bins,self.labels = get_bin_edges(cont_data=cont_data,method=method,IQRange = IQRange)

        return;

    def fit_transform(self, cont_data, method='auto', mapped=True,IQRange = (0.1,0.99)):
        self.cols = cont_data.columns
        self.bins,self.labels = get_bin_edges(cont_data=cont_data,method=method,IQRange = IQRange)

        i=0
        for col in self.cols:
            if mapped:
                cont_data[col] = pd.cut(x=cont_data[col],bins=self.bins[i],
                                        labels=self.labels[i])
            else:
                cont_data[col] = pd.cut(x=cont_data[col],bins=self.bins[i])
            i+=1

        return cont_data

    def transform(self, cont_data, mapped=True):
        i=0
        for col in self.cols:
            if mapped:
                cont_data[col] = pd.cut(x=cont_data[col],bins=self.bins[i],
                                        labels=self.labels[i])
            else:
                cont_data[col] = pd.cut(x=cont_data[col],bins=self.bins[i])
            i+=1

        return cont_data   


from sklearn import datasets


if __name__ == "__main__":
    iris = datasets.load_iris()
    iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

    print(iris.columns)
    UD = UnsupervisedDescritizer()
    UD.fit(cont_data=iris,IQRange=(0.1,0.9))
    print(UD.transform(cont_data=iris))


  


