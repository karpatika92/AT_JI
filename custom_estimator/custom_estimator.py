
# coding: utf-8

# In[150]:


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

import numpy as np


# In[151]:


#start from the simplest part, the gini_impurity
def _gini_impurity(y_pred,y_true, threshold):
    """Computes the gini impurity of a split.
    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        Array of true classes, must be either 0 or 1.
    y_pred : ndarray, shape (n_samples,)
        Array of predicted probabilities, must be between 0 and 1.
    threshold : float
        The value for cutting y.
    Returns
    -------
    out : float
        gini impurity
        
    References
    ----------
    Wikipedia - Decision Tree Learning
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    
    A Simple Explanation of Gini Impurity
    https://victorzhou.com/blog/gini-impurity/
    """
    #check if y is indeed a probability vector
    if not (np.max(y_pred)<=1)&(np.min(y_pred)>=0):
        raise ValueError("y_pred must be probabilities")
    
    if len(np.unique(y_true))!=2:
        raise ValueError("y_true must be binary labels")
    
    #now split the y vector and calculate the gini impurity by the wikipedia formula
    #source: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    #source 2:https://victorzhou.com/blog/gini-impurity/
    y_label_predicted=np.where(y_pred>threshold,1,0)
    #then get the predicted classes and the true classes
    class_dict={}
    #check if both labels exist
    if len(np.unique(y_label_predicted))==2:
        for label in [0,1]:
            #print(label)
            class_dict[label]={}
            class_dict[label]["prob_1"]=np.mean(y_true[y_label_predicted==label])
            class_dict[label]["weigth"]=np.sum(y_label_predicted==label)/len(y_label_predicted)
            class_dict[label]["gini"]=1-(class_dict[label]["prob_1"]*class_dict[label]["prob_1"]+                                         (1-class_dict[label]["prob_1"])*(1-class_dict[label]["prob_1"]))
        #print(class_dict)
        out=class_dict[0]["weigth"]*class_dict[0]["gini"]+class_dict[1]["weigth"]*class_dict[1]["gini"]
    else:
        #if there is only one label after the split, the gini must be calculated for the whole dataset
        prob_1=np.mean(y_true)
        out=1-(prob_1*prob_1+(1-prob_1)*(1-prob_1))
    
    return out


# In[152]:


#next, we create the thresholdbinarizer
#this class will take as an input y_true and y_pred, and will cycle through all possible threshold values to find the best split
class ThresholdBinarizer(BaseEstimator,TransformerMixin):
    """
    ThresholdBinarizer custom transformer
    
    The class implements a solution to find the optimal splitting threshold  in a vector
    of predicted probabilities, while knowing the true labels. The metric used for splitting is 
    the gini impurity
    
    Parameters
    ----------
    There are no parameters, only the predicted probabilities and the true classes are required
    
    Attributes
    ----------
    optimal_threshold : The calculated optimal threshold, that minimizes the gini impurity
    
    optimal_gini : The lowest achievable gini impurity
    
    Example
    --------
    from sklearn.datasets import load_breast_cancer
    X,y = load_breast_cancer(return_X_y=True)

    clf = LogisticRegression(random_state=0).fit(X, y)
    y_pred = clf.predict_proba(X)[:,0]

    binarizer=ThresholdBinarizer()
    binarizer.fit(y_pred,y)
    y_labels=binarizer.transform(y_pred)

    
    """
    def __init__(self):
        """
        Called when initializing the binarizer
        """
        
    def fit(self, y_pred,y_true,step=1000):
        """
        This should fit transformer. The purpose here is to find the optimal threshold to  cut the predicted probabilities.
        
        Parameters
        -----------
        y_pred : predicted probabilities for the classes
        
        y_true : the true labels of the instances
        
        step : the optimizer checks every k/step rationals for the gini split for k=0 to step.
        
        Returns
        --------
        self : object
        
        Notes
        -----------
        This step optimizes the treshold by trying every values with a 1/step on the [0,1] interval. 
        """
        
        self.optimal_threshold=0
        self.optimal_gini=_gini_impurity(y_pred,y_true, 0)
        print("STARTING GINI {}".format(self.optimal_gini))
        
        for threshold_ in np.arange(step)/step:
            _gini=_gini_impurity(y_pred,y_true, threshold_)
            #test if split is better than current best
            if _gini<self.optimal_gini:
                self.optimal_gini=_gini
                self.optimal_threshold=threshold_
        print("OPTIMAL THRESHOLD: {}".format(self.optimal_threshold))
        print("BEST GINI: {}".format(self.optimal_gini))

        return self
   
    def transform(self,y_pred):
        """This fitted classifier cuts the input vector accordng to the previously
        calculated optimal threshold
        
        Parameters
        -----------
        y_pred : the predicted probabilities
        
        Returns
        -----------
        out : array (n_samples,)
            an array of class labels
        
        """
        
        
        #check if the transformer was fitteed
        try:
            getattr(self, "optimal_threshold")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before transforming data!")
        #cut according to threshold
        out = np.where(y_pred>self.optimal_threshold,1,0)
        return out
        


# In[156]:


class custom_estimator(BaseEstimator, ClassifierMixin):  
    """A classifier I built for Aliz Tech job interview
    The purpose of this estimator is to fit a logistic regression on a binary classification task,
    then predict probabilities, and use a ThresholdBinarizer to assign labels according to the
    optimal minimum Gini impurity
    
    Parameters
    ----------
    estimator : an instance of a LogisticRegression from sklearn, custom properties can be defined
    
    Attributes
    ----------
    fitted_ : True, if the fit method has been executed
    
    estimator : the estimator passed to the _init_ method, defaults to LogisticRegression()
    
    y_pred : predicted probabilities by the estimator
    
    _binarizer : the ThresholdBinarizer instance to assign class labels
    

    Example
    --------
    from sklearn.datasets import load_breast_cancer
    X,y = load_breast_cancer(return_X_y=True)


    myEstimator=custom_estimator()
    myEstimator.fit(X,y)
    myEstimator.predict(X)    
    
    """

    def __init__(self, estimator=LogisticRegression()):
        """
        Called when initializing the classifier
        
        Parameters
        -----------
        
        estimator : the LogisticRegression instance used for restimating the outcome probabilities
        
        """
        self.estimator = estimator


    def fit(self, X, y_true):
        """
        This method fits the classifier to the inputdata features (X) and the true labels (y_true).
        
        Parameters
        ----------
        
        X : array (n_samples,n_features), array of features used for predicting the outcome
        
        y_true : array (n_samples,); array of labels for training examples
        
        Returns
        --------
        self : object

        """
        assert isinstance(self.estimator,LogisticRegression),"Estimator must be an instance of LogisticRegression"

        assert (len(np.unique(y_true))==2), "Labels must be binary"
        self.y_true=y_true
        self.estimator.fit(X,self.y_true)
        self.fitted_=True
        return self

    def predict_proba(self, X):
        """
        This method predicts probabilities
        
        Parameters
        ----------
        
        X : array (n_samples,n_features), array of features used for predicting the outcome
        
        
        Returns
        --------
        y_pred : array (n_samples,); vector of predicted probabilities

        """

        try:
            getattr(self, "fitted_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        
        self.y_pred=self.estimator.predict_proba(X)[:,0]
        
        #now binarize the prediction
        return self.y_pred
    
    def predict(self, X):
        """
        This method assigns class labels using ThresholdBinarizer to minimize gini impurity
        
        Parameters
        ----------
        
        X : array (n_samples,n_features), array of features used for predicting the outcome
        
        
        Returns
        --------
        self.y_labels : array (n_samples,); vector of predicted labels

        """

        try:
            getattr(self, "fitted_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        
        try:
            getattr(self, "y_pred")
        except AttributeError:
            self.y_pred=self.estimator.predict_proba(X)[:,0]
        
        self._binarizer=ThresholdBinarizer()
        self._binarizer.fit(self.y_pred,self.y_true)
        self.y_labels=self._binarizer.transform(self.y_pred)
        return self.y_labels

