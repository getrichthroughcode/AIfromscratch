# Logistic Regression

## What is logistic Regression

**Logistic Regression** is a linear model used for binary classification. It predicts the probability that a given input belongs to a particular class.

## Assumptions

**Logistic Regression**, like any statistical model, relies on several assumptions to function correctly and provide reliable results. These assumptions are not as strict as those for **Linear Regression**, but they are very important to be familiar with. Here are the key assumptions:
 - **Linearity of the logit** :
    The relationship between the independant variables (`features`) and the log-odds of the dependant variable (`logit`) is linear.
     Logistic regression models the probability of the outcome as a function of the linear combination of the features. Mathematically, it assumes:
   $$
    Logistic formula here 
   $$ 
This means that the log-odds of the outcome are linearly related to the predictor variables.
- **Independant of Errors**
The `observations`hence the errors are independant of each other.
**Logistic Regression** assumes that the outcome of one observation does not influence the outcome for another. This is particularly important in `time-series`data or `clustered` data (like repeated measurements on the same subjects). 
- **No Multicolinearity**
The independant variable are not highly correlated with each other. 
-



