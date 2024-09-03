# Logistic Regression

## What is logistic Regression

**Logistic Regression** is a linear model used for binary classification. It predicts the probability that a given input belongs to a particular class.

## Assumptions

**Logistic Regression**, like any statistical model, relies on several assumptions to function correctly and provide reliable results. These assumptions are not as strict as those for **Linear Regression**, but they are very important to be familiar with. Here are the key assumptions:
 - **Linearity of the logit** :
    The relationship between the independant variables (`features`) and the log-odds of the dependant variable (`logit`) is linear.
     Logistic regression models the probability of the outcome as a function of the linear combination of the features. Mathematically, it assumes:
    $$
    \text{logit}(P(y=1|X)) = \log\left(\frac{P(y=1|X)}{1 - P(y=1|X)}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n
    $$

    This means that the log-odds of the outcome are linearly related to the predictor variables.

