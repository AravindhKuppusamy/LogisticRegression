# Logistic Regression

### Problem Statement:

Implement a single Logistic Regressor.

### Description:

Logistic regression is a linear model. Ouptut is passed through a logistic function and squashed to get a value in the interval (0,1) as output. Usually, logistic regressors are used for binary classification by thresholding this output value (>= 0.5 is Class 1; otherwise Class 0). But logistic regression wouldn't solve the problem when the data is not linearly separable. It is analysed in this experiment.

------

### Execution of program:

**Main file name:** logreg.py

**Python version:** 3.5.1

To execute:

```powershell
python logreg.py
```
### Output:

1. **For Linearly Separable data:**

   ```powershell
    No. of Correctly classified samples      : 99
    No. of Incorrectly classified samples    : 0
   ```

   ![Linearly Separable data](https://github.com/AravindhKuppusamy/LogisticRegression/blob/master/output/ls.png)

   Here the data is linearly separable. So the logistic regression does a good job.

2. **For Non-linearly Separable data:**

   ```powershell
   No. of Correctly classified samples      : 142
   No. of Incorrectly classified samples    : 58
   ```

   ![Non-linearly Separable data](https://github.com/AravindhKuppusamy/LogisticRegression/blob/master/output/nls.png)

   Here the data is non-linearly separable. So we don't get a clear decision boundary by logistic regression.

------
