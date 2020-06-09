# SVM

**Support Vector Machines (SVM)** are ML models with the ability to perform linear or nonlinear classification and regression.  They are exceptional for classification of small-medium sized datasets.

**Linear SVM** is the basics to all SVMs and can be defined as finding the largest possible margin, called the **large margin classification** to distinguish classes.  Finding the largest margin causes the model to overfit the data, called **hard margin classification**.  Therefore, **soft margin classification** is utilized instead: create a large enough margin that is not sensitive to outliers (limit **margin violations**).

**Nonlinear SVMs** are basically identical to linear SVMs except an additional step is added to the process.  Before any of the margins are created, another feature is added to the dataset that is equal to a polynomial of the original value thus increasing the number of dimensions but making the problem linear.  This is called the **polynomial kernel**.  The degree of polynomial to add is another hyperparameter, but for SVMs, the **kernel trick** is utilized to achieve the affect of having to many polynomials.  Another technique is the **Gaussian RBF Kernel** that moves the data into a normalized set.

For regression, we simply use the line for the margin as the regression function.  Note that these problems can be computationally expensive to solve so lots of **quadratic programming** (field of math) is used to correctly minimize the loss.  To find a better lower bound, for example, the **dual problem** is created from the **primal** and is solved since it is easier.
