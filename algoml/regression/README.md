# Regression

**Regression**, in its purest sense, is the prediction of a **continuous** label i.e. it is numerical and can be any value.  Regression is a model-based solution and is supervised.  It is generally simple in concept, so this will be a great introduction.

More specifically, **linear regression** is regression based on a linear function.  Note that the function does not necessarily have to be in two-dimensional space.  Linear models are in the form:

<img src="https://bit.ly/2YcRXnL" align="center" border="0" alt="\hat{y}=\theta_0+\theta_1x_1+\theta_2x_2+\dots+\theta_nx_n=h_\theta(x)=\theta^Tx" width="382" height="22" />.

To find the theta to minimizes loss, utilize the normal equation:

<img src="https://bit.ly/3f36pp1" align="center" border="0" alt="\hat\theta = (X^TX)^{-1}X^Ty" width="136" height="24" />.

On the other hand, **non-linear regression** is regression based on a non-linear function.  One widely used technique is **polynomial regression** where we fit the model to a polynomial function.  The idea is a bit more complex, but we can simply add powers of each feature to the feature itself to create a new feature and linearize the data.

However, knowing what power to go to might seem very difficult as a power too high will overfit the data and too low might underfit the data.  If we graph the bias vs variance, we can look at the bias-variance tradeoff to potentially find a good value for the hyperparameter.

To reduce the issue of overfitting, models are **regularized**.  There are many types of regularization of linear models:

-   **Ridge Regression**: Utilizes the L2 weight as a regularizer:
    -   <img src="https://bit.ly/30ouMtg" align="center" border="0" alt="J(\theta)=MSE(\theta)+\frac{\alpha}{2}\sum_{i=1}^n\theta_i^2" width="199" height="50" />
-   **Least Absolute Shrinkage Operator Regression (LASSO)**: Utilizes the L1 weight as regulizer.  This technique usually eliminates weights of the least important feature.
    -   <img src="https://bit.ly/2UjxjRJ" align="center" border="0" alt="J(\theta)=MSE(\theta)+\alpha\sum_{i=1}^n\mathopen|\theta_i\mathclose|" width="199" height="50" />
-   **Elastic Net**: The middle ground between ridge and lasso.
    -   <img src="https://bit.ly/2BK5g7r" align="center" border="0" alt="J(\theta)=MSE(\theta)+r\alpha\sum_{i=1}^n\mathopen|\theta_i\mathclose|+\frac{1-r}[2}\alpha\sum_{i=1}^n\theta_i^2" width="339" height="50" />
-   **Early Stopping**: Stopping gradient descent early.

Another type of regression, **logistic regression**, is for classification purposes by estimating the probability that an instance belongs to some class.  The probability is defined as follows:

<img src="https://bit.ly/2YedQ6b" align="center" border="0" alt="\hat{p}=h_\theta(x)=\sigma(\theta^Tx)" width="158" height="22" />.

where sigma is the **sigmoid function**:

<img src="https://bit.ly/2YhIwTU" align="center" border="0" alt="\sigma(t)=\frac{1}{1+exp(-t)}" width="157" height="46" />.

The cost to minimize is called the **log loss**:

<img src="https://bit.ly/3dLUYSx" align="center" border="0" alt="J(\theta}=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(\hat{p}^{(i)})+(1-y^{(i)})log(1-\hat{p}^{(i)})]" width="406" height="50" />.

Logistic Regression can be transitioned to multiclass classification by utilizing the **softmax function** defined as:

<img src="https://bit.ly/2YcxnUo" align="center" border="0" alt="\hat{p}_k=\sigma(s(x))_k=\frac{exp(s_k(x))}{\sum_{j=1}^Kexp(s_j(x))}" width="262" height="56" />.

Equivalently, the cost function to minimize is called the **cross-entropy**:

<img src="https://bit.ly/2AQ2MEg" align="center" border="0" alt="J(\sigma)=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)}log(\hat{p}_k^{(i)})" width="239" height="53" />.

For the linear regression problem, the California Census data is utilized to predict house prices from a set of features.

When defining the problem, data is first loaded into the system trivially.

To preprocess the data, missing values can either be removed completely or in this case, the average or median replaces the value.  Any categorical features are converted to one-hot encoding to be more meaningful.  Attributes are combined and a values are scaled.

When segregating the data into a training and test set, the test set is created such that it is non-random and the model will not be able to eventually see the test set (through multiple iterations).  The data is stratified based on income to be more representative of the population.

To evaluate the solution, a variety of performance measures exists:

-   **Root Mean Square (RMSE)**: Corresponds to the Euclidean Norm (**L2 Error**).  This performs very well if outliers are very rare (such as in a normal curve).
    -   <img src="https://bit.ly/2A6oSlZ" align="center" border="0" alt="RMSE =\sqrt{\dfrac{1}{m} \sum_{i=1}^m (h(x^{i})-y^{(i)})^2 }" width="244" height="68" />
-   **Mean Absolute Error (MAE)**: Corresponds to the Manhattan Norm (**L1 Error**).  This performs better than RMSE if outliers are more prevalent.
    -   <img src="https://bit.ly/2MIaSl2" align="center" border="0" alt="MAE =\sqrt{\dfrac{1}{m} \sum_{i=1}^m \mathopen|h(x^{i})-y^{(i)}\mathclose| }" width="219" height="68" />

Note that the model can be enhanced by utiling a **Random Forest** and **Grid Search** or **Randomized Search** to find hyperparameters.
