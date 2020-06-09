# Basic Examples for Machine Learning Algorithms

The following modules contain code examples of multiple common machine learning algorithms.  This is intended for getting a basic understanding of machine learning concepts and ideas and how they can actually be implemented using higher level libraries, specifically scikit-learn and Tensorflow in Python.

Note that these examples were highly based off of Aurelien Geron's Hands-On Machine Learning with Scikit-Learn & Tensorflow book with the change of a lot of programming to follow more pythonic styling and introducing a more formal code base.

In this README, I will give notes on important topics related to each algorithm.

## 1 Introduction:

### Package Structure:

The top level directory (algoml) includes a subdirectory called algoml and another file called run.py.  The run.py file is an example of how to utilize the pipelines within the algoml package subdirectory.  Within the actual algoml package, the directories are categories of ML algorithms.  The backend directory is the exception as it includes common called used among the other directories.

To create your own ML application, this directory should not be followed as this is for learning purposes.  Instead look at the **cookie-cutter** package.

### Basic Terms:

The following are the basic terms that are always taught:

-   **Machine Learning (ML)**: Programming computers to _learn_ from data. Applying ML techniques to find patterns in big data, it is called **data mining**. ML is best utilized when:
    -   An existing solution requires constant changes.
    -   The problem is to complex to even find a solution.
-   **Human Supervision**: The involvement of human teaching to an ML application.
    -   **Supervised**: Data fed into the program has _all_ solutions already given.  These solutions are called **labels**.
    -   **Unsupervised**: Data fed into the program has _no_ solutions already given.
    -   **Semi-supervised**: Data fed into the program has _some_ solutions already given.
    -   **Reinforcement**: An agent is rewarded by acting in some simulated environment.
-   **Learning**: The act of a program utilizing data to solve a problem.
    -   **Online**: The system learns from data incrementally.  This can happen one at a time or a small group at a time (**mini-batches**).  The speed at which they adapt to new data is determined by the **learning rate**.
    -   **Batch (Offline)**: The system learns from data all at once.  Note that some datasets are simply too large to even fit on a computer's memory.  If this is the case, the algorithm must support **out-of-core learning**.
-   **Prediction**: The act of a program using some algorithm to predict new values based on old data.
    -   **Instance-based**: Predictions are made based on how similar to already seen data they are.
    -   **Model-based**: Predictions are made based on a mathematical model.
-   **Evaluation**: An assessment on how well your ML model actually performs.
    -   **Utility/Fitness Function**: How _good_ is your model.
    -   **Cost Function**: How _bad_ is your model.

### Pipeline:

For creating a good model, a pipeline is usually followed:
1\. **Data Ingestion**: Have the ability to utilize data for an ML application.
2\. **Data Exploration**: Explore the data for relevant features and possible issues (along with the source).
3\. **Feature Engineering**: Coming up with a good set of features to train on.
4\. **Data Preparation**: Prepare the data to actually be utilized by your ML application.
5\. **Data Segregation**: Split the data into a training, validation, and testing set to properly create a model and gain a performance metric on it.  It is split to gain insight into how well the model performs on new data.  This error is called the **generalization error (or out-of-sample error)**.  The validation set is used for tuning **hyperparameters**, parameters specific to machine learning algorithms.  Note that the test set must remain unused throughout the entire project pipeline.  If the ML application can eventually see the entire test set after multiple runs, that invalidates the performance metric.
6\. **Model Training**: Train the ML application on data.
7\. **Model Evaluation**: Evaluate how well the model did.  If the model does not predict the training data well, it is called **underfitting**.  If it predicts the training data too well and does not predict the testing set as well, it is called **overfitting**.
8\. **Model Deployment**: Once the model is chosen, it is deployed into a decision-making framework (hopefully).
9\. **Deployed Evaluation**: The model is assessed on completely new data to gain insight on how well your model truly performs compared to other models.
10\. **Model Monitoring**: The model is applied to new datasets such that it can possibly gain more insight into the problem or be calibrated while be utilized by the user.

Note that this pipeline is nowhere near concrete.  **MLOps** (DevOps but for ML) introduces new ideas to better allow for changing models.  For simplicity we stop at step 7.  This is usually done if we want to give an ML model and be done with it.

## 2 Regression

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

## 3 Classification

**Classification** is the other side of regression in that instead of predicting values, we are predicting classes.  There are, however, multiple types of classification.

For this model, we can utilize gradient descent to find appropriate weights.  **gradient descent** is an optimization algorithm to minimize a cost function in order to find a good function that solves our problem.  The idea is find the gradient of our error function and move in the direction to minimize it.  We start with random weights (**random initialization**).  The size of step we take towards a minimum is called the **learning rate**.  Note that we are not guaranteed to find the global minimum.  There are three types of gradient descent:

-   **Batch Gradient Descent (BGD)**: Find partial derivative of all points and step in that direction.  This is computationally more intensive, but accounts for more data.
-   **Stochastic Gradient Descent (SGD)**: Find the partial derivative on one point on the training set making the descent a lot more random.
-   **Mini-batch Gradient Descent (MGD)**: Find partial derivative of some points in the training data and move in that direction. This is a midpoint between batch and stochastic gradient descent.

**Binary classification** is the classification between two classes (i.e. is or is not a 7).  These are the easiest of classification problems and a lot of other classification problems converge into this.  For our example of MNIST handwritten number classification, the numbers are converted into the number being the number 5.

When choosing a performance measure, accuracy using cross validation can be scary (95% accuracy but 95% are not 5s so works as well as a baseline).  The better measure is through a **confusion matrix**, the number of true-positives, false-positives, true-negatives, and false-negatives per class is given.  We can get metrics by calculating:

-   <img src="https://bit.ly/3dLSofg" align="center" border="0" alt="Precision = \dfrac{TP}{TP+FP}" width="171" height="43" />
-   <img src="https://bit.ly/2YduMtw" align="center" border="0" alt="Recall = \dfrac{TP}{TP+FN}" width="146" height="43" />
-   <img src="https://bit.ly/2A6c7b6" align="center" border="0" alt="F1 = \dfrac{precision * recall}{precision + recall} = \dfrac{TP}{TP + \dfrac{FN+FP}{2}}" width="331" height="68" />

The hyperparameters are chosen based on the **precision-recall trade-off curve**.  The **reciever operating characteristic (ROC) curve** is also used by plotting the true positive rate versus the false positive rate.  Usually the higher the recall, the more false positives.

For **multiclass (multinomial) classification**, the application distinguishes between more than one class.  For some algorithms, this is supported (such as Random Forest Classifiers), but for others, it is not.  To get around this, there are 2 techniques:

-   **One-versus-all (OvA)**: Creating binary classifiers for all other classes and decide based on the highest probability.  The is primarily used.
-   **One-versus-one (OvO)**: Create binary classifiers comparing each class to the other class.  This creates lots of models depending on the class so it is not preferred.

For **multilabel classification**, the application outputs more than one prediction (such as probabilities for each class).  Some algorithms support such, but one of the easiest is **K-Nearest Neighbors (KNN)**.  KNNs are instance based and simply finds the K nearest points to the value and outputs the associated labels.

**Multioutput classification** is the combination of multiclass and multilabel classification.  Therefore, it predicts multiple labels for multiple classes.

## 4 SVM

**Support Vector Machines (SVM)** are ML models with the ability to perform linear or nonlinear classification and regression.  They are exceptional for classification of small-medium sized datasets.

**Linear SVM** is the basics to all SVMs and can be defined as finding the largest possible margin, called the **large margin classification** to distinguish classes.  Finding the largest margin causes the model to overfit the data, called **hard margin classification**.  Therefore, **soft margin classification** is utilized instead: create a large enough margin that is not sensitive to outliers (limit **margin violations**).

**Nonlinear SVMs** are basically identical to linear SVMs except an additional step is added to the process.  Before any of the margins are created, another feature is added to the dataset that is equal to a polynomial of the original value thus increasing the number of dimensions but making the problem linear.  This is called the **polynomial kernel**.  The degree of polynomial to add is another hyperparameter, but for SVMs, the **kernel trick** is utilized to achieve the affect of having to many polynomials.  Another technique is the **Gaussian RBF Kernel** that moves the data into a normalized set.

For regression, we simply use the line for the margin as the regression function.  Note that these problems can be computationally expensive to solve so lots of **quadratic programming** (field of math) is used to correctly minimize the loss.  To find a better lower bound, for example, the **dual problem** is created from the **primal** and is solved since it is easier.

## Decision Trees

**Decision Trees** are a versatile ML algorithm that can perform both regression and classification.  In it's most simple term, a decision tree splits data it sees into sections until it reaches some depth.  These decision can usually be represented as a DAG and are very useful in that way.  For example, to make predictions, it starts at the root nodes and continually asks questions until it reaches a leaf, or the actual prediction.

For each node, a lot is given:
- **Decision Boundary**: Boundary or condition used to make a decision.
- **Gini**: The nodes **impurity**, a measure of how variable the classification is.  For instance if the gini is 0, then all samples are of one class.
    - <img src="https://bit.ly/2YnbC4x" align="center" border="0" alt="G_i=1-\sum_{k=1}^np_{i,k}^2" width="129" height="50" />
- **Samples**: The number of training instances given to make that decision.
- **Value**: The number of trainin instance per class.
- **Class**: The class to predict if it were to stop at that node.

These decisions are powerful since they are not as **black-box** (unable to truly understand what is going on) as many other ML algorithms are.

To train these models, the **Classification And Regression Tree (CART) or Growing Trees algorithm** is used.  The idea is to simply split the training set and choose features that result in the purest values.  The algorithm is as follows:

<img src="https://bit.ly/3fbYupF" align="center" border="0" alt="J(k, t_k)=\frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}" width="279" height="40" />.

The computational complexity for predictions is **O(log(m))**.  This can be hastened by presorting the data (for smaller training sets).

An alternative to the gini impurity metric is **entropy** which measures the average information content of each message (an entropy of 0 is when all predictions are of one class).  Usually they are really similar, but gini tends to isolate the most frequent class into its own tree, but is computationally faster.  Entropy is defined as follows:

<img src="https://bit.ly/2YkVgcA" align="center" border="0" alt="H_i=-\sum_{k=1}^np_{i,k}log(p_{i,k})" width="182" height="50" />.

When regularizing hyperparameters, the **max depth** represents the maximum tree depth that the graph can go to.  Since decision trees tend to overfit, this is usually necessary to **prune the tree**.  A model that specifies this is called a **parametric model** and one that does not is a **non-parametric model**.
