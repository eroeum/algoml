# Classification

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
