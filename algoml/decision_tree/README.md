# Decision Trees

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

However, decision trees can be very **instable** in that they make decisions that usually are orthogonal to an axis.  Therefore, any rotation can cause massive changes.
