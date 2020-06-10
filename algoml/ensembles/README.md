# Ensembles

**Ensembles** are the act of taking multiple created models and unifying their predictions.  This is extremely useful, especially for fine-tuning a model since models tend to make better predictions with more inputs. This method is powerful enough to make a group of **weak learners**, models that predict a bit better than random guessing, into a single **strong learner**, a model that achieves high accuracy.  This is due to the **law of large numbers**.

One type of ensembles are **voting classifiers**.  These are methods of taking lots of models and listening to all of them.  There are 2 types of these classifiers:
- **Hard Voting**: Take the majority vote as the final prediction.
- **Soft Voting**: All models give a probability and take the highest probability.

**Bootstrap aggregating (Bagging)** is another ensemble that uses the same model but trains them on subsets of the training data with replacements.  **Pasting** is the same but without replacement.  The results are models with lower bias and variance and come at a better ability for training in parallel.  When bagging, some instances may never be sampled.  This is **out-of-bag instance**, instances that are never seen.  Another type of this sort is **random patches and subspaces**.  The idea to create models on a subset of features instead of instances.

A **Random Forest** is another ensemble method that is typically applying the bagging method to decision trees.  Random forests can take it one step forward and add more randomness by apply random thresholds for decision making at each node.  A very random forest is called an **extremly randomized trees (extre-trees)** and trades more bias for lower variance.  This can aid in data exploration by viewing how many trees use a feature for impurity detection thus showing feature importance.

**Hypothesis boosting (Boosting)** is another ensemble method that trains predictors sequentially trying to fix its predecessors.  When doing so, changing decreasing the learning rate requires more trees, causing **shrinkage**. For example,
- **AdaBoost**: Models try to boost the performance of instances that the predecessors seemed to underfit by tweaking the weights.  After creating the models, bagging is performed on all models except weighted based on accuracy.  The multiclass version is called **Stagewise Additive Modeling using a Multiclass Exponential Loss Function (SAMME)**.
- **Gradient Boosting (GBRT)**:  Models try to boost the performance of instances that the predecessors seemed to underfit by tweaking the model to the residual error.  After creating the models, bagging is performed on all models except weighted based on accuracy.
