# Ensembles

**Ensembles** are the act of taking multiple created models and unifying their predictions.  This is extremely useful, especially for fine-tuning a model since models tend to make better predictions with more inputs. This method is powerful enough to make a group of **weak learners**, models that predict a bit better than random guessing, into a single **strong learner**, a model that achieves high accuracy.  This is due to the **law of large numbers**.

One type of ensembles are **voting classifiers**.  These are methods of taking lots of models and listening to all of them. The idea of **Hard Voting** is to take the majority vote as the final prediction.
