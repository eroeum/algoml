# Basic Examples for Machine Learning Algorithms

The following modules contain code examples of multiple common machine learning algorithms.  This is intended for getting a basic understanding of machine learning concepts and ideas and how they can actually be implemented using higher level libraries, specifically scikit-learn and Tensorflow in Python.

Note that these examples were highly based off of Aurelien Geron's Hands-On Machine Learning with Scikit-Learn & Tensorflow book with the change of a lot of programming to follow more pythonic styling and introducing a more formal code base.

In this README, I will give notes on important topics related to each alogrithm.

## 1 Primary Terms:

The following are the basic terms that are always taught:
- **Machine Learning (ML)**: Programming computers to *learn* from data. Applying ML techniques to find patterns in big data, it is called **data mining**. ML is best utilized when:
    - An existing solution requires constant changes.
    - The problem is to complex to even find a solution.
- **Human Supervision**: The involvement of human teaching to an ML application.
    - **Supervised**: Data fed into the program has *all* solutions already given.  These solutions are called **labels**.
    - **Unsupervised**: Data fed into the program has *no* solutions already given.
    - **Semi-supervised**: Data fed into the program has *some* solutions already given.
    - **Reinforcement**: An agent is rewarded by acting in some simulated environment.
- **Learning**: The act of a program utilizing data to solve a problem.
    - **Online**: The system learns from data incrementally.  This can happen one at a time or a small group at a time (**mini-batches**).  The speed at which they adapt to new data is determined by the **learning rate**.
    - **Batch (Offline)**: The system learns from data all at once.  Note that some datasets are simply too large to even fit on a computer's memory.  If this is the case, the algorithm must support **out-of-core learning**.
- **Prediction**: The act of a program using some algorithm to predict new values based on old data.
    - **Instance-based**: Predictions are made based on how similar to already seen data they are.
    - **Model-based**: Predictions are made based on a mathematical model.
- **Evaluation**: An assessment on how well your ML model actually performs.
    - **Utility/Fitness Function**: How *good* is your model.
    - **Cost Function**: How *bad* is your model.

For creating a good model, a pipeline is usually followed:
1. **Data Ingestion**: Have the ability to utilize data for an ML application.
2. **Data Exploration**: Explore the data for relevant features and possible issues (along with the source).
3. **Feature Engineering**: Coming up with a good set of features to train on.
4. **Data Preparation**: Prepare the data to actually be utilized by your ML application.
5. **Data Segregation**: Split the data into a training, validation, and testing set to properly create a model and gain a performance metric on it.  It is split to gain insight into how well the model performs on new data.  This error is called the **generalization error (or out-of-sample error)**.  The validation set is used for tuning **hyperparameters**, parameters specific to machine learning algorithms.
6. **Model Training**: Train the ML application on data.
7. **Model Evaluation**: Evaluate how well the model did.  If the model does not predict the training data well, it is called **underfitting**.  If it predicts the training data too well and does not predict the testing set as well, it is called **overfitting**.
8. **Model Deployment**: Once the model is chosen, it is deployed into a decision-making framework (hopefully).
9. **Deployed Evaluation**: The model is assessed on completely new data to gain insight on how well your model truly performs compared to other models.
10. **Model Monitoring**: The model is applied to new datasets such that it can possibly gain more insight into the problem or be calibrated while be utilized by the user.

Note that this pipeline is nowhere near concrete.  **MLOps** (DevOps but for ML) introduces new ideas to better allow for changing models.  For simplicity we stop at step 7.  This is usually done if we want to give an ML model and be done with it.
