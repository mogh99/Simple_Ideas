## JuhaAlgorithm
An idea is to develop a clustering algorithm for the clean cluster dataset. So, we take for example 3 clusters as 3 normal labels. For the fourth label, we generate random data around the 3 clusters and give them a fourth label as an anomaly. After, that use any supervised learning algorithm to train the model.

The idea is to force the model to generalize even if the dataset is small. So if we have two clusters or two classes we can generalize the model to consider any possible observation as another class. However, we make the model stronger to predict the two classes.

### Assumptions: 
- The data is cleanly separable clusters known beforehand via the business team.
- We don't want to use any clustering algorithm