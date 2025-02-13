# Data Science x Logistic Regression

## Introduction
This project aims to continue on the basics of machine learning through the implementation of a recoded logistic regression model.

The subject is to recreate the role of the Sorting Hat in the Harry Potter franchise. It will determine on which house a new student is meant to go, based on previous students' datas.

## Objectives
- Analyzing data: understanding the structure of datasets, selecting and cleaning data.
- Visualizing data: using visual tools to understand the data (Heatmap, histogram, scatter plot, pair plot). This will help to select the most relevant features for the model training.
- Logistic regression: understanding the logistic regression model, implementing it (in python for us) and using it on the dataset to train it and make predictions. It has to have at least 98% accuracy.

## Dataset

We have a dataset of 1600 previous students, with 14 features to train our model:
- Their House (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)
- Arithmancy
- Astronomy
- Herbology
- Defense Against the Dark Arts
- Divination
- Muggle Studies
- Ancient Run
- History of Magic
- Transfiguration
- Potions
- Care of Magical Creatures
- Charms
- Flying

After analyzing the data, we can see that some features allow to classify the students 
in a house more easily than others. For example, the feature `History of Magic` 
has a high correlation with the house Gryffindor, or `Muggle Studies` with Ravenclaw.

This allows us to select the most relevant features (we kept 9) to train our model and increase its accuracy.

## Logistic Regression

The logistic regression algorithm is a classification algorithm: it can predict 
the probability of an event to be true or false. In our case, it is a multi-classification, 
e.g. will the student be in Gryffindor, Slytherin, Ravenclaw or Hufflepuff. 
We use the One vs All method to train the model: For each house, we divide the samples 
in 2 groups: `1` if the student is in the house, `0` otherwise. 

The logistic regression model is based on the sigmoid function, which is a function 
that maps any real value into a value between 0 and 1. 

As in the linear regression, we use the gradient descent algorithm to minimize the 
cost function and increase the accuracy of the model. 

## Gradient descent algorithms

There are different types of gradient descent algorithms that can be used:

### Batch Gradient Descent

This is the most common type of gradient descent. It calculates the gradient of the cost function 
for the entire dataset and updates the weights at each iteration. It is very slow and 
computationally expensive, but it is guaranteed to converge to the global minimum.
Since it applies on all the dataset, the learning rate and the number of iterations can be set 
respectively to 0.01-0.1 and 500-2000.

### Stochastic Gradient Descent

This algorithm is faster than the batch gradient descent, but it is less accurate.
It calculates the gradient of the cost function for each sample and updates the weights at each iteration.
It is more noisy and can converge to a local minimum, but it is more efficient for large datasets.
The learning rate and the number of iterations can be set respectively to 0.001-0.01 and 100-500.

### Mini-batch Gradient Descent

This algorithm is a compromise between the batch and stochastic gradient descent.
It calculates the gradient of the cost function for a subset of the dataset and updates the weights at each iteration.
It is faster than the batch gradient descent and more accurate than the stochastic gradient descent.
The learning rate and the number of iterations can be set respectively to 0.01-0.1 and 100-500.

## Usage

For these programs to work, you need to have the following libraries installed:
`pandas`, `numpy`, `mathplotlib`, `seaborn`, `tqdm`

- To run `describe.py`, `heatmap.py`, `pairplot.py` and `scatterplot.py` :

        python3 describe.py data/dataset_train.csv

- To run `logreg_train.py` :

        python3 logreg_train.py data/dataset_train.csv <algo>

## Ressources

https://datascientest.com/regression-logistique-quest-ce-que-cest

https://www.ibm.com/fr-fr/topics/logistic-regression

https://mrmint.fr/logistic-regression-machine-learning-introduction-simple

https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/

https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/

https://prasad07143.medium.com/variants-of-gradient-descent-and-their-implementation-in-python-from-scratch-2b3cceb7a1a0

https://www.geeksforgeeks.org/principal-component-analysis-pca/