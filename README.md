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

## Logistic Regression

The logistic regression algorithm is a classification algorithm: it can predict the probability of an event to be true or false. In our case, it is a multi-classification, e.g. will the student be in Gryffindor, Slytherin, Ravenclaw or Hufflepuff.
We use the One vs All method to train the model: For each house, we divide the samples in 2 groups: `1` if the student is in the house, `0` otherwise.

The logistic regression model is based on the sigmoid function, which is a function that maps any real value into a value between 0 and 1.

## Ressources

https://datascientest.com/regression-logistique-quest-ce-que-cest

https://www.ibm.com/fr-fr/topics/logistic-regression