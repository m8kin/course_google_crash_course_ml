# THE BEGINING

## FRAMING ML with LINEAR REGRESSION

Supervised ML systems learn how to combine input to produce useful predictions on never before seen data.

**Labels**: is the thing we're predicting. It's the `y variable` in linear regression

<img src="https://latex.codecogs.com/gif.latex?y"/>
</br>

**Features**: is an input variable. It's the `x variable` in linear regression.

<img src="https://latex.codecogs.com/gif.latex?x_1,x_2,...,x_N"/>
</br>

**Models**: defines the relationship between features and label
- **Training**: creating or learning the model. You show the model labeled examples and enable the model to gradually learn the relationships between features and label

- **Inference**: means applying the trained model to unlabeled examples
</br>

**Linear Regression**: a `Model` where linear combinations of `Features` are used to precited the `Label`.

<img src="https://latex.codecogs.com/gif.latex?y'=b+w_1x_1"/>

where

- <img src="https://latex.codecogs.com/gif.latex?y'"/>: the value we're trying to predict
- <img src="https://latex.codecogs.com/gif.latex?b"/>: the bias or y-intercept. the value for <img src="https://latex.codecogs.com/gif.latex?y'"/> when <img src="https://latex.codecogs.com/gif.latex?x_1=0"/>
- <img src="https://latex.codecogs.com/gif.latex?w_1"/>: weigth applied to feature <img src="https://latex.codecogs.com/gif.latex?x_1"/>, also known as the slope. for every 1 increase in <img src="https://latex.codecogs.com/gif.latex?x_1"/>, also known as the slope. for every 1 increase in <img src="https://latex.codecogs.com/gif.latex?x_1"/>, <img src="https://latex.codecogs.com/gif.latex?y'"/> increase by that amount
- <img src="https://latex.codecogs.com/gif.latex?x_1"/>: the feature

example:

Using the Feature "Chirps per Minute" to precited the Label "Temperature in Celsius". The `Model` is the line.

<img src="https://latex.codecogs.com/gif.latex?y'=2+0.25*x_1"/>

</br>

<img src="img/img01.png"/>

Linear regression can have multiple features all with their own weights:

<img src="https://latex.codecogs.com/gif.latex?y'=b+w_1x_1+w_2x_2+w_3x_3+...+w_Nx_N"/>

</br></br>

## A LOSS FUNCTION

In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

**Loss**: is the penalty for a bad prediction

**Mean Squared Error (L2 loss)**: a popular linear regression loss function for continuous labels

<img src="https://latex.codecogs.com/gif.latex?MSE=\frac1N\sum_{(x,y)\in D}(y-prediction(x))^2"/>

where:
- <img src="https://latex.codecogs.com/gif.latex?(x,y)"/> is an example in which
    - <img src="https://latex.codecogs.com/gif.latex?x"/> is the set of features
    - <img src="https://latex.codecogs.com/gif.latex?y"/> is the example's label
- <img src="https://latex.codecogs.com/gif.latex?prediction(x)"/> is the output of the function of the weights and bias in combination with the set of features
- <img src="https://latex.codecogs.com/gif.latex?D"/> is a data set containing many labeled examples
- <img src="https://latex.codecogs.com/gif.latex?N"/> is the number of examples in <img src="https://latex.codecogs.com/gif.latex?D"/>


High loss in the left model; low loss in the right model.

<img src="img/img02.png"/>


## REDUCING LOSS: AN ITERATIVE APPROACH

<img src="img/img03.svg"/>
