# THE BEGINING

## FRAMING ML

Supervised ML systems learn how to combine input to produce useful predictions on never before seen data.

**Labels**: is the thing we're predicting. It's the `y variable` in linear regression

<img src="https://latex.codecogs.com/gif.latex? y"/>

**Features**: is an input variable. It's the `x variable` in linear regression.

<img src="https://latex.codecogs.com/gif.latex? x_1, x_2,...,x_N"/>

**Models**: defines the relationship between features and label
- **Training**: creating or learning the model. You show the model labeled examples and enable the model to gradually learn the relationships between features and label

- **Inference**: means applying the trained model to unlabeled examples

**Linear Regression**: a `Model` where linear combinations of `Features` are used to precited the `Label`.

<img src="https://latex.codecogs.com/gif.latex? y`=b+w_1x_1"/>

Where

- <img src="https://latex.codecogs.com/gif.latex? y`"/>: the value we're trying to predict
- <img src="https://latex.codecogs.com/gif.latex? b"/>: the bias or y-intercept. the value for <img src="https://latex.codecogs.com/gif.latex? y`"/> when <img src="https://latex.codecogs.com/gif.latex? x_1 = 0"/>
- <img src="https://latex.codecogs.com/gif.latex? w_1"/>: weigth applied to feature <img src="https://latex.codecogs.com/gif.latex? x_1"/>, also known as the slope. for every 1 increase in <img src="https://latex.codecogs.com/gif.latex? x_1"/>, also known as the slope. for every 1 increase in <img src="https://latex.codecogs.com/gif.latex? x_1"/>, <img src="https://latex.codecogs.com/gif.latex? y`"/> increase by that amount
- <img src="https://latex.codecogs.com/gif.latex? x_1"/>: the feature

Example: using the Feature "Chirps per Minute" to precited the Label "Temperature in Celsius". The `Model` is the line.<img src="https://latex.codecogs.com/gif.latex? y`=2 + 0.25 * x_1"/>

<img src="img/img01.png"/>

Linear regression can have multiple features all woth their own weights:

<img src="https://latex.codecogs.com/gif.latex? y`=b+w_1x_1+w_2x_2+w_3x_3+...+w_Nx_N"/>
