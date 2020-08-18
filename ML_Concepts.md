# THE BEGINING

[FRAMING](#FRAMING-ML-with-LINEAR-REGRESSION)

[LOSS FUNCTION](#A-LOSS-FUNCTION)

[REDUCING LOSS](#REDUCING-LOSS-AN-ITERATIVE-APPROACH)

[LEARNING RATE](#LEARNING-RATE)

[BATCH](#MINI-BATCH-AND-STOCHASTIC-GRADIENT-DESCENT)

[EPOCH](#EPOCH)

[SUMMARY](#SUMMARY-OF-HYPER-PARAMETER-TUNING)



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

<img src="img/img01.png" width="600"/>

Linear regression can have multiple features all with their own weights:

<img src="https://latex.codecogs.com/gif.latex?y'=b+w_1x_1+w_2x_2+w_3x_3+...+w_Nx_N"/>

</br></br>

## A LOSS FUNCTION

In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

**Loss**: is the penalty for a bad prediction

**Mean Squared Error (L2 loss)**: a popular linear regression loss function for continuous labels

<img src="https://latex.codecogs.com/gif.latex?MSE=\frac1N\sum_{(x,y)\in_D}(y-prediction(x))^2"/>

where:
- <img src="https://latex.codecogs.com/gif.latex?(x,y)"/> is an example in which
    - <img src="https://latex.codecogs.com/gif.latex?x"/> is the set of features
    - <img src="https://latex.codecogs.com/gif.latex?y"/> is the example's label
- <img src="https://latex.codecogs.com/gif.latex?prediction(x)"/> is the output of the function of the weights and bias in combination with the set of features
- <img src="https://latex.codecogs.com/gif.latex?D"/> is a data set containing many labeled examples
- <img src="https://latex.codecogs.com/gif.latex?N"/> is the number of examples in <img src="https://latex.codecogs.com/gif.latex?D"/>


High loss in the left model; low loss in the right model.

<img src="img/img02.png" width="600"/>

</br></br>

## REDUCING LOSS AN ITERATIVE APPROACH

Iterative strategies are common in machine learning because they scale well to large data sets

The simple STEPS:

1) initialise with random `intercept` and `weight` values
2) run model
3) compute the loss
4) update `intercept` and `weights` (parameter update)
5) run model
6) compute the loss aka measure the `evaluation metric`
7) continue steps 4 to 6 until convergence (where loss reduces no futher)

<img src="img/img03.svg" width="600"/>

So the big question is how does STEP 4 update the `parameters` so loss is reduced and not increased?

An answer is **Gradient Descent**

Regression problems have a lovely property and that is that their **loss function** is **convex** or bowl shaped in relation to their `parameters`. This means we can easily find the **Gradient** of the loss function's slope and adjust the `paramters` so we can iteratively move in a negative direction (down the slope) and therefore reduce the loss to a `global minima`.

**Gradients** have both a `direction` (- or +) and a `magnitude` steepness of slop

<img src="img/img04.svg" width="400"/>  <img src="img/img05.svg" width="400"/>

Unfortunately not all loss functions are convex. **Neural Networks** for example have both a `global` and many `local minima`. Thier loss function might look the the swell of on ocean. This means that they have very strong dependecies on their `initial weights` in STEP 1.

</br>

### LEARNING RATE

So during STEP 4 when we update the `paramaters`, just how far of a step should we take? This is known as the **Learning Rate** or **Step Size**.

**Learning Rates** are a value (i.e. 0.001, 0.01, 0.1, or 1) that is multipled by the gradent to calcuate how big a step to take. SO if the `gradient magnitude` is 2.5 and the `learning rate` is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point in the `negative direction`

- Too large a step and we overshoot the `minima`
- Too small a step and we wait years for `convergence`
- Goldilocks step
    - flatter gradients can have large learning rates
    - steeper gradients require smaller learning rates

<img src="img/img06.svg" width="250"/>  <img src="img/img07.svg" width="300"/>  <img src="img/img08.svg" width="250"/>

</br>

### MINI BATCH AND STOCHASTIC GRADIENT DESCENT

The reason **Gradient Descent** scales so well to large datasets is because it can be calculated on just a sample of data.

- **Gradient Descent**: uses all the data to calculate the gradient, slow

- **Stochastic Gradient Descent**: just one sample at a time to calculate the gradient, fast but noisy

- **Mini-Batch Gradient Descent**: batches of 10-1000 to calculate the gradient, good

Checkout http://playground.tensorflow.org/ from Google!

</br>

### EPOCH

A full training pass over the entire dataset such that each example has been seen once. Thus, an epoch represents N/batch size training iterations, where N is the total number of examples.

</br>

### SUMMARY OF HYPER PARAMETER TUNING

Most machine learning problems require a lot of hyperparameter tuning. Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly. You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:

- Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
- If the training loss does not converge, train for more epochs.
- If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
- If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
- Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
- Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
- For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.


</br>

## GENERALISTION

Refers to a model's ability to adapt to new, previously unseen data. It is assumed that the new data is drawn from the same **distribution** as the one used to create the model meaning the distribution is `stationary`

`Overfitting` is when the mode is too attuned to the training data and doesn't `generalise` well to new data.

</br>

### Train, Test and Validation Data Subsets

The idea is to `train` the `model` on a subset of data and then `test` it or evaluate it on an unseen set of data. We the choose a model which performs well across both the `train` and `test` sets. A final evaluation is performed on the `validation` set.

</br>

**Train Set**: a random subset of data is drawn from the complete data set and is used to build the `model`.

- needs to be large enough to represent the complete data distribution
- a rule of thumb is 70% to 80% of the data

**Test Set**: half of the remaining data not drawn from the **training set** is used to test the `model` and to tweak any `hyperparameters` or `features`

- needs to be large enough to yield meaningful results
- is representative of the data as a whole
- a `generalised model` will perfom equal on the traing and test set

**Validation Set**: the remaing data is used as a final evaluation of the model

</br>

### k-Fold Cross Validation
Takes training and testing to the next level. In this method NO `validation` set is used.

Steps:
1. divide up the data into `k` blocks (5 to 10 blocks is typical)
2. `train` the `model` all blocks bar one
3. `test` the `model`one block and retain the `evaluation metric` for both the `train` and `test` sets
4. repeate this process `k` times and retain each `evaluation metric`
5. check the distributions of the `evaluation metric` and 

If the `evaluation metric` is stable, meaning it is similar between `traing` and `test` sets and also doesn't vary greatly then the `model` should generalise well.

The super extreme to the is **N-Fold Cross Validation** where the data is broken into N blocks

</br>

### Gradient Descent and Cross Validation
TBC

</br>

## REPRESENTATION

