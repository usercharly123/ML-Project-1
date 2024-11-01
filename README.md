# ML-Project-1

## Group

Charles-Edouard Rouault

Tistou Luisière

Noé Prat

## Code organisation

The dataset could not be pushed in the git, one may need to add it after cloning the repository, in the same directory as `run.py`

`implementations.py` contains the basic functions: 

- `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)`

- `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)`

- `least_squares(y, tx)`

- `ridge_regression(y, tx, lambda_)`

- `logistic_regression(y, tx, initial_w, max_iters, gamma)`

- `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)`

To reproduce our best submission, run `run.py`.

## Data preparation

- **Outlier removal and pruning**

To tackle the high redundancy, we pruned the features, by removing every variable highly correlated (> 0.8 in modulus) to another variable of the dataset. Depending on the normalization preprocessing step that we apply, we remove 94 or 115 columns with this process.

- **Normalization**

We used standard normalization. We only transformed the "don't know" answers in NaNs (and not the "prefer not to say") in our model.


- **Sparsity of the data: NaN handling**

We wanted that the NaN values were considered as "neutral" information in our model.
Thus, for both categorical and continuous features, we chose to assess whether transforming them in the median value for the features, or in zeros, would be the best option for the neutrality.

We performed this step after the normalization, so that this artificial transformation does not introduce bias in the normalization.

