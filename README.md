# Communities & Crime: SVD, Regression, and MDP

Analysis of the [UCI Communities and Crime dataset](https://archive.ics.uci.edu/dataset/183/communities+and+crime) - 1994 US communities, 128 socioeconomic features, target: violent crime rate per population.

## What this project does

Four components, all implemented from scratch using NumPy:

**SVD/PCA** - Dimensionality reduction from 100 features to 35 orthogonal components retaining 95% of variance. Eliminates multicollinearity without discarding variance information.

**Exploratory Data Analysis** - Distribution analysis, normality testing (Shapiro-Wilk), hypothesis testing (Welch t-test), and component-target correlation analysis.

**Regression models** - Linear regression and single-hidden-layer neural network with gradient descent and backpropagation. Hyperparameters tuned via 5-fold cross-validation on the training set only.

**Markov Decision Process** - Value iteration over a 4-state crime-risk MDP modelling community intervention policy. Optimal policy recovered via greedy extraction from V*.

## Results

| Model             | Test MSE | Test R² |
|-------------------|----------|---------|
| Linear Regression | 0.02018  | 0.61970 |
| Neural Network    | 0.02019  | 0.61947 |

- 35 principal components retained 95% of total variance
- PC1 (economic affluence/deprivation axis) correlates most strongly with crime (r = −0.63)
- MDP optimal policy: `maintain` in low-risk, `invest_prevention` in moderate and high-risk states

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Download the dataset - see [`data/README.md`](data/README.md).

Then open and run `notebook.ipynb`.

## Report

Full methodology, analysis, and ethical considerations in [`report.pdf`](report.pdf).
