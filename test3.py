import pandas as pd
import shap
import sklearn.ensemble
from sklearn.datasets import fetch_california_housing

# ----- Load dataset -----
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ----- Train a model -----
model = sklearn.ensemble.RandomForestRegressor()
model.fit(X, y)

# ----- Compute SHAP values -----
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Pick two features for a dependence plot
feat_main = "MedInc"  # Median income
feat_interact = "AveRooms"  # Number of average rooms

# ----- SHAP dependence-style plot -----
shap.plots.scatter(
    shap_values[:, feat_main],  # SHAP for main feature
    color=shap_values[:, feat_interact],  # SHAP for interacting feature
)
