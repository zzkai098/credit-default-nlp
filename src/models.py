"""PD, LGD, EAD model training and Expected Loss computation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie


def train_pd_model(X, y, test_size=0.3, random_state=42):
    """
    Train PD model using Logistic Regression.

    Returns:
        (model, X_train, X_test, y_train, y_test, y_pred_proba)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return model, X_train, X_test, y_train, y_test, y_pred_proba


def train_lgd_model(X, y):
    """
    Train LGD model using Beta Regression (GLM with logit link).

    LGD values are bounded in (0, 1), so Beta Regression is appropriate.

    Returns:
        (model, summary)
    """
    # Clamp y to (0.001, 0.999) to avoid boundary issues
    y_clamped = np.clip(y, 0.001, 0.999)

    X_const = sm.add_constant(X)

    # Use Tweedie family with log link as approximation for Beta regression
    model = sm.GLM(y_clamped, X_const, family=sm.families.Tweedie(var_power=1.5))
    result = model.fit()

    return result, result.summary()


def train_ead_model(X, y):
    """
    Train EAD model using Linear Regression on Credit Conversion Factors.

    Returns:
        (model, predictions)
    """
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions


def compute_expected_loss(pd_scores, lgd_scores, ead_scores):
    """Compute Expected Loss = PD * LGD * EAD."""
    return pd_scores * lgd_scores * ead_scores


def get_model_coefficients(model, feature_names):
    """Extract feature coefficients from a fitted LogisticRegression model."""
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        "abs_coefficient": np.abs(model.coef_[0]),
    })
    return coef_df.sort_values("abs_coefficient", ascending=False)
