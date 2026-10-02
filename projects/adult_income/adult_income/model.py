from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import NUMERIC, CATEGORICAL, prepare_features

def build_pipeline(model="gradient_boosting", seed=42):
    estimators = {
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "logistic": LogisticRegression(max_iter=1000, random_state=seed),
    }
    if model not in estimators:
        raise ValueError("Unknown estimator: " + model)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocessor), ("classifier", estimators[model])])

def predict_records(pipeline, frame):
    frame = prepare_features(frame)
    labels = pipeline.predict(frame)
    probabilities = pipeline.predict_proba(frame)
    high_index = list(pipeline.classes_).index(1)
    return [{"prediction": ">50K" if int(label) else "<=50K", "probability_gt_50k": float(probability[high_index])} for label, probability in zip(labels, probabilities)]
