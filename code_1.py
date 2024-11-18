import pandas as pd
import numpy as np
import json

# Load parameters from JSON file
with open('algoparams_from_ui.json', 'r') as f:
    params = json.load(f)

# Extracting target and feature handling details
target = params['design_state_data']['target']
feature_handling = params['design_state_data']['feature_handling']
dataset_name = params['design_state_data']['session_info']['dataset']

# Load dataset
df = pd.read_csv(dataset_name)

# Data Preprocessing
for col, feature in feature_handling.items():
    if not feature['is_selected']:
        df.drop(col, axis=1, inplace=True)

    if feature["feature_variable_type"] == "numerical":
        if feature['feature_details']["missing_values"] == "Impute":
            if feature['feature_details']['impute_with'] == "Average of values":
                df[col].fillna(df[col].mean(), inplace=True)
            elif feature['feature_details']['impute_with'] == "custom":
                df[col].fillna(feature['feature_details']['impute_value'], inplace=True)
            else:
                raise AssertionError(f"Unknown imputation method: {feature['feature_details']['impute_with']}")
    elif feature["feature_variable_type"] == "text":
        labels = {key: num for num, key in enumerate(df[col].unique())}
        df[col] = df[col].apply(lambda x: labels[x])
    else:
        raise AssertionError(f"Unknown feature type: {feature['feature_variable_type']}")

# Feature Reduction
config = params['design_state_data']['feature_reduction']
target_col = target['target']
X = df.drop(target_col, axis=1).values
y = df[target_col].values

if config['feature_reduction_method'] == "Tree-based":
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel

    if target['type'] == "regression":
        model = RandomForestRegressor(n_estimators=int(config['num_of_trees']), max_depth=int(config['depth_of_trees']))
    elif target['type'] == "classification":
        model = RandomForestClassifier(n_estimators=int(config['num_of_trees']), max_depth=int(config['depth_of_trees']))
    else:
        raise AssertionError(f"Unknown target type: {target['type']}")

    sel = SelectFromModel(model)
    sel.fit(X, y)
    feature_importance = sel.estimator_.feature_importances_
    sorted_indices = np.argsort(feature_importance)[::-1]
    keep_columns = df.columns[np.concatenate((sorted_indices[:int(config['num_of_features_to_keep'])], [list(df.columns).index(target_col)]))]
    df = df[keep_columns]

elif config['feature_reduction_method'] == "No Reduction":
    pass

elif config['feature_reduction_method'] == "Correlation with target":
    corr = df.corr()[target_col].drop(target_col)
    sorted_cor = sorted(dict(abs(corr).items()).items(), key=lambda x: x[1], reverse=True)[:int(config['num_of_features_to_keep'])]
    keep_columns = np.array([key for key, value in sorted_cor] + [target_col])
    df = df[keep_columns]

elif config['feature_reduction_method'] == "Principal Component Analysis":
    from sklearn.decomposition import PCA
    pca = PCA(n_components=int(config['num_of_features_to_keep']))
    pca.fit(X)
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    X = pca.transform(X)

else:
    raise AssertionError(f"Unknown feature reduction method: {config['feature_reduction_method']}")

# Model selection and training
algorithms = params['design_state_data']['algorithms']
for algo_name, hyperparameters in algorithms.items():
    is_selected = hyperparameters.pop('is_selected')
    name = hyperparameters['model_name']

    if not is_selected:
        continue

    if name == "Random Forest Classifier":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X, y)
    elif name == "Random Forest Regressor":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        parameters = {
            'n_estimators': [hyperparameters["min_trees"], hyperparameters["max_trees"]],
            'max_depth': [hyperparameters["min_depth"], hyperparameters["max_depth"]],
            'min_samples_leaf': [hyperparameters["min_samples_per_leaf_min_value"], hyperparameters["min_samples_per_leaf_max_value"]]
        }
        grid = GridSearchCV(RandomForestRegressor(), parameters, cv=5, n_jobs=-1)
        grid.fit(X, y)
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best score: {grid.best_score_}")
        model = grid.best_estimator_

    print(f"Model trained: {name}")
    print(f"Hyperparameters: {hyperparameters}")
    break

print("Finished")
