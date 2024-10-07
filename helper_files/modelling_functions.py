import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import optuna
from imblearn.over_sampling import SMOTE


def get_data(data_filepath='./data/processed/data_splits_0926.jblb', ec_filepath='./data/processed/player_ec.jblb', include_ec=False):
    data = joblib.load(data_filepath)
    train_df = data['train_df'].drop(['total_points'], axis=1)
    test_df_random = data['test_df_random'].drop(['total_points'], axis=1)
    test_df_time = data['test_df_time'].drop(['total_points'], axis=1)
    test_df_thrower = data['test_df_thrower'].drop(['total_points'], axis=1)

    if include_ec:
        player_ec = joblib.load(ec_filepath).drop(['total_throws'], axis=1)
        train_df = pd.merge(train_df, player_ec, how='left', left_on='thrower', right_on='thrower')
        test_df_random = pd.merge(test_df_random, player_ec, how='left', left_on='thrower', right_on='thrower')
        test_df_time = pd.merge(test_df_time, player_ec, how='left', left_on='thrower', right_on='thrower')
        test_df_thrower = pd.merge(test_df_thrower, player_ec, how='left', left_on='thrower', right_on='thrower')
    train_df = train_df.replace(-np.inf, np.nan)
    test_df_time = test_df_time.replace(-np.inf, np.nan)
    test_df_random = test_df_random.replace(-np.inf, np.nan)
    test_df_thrower = test_df_thrower.replace(-np.inf, np.nan)
    return train_df, test_df_time, test_df_random, test_df_thrower

def process_data(train_df, test_dfs, features, target, mirror=False, noise_factor=0.0, smote=False):
    """
    Process the training and test DataFrames by scaling the features and applying optional data augmentation.

    Parameters:
        train_df (pd.DataFrame): The training DataFrame.
        test_dfs (list): List of test DataFrames.
        features (list): List of feature names to scale.
        target (str): Target column name.
        mirror (bool): Whether to mirror/augment the data (default False).
        noise_factor (float): Noise factor for jittering (default 0.0, no jitter).
        smote (bool): Whether to apply SMOTE for balancing the classes (default False).

    Returns:
        tuple: Scaled training data, list of scaled test data, target values for training and test sets, and fitted scaler.
    """
    
    if mirror:
        # Apply mirroring to the data (negate '_x' features)
        new_train_df = train_df.copy()
        for col in [x for x in features if '_x' in x]:
            new_train_df[col] = new_train_df[col] * -1  # Mirroring (negating values)
        # Combine original and augmented data
        train_df = pd.concat([train_df, new_train_df], ignore_index=True)

    

    # Identify features with more than 2 unique values
    features_with_jitter = [feature for feature in features if train_df[feature].nunique() > 2]
    feature_mins = train_df[features_with_jitter].min()
    feature_maxs = train_df[features_with_jitter].max()
    if noise_factor > 0 and features_with_jitter:  # Ensure there are features to apply jitter to
        # Add uniform jitter noise to the selected features
        jitter = np.random.uniform(-noise_factor, noise_factor, size=train_df[features_with_jitter].shape) * np.array(feature_maxs - feature_mins)
        noisy_train_df = train_df.copy()
        noisy_train_df[features_with_jitter] += jitter
        
        # Clip the noisy data to the original min/max values for selected features
        for feature in features_with_jitter:
            noisy_train_df[feature] = noisy_train_df[feature].clip(lower=feature_mins[feature], upper=feature_maxs[feature])
        
        # Combine original and jittered data
        train_df = pd.concat([train_df, noisy_train_df], ignore_index=True)

    # Apply SMOTE to balance classes if required
    if smote:
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(train_df[features], train_df[target])
        train_df_resampled = pd.DataFrame(X_resampled, columns=features)
        train_df_resampled[target] = y_resampled
        train_df = train_df_resampled  # Replace original training data with resampled data

    # Scale the training data
    X_train = train_df[features]
    X_train = X_train.fillna(X_train.median())
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = train_df[target].values
    
    # Scale the test data
    X_scaled_tests, y_tests = [], []
    for test_df in test_dfs:
        X_test = test_df[features]
        X_test = X_test.fillna(X_train.median())  # Use training median for imputation
        X_test_scaled = scaler.transform(X_test)
        X_scaled_tests.append(X_test_scaled)
        y_tests.append(test_df[target].values)

    return X_train_scaled, y_train, X_scaled_tests, y_tests, scaler


def attempt_delete(study_name, storage):
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass

def evaluate_models(best_models, data_config):
    metrics_list = []

    for model_name, model in best_models.items():
        model_metrics = {'Model': model_name}

        # Make predictions
        y_pred = model.predict(data_config['train_data_final'][0])
        y_pred_proba = model.predict_proba(data_config['train_data_final'][0])[:, 1]  # Get probabilities for AUC
        y_test = (data_config['train_data_final'][1])
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate NPV (Negative Predictive Value)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        test_name = 'train'
        # Store metrics for the current test set
        model_metrics[f'{test_name}_AUC'] = auc
        model_metrics[f'{test_name}_Accuracy'] = accuracy
        model_metrics[f'{test_name}_NPV'] = npv
        # model_metrics[f'{test_name}_F1'] = f1
        for test_name, (X_test, y_test) in data_config['test_datas_final'].items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for AUC

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            # Calculate NPV (Negative Predictive Value)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            # Store metrics for the current test set
            model_metrics[f'{test_name}_AUC'] = auc
            model_metrics[f'{test_name}_Accuracy'] = accuracy
            model_metrics[f'{test_name}_NPV'] = npv
            # model_metrics[f'{test_name}_F1'] = f1

        metrics_list.append(model_metrics)

    # Convert the metrics list into a DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    return metrics_df

def run_optuna_trials(data_config, model_config, suffix, delete_models=False, storage = 'sqlite:///optuna_dbs/spatial.db'):
    X, y, best_models, best_params = data_config['train_data_final'][0], data_config['train_data_final'][1], {}, {}

    # Loop over each model type in the configuration
    for model_key, model_info in model_config['models'].items():
        def objective(trial):
            # Create model instance
            model_class = model_info['model_class']
            model_params = {key: value(trial) if callable(value) else value for key, value in model_info['params'].items()}
            model = model_class(**model_params)

            auc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', error_score='raise')
            return auc_scores.mean()
        
        study_name = f"{model_key}_{suffix}"
        if delete_models:
            attempt_delete(study_name, storage)
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")
        n_trials = model_config['n_trials'] if model_key != 'logreg' else 1
        study.optimize(objective, n_trials=n_trials)

        best_model = model_info['model_class'](**study.best_params)
        best_model.fit(X, y)
        best_models[model_key] = best_model
        best_params[model_key] = study.best_params

    return best_models, best_params

def get_data_config(features, target, train_df, test_dfs, mirror=False, noise_factor=0.0, smote=False):
    new_features = [x for x in features if x in train_df]
    if len([x for x in features if x not in new_features]) > 0:
        print('features not used: ', [x for x in features if x not in new_features])
    X_train_scaled, y_train, X_scaled_tests,y_tests, scaler = process_data(train_df, test_dfs, new_features, target, mirror, noise_factor=noise_factor, smote=smote)
    data_config = {
        'train_df_raw': train_df,
        'test_dfs_raw': {
            'test_df_random': test_dfs[0],
            'test_df_time': test_dfs[1],
            'test_df_thrower': test_dfs[2]
        },
        'train_df_scaled': pd.DataFrame(X_train_scaled, columns=new_features),
        'test_datas_final': {
            'test_df_random': (pd.DataFrame(X_scaled_tests[0], columns=new_features), y_tests[0]),
            'test_df_time': (pd.DataFrame(X_scaled_tests[1], columns=new_features), y_tests[1]),
            'test_df_thrower': (pd.DataFrame(X_scaled_tests[2], columns=new_features), y_tests[2]),
        },
        'train_data_final': (pd.DataFrame(X_train_scaled, columns=new_features), y_train),
        'features':new_features,
        'target':target,
        'scaler':scaler
    }
    return data_config

def get_best_model(study_name, storage, train_df, test_dfs, model_config, features, target, mirror=False, noise_factor=0.0, smote=False):
    loaded_study = optuna.load_study(study_name=study_name, storage=storage)
    loaded_study.best_trial
    data_config = get_data_config(features, target, train_df, test_dfs, mirror, noise_factor, smote)
    model = model_config['models'][study_name.split('_')[0]]['model_class'](**loaded_study.best_params)
    model.fit(data_config['train_data_final'][0], data_config['train_data_final'][1])
    return {'model':model, 'features':features, 'scaler':data_config['scaler']}

