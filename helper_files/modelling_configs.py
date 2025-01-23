from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

thrower_context = ['thrower_x', 'thrower_y']
receiver_context = ['receiver_x', 'receiver_y']
throw_context = ['throw_distance', 'throw_angle', 'y_diff', 'x_diff']
game_context = ['times']
player_context = ['thrower_completion_percentage', 'throw_ec_per_possession', 'distance_tfidf', 'direction_tfidf', 
                  'distance_direction_tfidf', 'backward_percentage', 'sideways_percentage', 'forward_percentage']
fv_features = thrower_context + game_context + ['thrower_x_squared', 'thrower_y_squared', 'thrower_interaction_squared']

model_config = {
    'n_trials': 15, 
    'models': {
        'logreg': {
            'model_name': 'logreg',
            'model_class': LogisticRegression,
            'params': {
                'random_state': 0
            }
        },
        # 'knn': {
        #     'model_name': 'knn',
        #     'model_class': KNeighborsClassifier,
        #     'params': {
        #         'n_neighbors': lambda trial: trial.suggest_int("n_neighbors", 1, 50)
        #     }
        # },
        # 'rf': {
        #     'model_name': 'random_forest',
        #     'model_class': RandomForestClassifier,
        #     'params': {
        #         'n_estimators': lambda trial: trial.suggest_int("n_estimators", 10, 200),
        #         'max_depth': lambda trial: trial.suggest_int("max_depth", 1, 10),
        #         'random_state': 0
        #     }
        # },
        # 'mlp': {
        #     'model_name': 'mlp',
        #     'model_class': MLPClassifier,
        #     'params': {
        #         'hidden_layer_sizes': lambda trial: trial.suggest_int("hidden_layer_sizes", 50, 200),  # Number of neurons
        #         'activation': lambda trial: trial.suggest_categorical("activation", ['relu', 'tanh', 'logistic']),
        #         'solver': lambda trial: trial.suggest_categorical("solver", ['adam']),
        #         'learning_rate': lambda trial: trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
        #         'early_stopping': True,
        #         'max_iter': 25,
        #         'random_state': 0
        #     }
        # },
        'xgb': {
            'model_name': 'xgboost',
            'model_class': XGBClassifier,
            'params': {
                'n_estimators': lambda trial: trial.suggest_int("n_estimators", 10, 200),
                'learning_rate': lambda trial: trial.suggest_float("learning_rate", 1e-4, 1, log=True),
                'max_depth': lambda trial: trial.suggest_int("max_depth", 2, 8),
                'random_state': 0
            }
        }
    }
}