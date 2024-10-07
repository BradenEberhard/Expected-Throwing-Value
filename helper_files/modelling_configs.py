from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

thrower_context = ['thrower_x', 'thrower_y']
receiver_context = ['receiver_x', 'receiver_y']
throw_context = ['throw_distance', 'distance_short', 'distance_medium', 'distance_long', 
                 'direction_forward', 'direction_sideways', 'direction_backward']
game_context = ['possession_num', 'possession_throw', 'game_quarter', 'quarter_point', 'score_diff', 'times']
player_context = ['thrower_completion_percentage', 'throw_ec_per_possession', 'distance_tfidf', 'direction_tfidf', 
                  'distance_direction_tfidf', 'backward_percentage', 'sideways_percentage', 'forward_percentage']

model_config = {
    'n_trials': 10, 
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
        #         'max_iter': 50,
        #         'random_state': 0
        #     }
        # },
        'xgb': {
            'model_name': 'xgboost',
            'model_class': XGBClassifier,
            'params': {
                'n_estimators': lambda trial: trial.suggest_int("n_estimators", 10, 200),
                'learning_rate': lambda trial: trial.suggest_float("learning_rate", 1e-4, 1, log=True),
                'max_depth': lambda trial: trial.suggest_int("max_depth", 1, 10),
                'random_state': 0
            }
        }
    }
}