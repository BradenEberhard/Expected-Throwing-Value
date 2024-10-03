from sklearn.preprocessing import StandardScaler
from helper_files.modelling_configs import *
from helper_files.modelling_functions import process_data
import pandas as pd

class ETVModel:
    def __init__(self, cp_model, fv_model, thrower_context=thrower_context, receiver_context=receiver_context, throw_context=throw_context, game_context=game_context):
        self.cp_model = cp_model
        self.fv_model = fv_model
        self.thrower_context = thrower_context
        self.receiver_context = receiver_context
        self.throw_context = throw_context
        self.game_context = game_context

        # Define feature sets for each model
        self.cv_features = thrower_context + receiver_context + throw_context + game_context
        self.fv_features = receiver_context + game_context

    def process_data(self, df, features):
        """Process the data and return scaled features."""
        X_train = df[features]
        X_train = X_train.fillna(X_train.median())
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        return X_train_scaled

    def predict(self, df, etv_only=True):
        """Make predictions using both models and compute the ETV."""

        def get_opponent_df():
            opponent_df = df[self.fv_features].copy()
            # on a turnover, the field flips mirrors in the x dimension
            opponent_df.loc[:, 'receiver_x'] = -opponent_df.loc[:, 'receiver_x']
            # discs turned over in either endzone are brought to the front line
            opponent_df.loc[:, 'receiver_y'] = (120 - opponent_df.loc[:, 'receiver_y']).clip(lower=20, upper=100)
            # the possession number increments, throw counter resets and score differential inverts
            if 'possession_num' in opponent_df.columns:
                opponent_df.loc[:, 'possession_num'] += 1

            if 'possession_throw' in opponent_df.columns:
                opponent_df.loc[:, 'possession_throw'] = 1

            if 'score_diff' in opponent_df.columns:
                opponent_df.loc[:, 'score_diff'] = -opponent_df.loc[:, 'score_diff']
                return opponent_df

        # 1. Get the completion probability predictions
        X_cp = self.process_data(df, self.cv_features)
        cp_preds = self.cp_model.predict_proba(X_cp)[:, 1]

        # 2. Get the field value (spatial) predictions
        X_fv = self.process_data(df, self.fv_features)
        fv_preds = self.fv_model.predict_proba(X_fv)[:, 1]

        # 3. Generate opponent features for spatial model
        opponent_df = get_opponent_df()
        X_fv_opponent = self.process_data(opponent_df, self.fv_features)
        fv_preds_opponent = self.fv_model.predict_proba(X_fv_opponent)[:, 1]

        # 4. Compute ETV as (CP * FVo) - ((1 - CP) * FVd)
        etv_preds = (cp_preds * fv_preds) - ((1 - cp_preds) * fv_preds_opponent)
        if etv_only:
            return etv_preds
        return cp_preds, fv_preds, etv_preds
