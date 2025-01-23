from sklearn.preprocessing import StandardScaler
from helper_files.modelling_configs import *
from helper_files.modelling_functions import process_data
import pandas as pd

class ETVModel:
    def __init__(self, cp_model, fv_model, thrower_context=thrower_context, receiver_context=receiver_context, throw_context=throw_context, game_context=game_context):
        self.cp_model = cp_model['model']
        self.fv_model = fv_model['model']
        self.cp_scaler = cp_model['scaler']
        self.fv_scaler = fv_model['scaler']
        self.thrower_context = thrower_context
        self.receiver_context = receiver_context
        self.throw_context = throw_context
        self.game_context = game_context

        # Define feature sets for each model
        self.cp_features = cp_model['features']
        self.fv_features = fv_model['features']
    
    def predict_new(self, df):
        return self.predict(df, etv_only=False, new_etv_only=True)

    def predict(self, df, etv_only=True, new_etv_only=False):
        """Make predictions using both models and compute the ETV."""

        def get_opponent_df(df):
            opponent_df = df[self.fv_features].copy()
            # on a turnover, the field flips mirrors in the x dimension
            opponent_df.loc[:, 'thrower_x'] = -opponent_df.loc[:, 'thrower_x']
            # discs turned over in either endzone are brought to the front line
            opponent_df.loc[:, 'thrower_y'] = (120 - opponent_df.loc[:, 'thrower_y']).clip(lower=20, upper=100)
            # the possession number increments, throw counter resets and score differential inverts
            if 'possession_num' in opponent_df.columns:
                opponent_df.loc[:, 'possession_num'] += 1

            if 'possession_throw' in opponent_df.columns:
                opponent_df.loc[:, 'possession_throw'] = 1

            if 'score_diff' in opponent_df.columns:
                opponent_df.loc[:, 'score_diff'] = -opponent_df.loc[:, 'score_diff']

            return opponent_df

        # 1. Get the completion probability predictions
        X_cp = self.cp_scaler.transform(df[self.cp_features])
        cp_preds = self.cp_model.predict_proba(X_cp)[:, 1]

        # 2. Get the field value (spatial) predictions
        fv_df = df[self.fv_features]
        X_fv_start = self.fv_scaler.transform(fv_df)
        receiver_features = [x.replace('thrower', 'receiver') for x in self.fv_features]
        fv_df = df[receiver_features]
        fv_df = fv_df.rename(columns={'receiver_x': 'thrower_x', 'receiver_y': 'thrower_y'})
        X_fv_end = self.fv_scaler.transform(fv_df)
        fv_preds_start = self.fv_model.predict_proba(X_fv_start)[:, 1]
        fv_preds_end = self.fv_model.predict_proba(X_fv_end)[:, 1]
        fv_preds_end[df['receiver_y'] > 100] = 1

        # 3. Generate opponent features for spatial model
        opponent_df = get_opponent_df(fv_df)
        X_fv_opponent = self.fv_scaler.transform(opponent_df[self.fv_features])
        fv_preds_opponent = self.fv_model.predict_proba(X_fv_opponent)[:, 1]

        # 4. Compute ETV as (CP * FVo) - ((1 - CP) * FVd)
        etv_preds = (cp_preds * (fv_preds_end)) - ((1 - cp_preds) * fv_preds_opponent)

        if etv_only:
            return etv_preds        
        return cp_preds, fv_preds_start, fv_preds_end, fv_preds_opponent, etv_preds
