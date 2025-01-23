import numpy as np
import pandas as pd

def pivot_metric_columns(df, metrics):
    """Pivot metrics columns into a long format."""
    return df.melt(
        id_vars=['player', 'year'], 
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )

def calculate_discrimination(bv, sv):
    """Calculate discrimination metric from bootstrapped and seasonal variance."""
    discrimination = (
        bv
        .merge(sv, on=['year', 'metric'], suffixes=('_bv', '_sv'))  # Inner join on year and metric
        .assign(discrimination=lambda x: 1 - x['bv'] / x['sv'])  # Calculate discrimination
        .sort_values(by=['year', 'metric'])  # Sort by year and metric
    )
    
    return discrimination[['year', 'metric', 'discrimination']]

def calculate_within_player_variance(player_game_season_stats, metrics):
    """Calculate within-player variance based on the player game season stats dataframe."""
    # Pivoting metrics for calculation
    pivoted_stats = (
        player_game_season_stats
        .melt(id_vars=['player', 'year'], 
              value_vars=metrics,
              var_name='metric', value_name='value')
    )

    within_player_variance = (
        pivoted_stats
        .groupby(['player', 'metric'], as_index=False)  # Group by player and metric
        .agg(
            seasons_played=('value', 'count'),  # Count of seasons played
            wv=('value', lambda x: x.var(ddof=0))  # Calculate variance
        )
    )

    # Filter for players with a minimum of 3 seasons played and calculate mean wv
    wv = (
        within_player_variance[within_player_variance['seasons_played'] >= 3]
        .groupby('metric', as_index=False)
        .agg(wv=('wv', 'mean'))  # Mean within-player variance
        .sort_values('metric')
    )
    
    return wv

def calculate_total_variance(player_game_season_stats, metrics):
    """Calculate total variance for each metric."""
    # Pivoting metrics for calculation
    pivoted_stats = (
        player_game_season_stats
        .melt(id_vars=['player', 'year'], 
              value_vars=metrics,
              var_name='metric', value_name='value')
    )

    # Calculate total variance for each metric
    tv = (
        pivoted_stats
        .groupby('metric', as_index=False)
        .agg(tv=('value', lambda x: x.var(ddof=0)))  # Total variance
        .sort_values('metric')
    )

    return tv

def calculate_stability(bv, tv, wv):
    """Calculate stability based on average bootstrap variance, total variance, and within-player variance."""
    
    # Calculate average bootstrap variance
    average_bv = bv.groupby('metric', as_index=False).agg(bv=('bv', 'mean')).reset_index()
    
    # Merge dataframes for stability calculation
    stability = (
        average_bv
        .merge(tv, on='metric', how='inner')
        .merge(wv, on='metric', how='inner')
        .assign(stability=lambda x: 1 - (x['wv'] - x['bv']) / (x['tv'] - x['bv']))
        .sort_values('metric')
    )

    return stability

def get_opponent_data(df, features):
    """flip the team possession to get the opponents features if turnover"""
    opponent_data = df[features].copy()
    opponent_data['thrower_x'] = -opponent_data.receiver_x
    opponent_data.loc[:, 'thrower_y'] = (120 - opponent_data.loc[:, 'receiver_y']).clip(lower=20, upper=100)
    if 'possession_num' in opponent_data.columns:
        opponent_data.loc[:, 'possession_num'] += 1

    if 'possession_throw' in opponent_data.columns:
        opponent_data.loc[:, 'possession_throw'] = 1

    if 'score_diff' in opponent_data.columns:
        opponent_data.loc[:, 'score_diff'] = -opponent_data.loc[:, 'score_diff']
    opponent_data = opponent_data.drop(['receiver_x', 'receiver_y'], axis=1)
    return opponent_data

def calculate_metric(data, group_keys, agg_func, new_name, col_to_extract=None):
    if col_to_extract is not None:
        return data.groupby(group_keys)[col_to_extract].agg(agg_func).reset_index(name=new_name)
    return data.groupby(group_keys).agg(agg_func).reset_index(name=new_name)

def calculate_goals(data):
    return calculate_metric(data[(data['receiver_y'] > 100) & (data['completion'])], ['receiver', 'year', 'gameID'], 'size', 'goals')

def calculate_assists(data):
    return calculate_metric(data[(data['completion']) & (data['receiver_y'] > 100)], ['thrower', 'year', 'gameID'], 'size', 'assists')

def calculate_completion_percentage(data):
    throws = calculate_metric(data, ['thrower', 'year', 'gameID'], 'size', 'total_throws')
    completions = calculate_metric(data[data['completion'] == 1], ['thrower', 'year', 'gameID'], 'size', 'completed_throws')
    completion = throws.merge(completions, on=['thrower', 'year', 'gameID'], how='left').fillna(0)
    completion['completion_percentage'] = completion['completed_throws'] / completion['total_throws'] * 100
    return completion[['thrower', 'year', 'gameID', 'completion_percentage']]

def calculate_yards(data, group_key, yard_col):
    successful_throws = data[data['completion'] == 1]
    successful_throws[yard_col] = successful_throws['receiver_y'].clip(0, 100) - successful_throws['thrower_y']
    return calculate_metric(successful_throws, group_key, 'sum', yard_col, yard_col)

def calculate_turnovers(data):
    # Assuming you have a way to calculate turnovers (e.g., based on specific conditions like mistakes or failed completions)
    return calculate_metric(data[(data['turnover'] == 1)], ['thrower', 'year', 'gameID'], 'size', 'turnovers')

def create_stats(data, stats):
    """this function calculates all metrics both novel and traditional for the data"""
    def calculate_stat(metric):
        thrower_sum = calculate_metric(data, ['thrower', 'year', 'gameID'], 'sum', f"thrower_{metric}_sum", metric).rename(columns={'thrower': 'player'})
        receiver_sum = calculate_metric(data, ['receiver', 'year', 'gameID'], 'sum', f"receiver_{metric}_sum", metric).rename(columns={'receiver': 'player'})
        thrower_mean = calculate_metric(data, ['thrower', 'year', 'gameID'], 'mean', f"thrower_{metric}_mean", metric).rename(columns={'thrower': 'player'})
        receiver_mean = calculate_metric(data, ['receiver', 'year', 'gameID'], 'mean', f"receiver_{metric}_mean", metric).rename(columns={'receiver': 'player'})
        return thrower_sum, thrower_mean, receiver_sum, receiver_mean
    
    metrics = {
        'goals': calculate_goals(data).rename(columns={'receiver': 'player'}),
        'assists': calculate_assists(data).rename(columns={'thrower': 'player'}),
        'turnovers': calculate_turnovers(data).rename(columns={'thrower': 'player'}), 
        'completion_percentage': calculate_completion_percentage(data).rename(columns={'thrower': 'player'}),
        'offensive_possessions': calculate_metric(data.drop_duplicates(['thrower', 'year', 'gameID', 'home_team_score', 'away_team_score', 'possession_num', 'game_quarter']),
                                                   ['thrower', 'year', 'gameID'], 'size', 'offensive_possessions').rename(columns={'thrower': 'player'}),
        'completions': calculate_metric(data[data['completion'] == 1], ['thrower', 'year', 'gameID'], 'size', 'completions').rename(columns={'thrower': 'player'}),
        'throwing_yards': calculate_yards(data, ['thrower', 'year', 'gameID'], 'throwing_yards').rename(columns={'thrower': 'player'}),
        'receiving_yards': calculate_yards(data, ['receiver', 'year', 'gameID'], 'receiving_yards').rename(columns={'receiver': 'player'}),
        # 'etv_decision': calculate_metric(data, ['thrower', 'year', 'gameID'], 'mean', 'etv_decision', 'etv_decision').rename(columns={'thrower': 'player'}),
        'hockey_assists': calculate_metric(data[data['hockey_assist'] == 1], ['thrower', 'year', 'gameID'], 'sum', 'hockey_assists', 'hockey_assist').rename(columns={'thrower': 'player'}),
        'games_played': calculate_metric(data, ['thrower', 'year', 'gameID'], 'count', 'games_played', 'gameID').rename(columns={'thrower': 'player'}),
        'offensive_team_goals': calculate_metric(data.drop_duplicates(['thrower', 'year', 'gameID', 'home_team_score', 'away_team_score', 'possession_num', 'game_quarter']),
                                                   ['thrower', 'year', 'gameID'], 'sum', 'offensive_team_goals', 'point_outcome').rename(columns={'thrower': 'player'}),
    }


    for stat_type in stats:
        metrics[f'thrower_{stat_type}_sum'], metrics[f'thrower_{stat_type}_mean'], metrics[f'receiver_{stat_type}_sum'], metrics[f'receiver_{stat_type}_mean'] = calculate_stat(stat_type)

    season_stats = None
    for _, metric_df in metrics.items():
        if season_stats is None:
            season_stats = metric_df
        else:
            season_stats = pd.merge(season_stats, metric_df, on=['player', 'year', 'gameID'], how='outer')
    _, cpoe, _, _ = calculate_stat('cpoe')
    season_stats['cpoe'] = cpoe.thrower_cpoe_mean
    season_stats['total_yards'] = season_stats['throwing_yards'].fillna(0) + season_stats['receiving_yards'].fillna(0)
    season_stats['total_scores'] = season_stats['goals'].fillna(0) + season_stats['assists'].fillna(0)
    season_stats['offensive_efficiency'] = season_stats['offensive_team_goals'].fillna(0) / season_stats['offensive_possessions']
    season_stats['plus_minus'] = (season_stats['goals'] + season_stats['assists'] - season_stats['turnovers']).fillna(0)

    for stat_type in stats:
        season_stats[f'total_{stat_type}'] = season_stats[f'thrower_{stat_type}_sum'].fillna(0) + season_stats[f'receiver_{stat_type}_sum'].fillna(0)

    season_stats = season_stats.drop(['offensive_team_goals'], axis=1)

    season_stats[['goals', 'assists', 'offensive_possessions', 'completions']] = season_stats[
        ['goals', 'assists', 'offensive_possessions', 'completions']].fillna(0)
    return season_stats