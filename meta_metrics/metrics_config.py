baseline_metrics_total = ['goals',
 'assists',
 'completion_percentage',
 'offensive_possessions',
 'completions',
 'throwing_yards',
 'receiving_yards',
 'hockey_assists',
 'games_played',
 'offensive_efficiency',
 'total_yards',
 'total_scores',
 'plus_minus']

baseline_metrics_per_possession = ['goals_per_possession',
 'assists_per_possession',
 'completions_per_possession',
 'throwing_yards_per_possession',
 'receiving_yards_per_possession',
 'hockey_assists_per_possession',
 'total_yards_per_possession',
 'total_scores_per_possession',
 'plus_minus_per_possession']

## possible novel metrics
novel_metrics_total = [
    # 'total_etv', #low independence
 'total_ec_normalized', #good!
 'total_ec', #good!
#  'total_etv_decision', #decent but associated with volume
#  'total_disc_advancement', #good!
#  'total_etv_per_possession', #low independence
#  'total_ec_cp_per_possession', #low stability
#  'total_etv_decision_per_possession', #low independence
#  'total_disc_advancement_per_possession' #good!
]

novel_metrics_partial = [
    'thrower_etv_sum', #low independence
#  'thrower_etv_mean', #low stability
#  'receiver_etv_sum', #low independence
#  'receiver_etv_mean', #low stability
 'thrower_ec_normalized_sum', #good! slight association with volume
 'thrower_ec_sum', #good! slight association with volume
#  'thrower_ec_normalized_mean', #low stability
 'receiver_ec_normalized_sum', #good!
 'receiver_ec_sum', #good!
#  'receiver_ec_normalized_mean', #low stability
#  'thrower_etv_decision_sum', #good! but highly explained by volume
#  'thrower_etv_decision_mean', #low discrimination
#  'receiver_etv_decision_sum', #highly explained by volume
#  'receiver_etv_decision_mean', #low stability
#  'thrower_disc_advancement_sum', #good! 
#  'thrower_disc_advancement_mean', #low stability
#  'receiver_disc_advancement_sum', #good!
#  'receiver_disc_advancement_mean', #low discrimination
 'cpoe',
 'expected_cp',
 'relative_throw_value'] 

novel_metrics_per_possession = [
#  'thrower_etv_sum_per_possession', #low independence
#  'receiver_etv_sum_per_possession', #low independence
 'thrower_ec_normalized_sum_per_possession', #low stability
 'receiver_ec_normalized_sum_per_possession', #same as volume
 'thrower_ec_sum_per_possession', #low stability
 'receiver_ec_sum_per_possession', #same as volume
#  'thrower_etv_decision_sum_per_possession', #correlated with volume but high indepence
#  'receiver_etv_decision_sum_per_possession', #correlated with volume but decent indepence
#  'thrower_disc_advancement_sum_per_possession', #low stability
#  'receiver_disc_advancement_sum_per_possession', #good!
]


baseline_metrics = baseline_metrics_total + baseline_metrics_per_possession
novel_metrics = novel_metrics_partial+novel_metrics_total+novel_metrics_per_possession