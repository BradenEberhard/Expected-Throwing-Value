import pandas as pd 
import numpy as np

def categorize_distance(df):
    """
    Categorizes the 'throw_distance' column into 'short', 'medium', or 'long'
    based on predefined distance ranges, and adds the result as a new column 
    'distance_category' to the dataframe.
    """
    df['distance_category'] = pd.cut(
        df['throw_distance'], 
        bins=[0, 10, 40, 200], 
        labels=['short', 'medium', 'long'], 
        include_lowest=True
    ).astype(str)
    return df

def calculate_tfidf_category(df, category_column, thrower_column, tfidf_name):
    """
    Calculates the TF-IDF (Term Frequency - Inverse Document Frequency) score
    for each thrower and their associated category. Returns a dataframe with 
    the thrower, category, and calculated TF-IDF values.
    """
    tf = df.groupby([thrower_column, category_column]).size().reset_index(name='TF')
    df_counts = df[category_column].value_counts().reset_index()
    df_counts.columns = [category_column, 'DF']
    total_players = df[thrower_column].nunique()
    df_counts['IDF'] = np.log(total_players / df_counts['DF'])
    
    tf_idf = pd.merge(tf, df_counts, on=category_column)
    tf_idf[tfidf_name] = tf_idf['TF'] * tf_idf['IDF']
    
    return tf_idf[[thrower_column, category_column, tfidf_name]]

def categorize_direction(angle):
    """
    Categorizes the direction of the throw based on the angle provided. 
    The function returns 'forward', 'sideways', or 'backward' based on the angle value.
    """
    if -45 <= angle < 45:
        return 'forward'
    elif 45 <= angle < 135 or -135 <= angle < -45:
        return 'sideways'
    else:
        return 'backward'

def calculate_directions(df):
    """
    Calculates the direction of the throw in degrees based on the coordinates 
    of the thrower and receiver. Adjusts the calculated direction and adds it
    as a new column 'direction' to the dataframe.
    """
    df['direction'] = np.degrees(
        np.arctan2(df['receiver_y'] - df['thrower_y'], df['receiver_x'] - df['thrower_x'])
    ) - 45
    df.loc[df['direction'] < -180, 'direction'] += 360
    return df

def merge_tfidf_with_test_dfs(test_df, train_df, category_columns):
    """
    Merges the test dataframe with the relevant TF-IDF values from the training dataframe 
    based on the 'thrower' and category columns.
    """
    return test_df.merge(
        train_df[category_columns],
        on=['thrower'],
        how='left'
    )

def calculate_tfidf(train_df, test_dfs=None):
    """
    Calculates the TF-IDF values for the distance, direction, and combined distance-direction 
    categories in the training dataframe. If test dataframes are provided, it also applies 
    the same TF-IDF logic to them.
    """
    # Categorize distance for the training data
    train_df = categorize_distance(train_df)

    # Calculate TF-IDF for distance categories
    tf_idf_distance = calculate_tfidf_category(train_df, 'distance_category', 'thrower', 'distance_tfidf')
    train_df = train_df.merge(tf_idf_distance, on=['thrower', 'distance_category'], how='left')

    # Calculate and categorize directions for the training data
    train_df = calculate_directions(train_df)
    train_df['direction_category'] = train_df['direction'].apply(categorize_direction)

    # Calculate TF-IDF for direction categories
    tf_idf_direction = calculate_tfidf_category(train_df, 'direction_category', 'thrower', 'direction_tfidf')
    train_df = train_df.merge(tf_idf_direction, on=['thrower', 'direction_category'], how='left')

    # Combine distance and direction categories
    train_df['combined_category'] = train_df['distance_category'] + '-' + train_df['direction_category']

    # Calculate TF-IDF for combined categories
    tf_idf_combined = calculate_tfidf_category(train_df, 'combined_category', 'thrower', 'distance_direction_tfidf')
    train_df = train_df.merge(tf_idf_combined, on=['thrower', 'combined_category'], how='left')
    
    test_dfs_final = []
    
    # If test dataframes are provided, apply the same processing and merge with training TF-IDF
    if test_dfs is not None:
        for df in test_dfs:
            df = categorize_distance(df)
            df = calculate_directions(df)
            df['direction_category'] = df['direction'].apply(categorize_direction)
            df['combined_category'] = df['distance_category'] + '-' + df['direction_category']
            df = pd.merge(df, tf_idf_distance, on=['thrower', 'distance_category'], how='left')
            df = pd.merge(df, tf_idf_direction, on=['thrower', 'direction_category'], how='left')
            df = pd.merge(df, tf_idf_combined, on=['thrower', 'combined_category'], how='left')
            test_dfs_final.append(df)

    return train_df, test_dfs_final