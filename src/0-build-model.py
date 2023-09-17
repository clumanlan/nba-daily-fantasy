
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import BoxScoreAdvancedV2, LeagueGameLog, TeamPlayerDashboard, BoxScoreDefensive, commonplayerinfo
import time as time
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import time
import awswrangler as wr
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# to do: 
#   try xgboost
#   try neurnal net
#   visualize 100 random sample points
#   figure out which model works better
#   try shap/lime to figure out most important variables
# extra:
#   explicitly define what each column is tht is being pulled in (game header to start)

#%%
# Custom Functions ------------------------------------------
# converts minutes played into total seconds --------------------
def get_sec(time_str):
    """Get seconds from time."""
    if ':' in time_str:
        if '.' in time_str:
            time_str = time_str.replace('.000000', '')
            m, s = time_str.split(':')
            time_sec = int(m) *60 + int(s)
        else:
            m, s = time_str.split(':')
            time_sec = int(m)*60 + int(s)
    
    if ':' not in time_str:
        if time_str == 'None':
            time_sec = 0
        else: 
            time_sec = int(time_str)*60

    return time_sec

# calculates fantasy points based on draftkings 
def calculate_fantasy_points(df):
        return df['PTS'] + df['FG3M']*0.5 + df['REB']*1.25 + df['AST']*1.5 + df['STL']*2 + df['BLK']*2 - df['TO']*0.5


# define statistics we pull of stats
def create_aggregate_rolling_functions(window_num = 20, window_min = 1):
    ## aggregate rolling functions to create summary stats
    f_min = lambda x: x.rolling(window=window_num, min_periods=window_min).min() 
    f_max = lambda x: x.rolling(window=window_num, min_periods=window_min).max()
    f_mean = lambda x: x.rolling(window=window_num, min_periods=window_min).mean()
    f_std = lambda x: x.rolling(window=window_num, min_periods=window_min).std()
    f_sum = lambda x: x.rolling(window=window_num, min_periods=window_min).sum()

    return f_min, f_max, f_mean, f_std, f_sum

# Function to lag all stats and calculate basic statistics of them from above

def create_lagged_player_stats(df, rel_num_cols):
    
    df = df.sort_values(['GAME_DATE_EST'])

    df = (
        df.assign(**{
        f'player_{col}_lagged': df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['GAME_DATE_EST']).groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return df


def create_lagged_team_stats(df, rel_num_cols):
    
    df = df.sort_values(['GAME_DATE_EST'])

    df = (
        df.assign(**{
        f'team_{col}_lagged': df.groupby(['TEAM_ID', 'SEASON_ID'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['GAME_DATE_EST']).groupby(['TEAM_ID', 'SEASON_ID'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return df


#%%

# READ IN DATA & QUICK PROCESSING -------------------------------------------------------
#%%
player_info_path = "s3://nbadk-model/player_info"

player_info_df = wr.s3.read_parquet(
    path=player_info_path,
    path_suffix = ".parquet" ,
    use_threads =True
)
player_info_df = player_info_df[['PERSON_ID', 'HEIGHT', 'POSITION']].drop_duplicates()
player_info_df = player_info_df.rename({'PERSON_ID': 'PLAYER_ID'}, axis=1)

game_stats_path_initial = "s3://nbadk-model/game_stats/game_header/initial"
game_stats_path_rolling = "s3://nbadk-model/game_stats/game_header/rolling"

game_headers_initial_df = wr.s3.read_parquet(
    path=game_stats_path_initial,
    path_suffix = ".parquet" ,
    use_threads =True
)

game_header_rolling_df = wr.s3.read_parquet(
    path=game_stats_path_rolling,
    path_suffix = ".parquet" ,
    use_threads =True
)

game_headers_df = pd.concat([game_headers_initial_df, game_header_rolling_df])


game_headers_df = (game_headers_df
    .assign(
        gametype_string = game_headers_df.GAME_ID.str[:3],
        game_type = lambda x: np.where(x.gametype_string == '001', 'Pre-Season',
            np.where(x.gametype_string == '002', 'Regular Season',
            np.where(x.gametype_string == '003', 'All Star',
            np.where(x.gametype_string == '004', 'Post Season',
            np.where(x.gametype_string == '005', 'Play-In Tournament', 'unknown'))))),
        GAME_ID = game_headers_df['GAME_ID'].astype(str),
        GAME_DATE_EST = pd.to_datetime(game_headers_df['GAME_DATE_EST'])

    )
)

game_headers_df = game_headers_df.drop_duplicates(subset=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'])

rel_cols = ['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS', 'HOME_TEAM_LOSSES']

game_headers_df_processed_filtered = game_headers_df[rel_cols]
game_headers_df_processed_filtered = game_headers_df_processed_filtered.drop_duplicates()


boxscore_trad_player_path = "s3://nbadk-model/player_stats/boxscore_traditional/"

boxscore_trad_player_df = wr.s3.read_parquet(
    path=boxscore_trad_player_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_trad_player_df['GAME_ID'] = boxscore_trad_player_df['GAME_ID'].astype(str)

boxscore_trad_team_path = "s3://nbadk-model/team_stats/boxscore_traditional/"
boxscore_trad_team_df = wr.s3.read_parquet(
    path=boxscore_trad_team_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_trad_team_df['GAME_ID'] = boxscore_trad_team_df['GAME_ID'].astype(str)
boxscore_trad_team_df = boxscore_trad_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

boxscore_adv_player_path = "s3://nbadk-model/player_stats/boxscore_advanced/"
boxscore_adv_player_df = wr.s3.read_parquet(
    path=boxscore_adv_player_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_adv_player_df = boxscore_adv_player_df.drop_duplicates(subset=['GAME_ID','PLAYER_ID'])

boxscore_adv_team_path = "s3://nbadk-model/team_stats/boxscore_advanced/"
boxscore_adv_team_df = wr.s3.read_parquet(
    path=boxscore_adv_team_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_adv_team_df = boxscore_adv_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

# create long table in order to flag teams that are home or away
game_home_away = game_headers_df[['GAME_ID','HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
game_home_away = pd.melt(game_home_away, id_vars='GAME_ID', value_name='TEAM_ID', var_name='home_away')
game_home_away['home_away'] = game_home_away['home_away'].apply(lambda x: 'home' if x == 'HOME_TEAM_ID' else 'away')


# Merge tables to create two dataframes at the player and team level  -----------------------------------------------------------

# Player Level DF ------------------------------------
# Merge the player info dataframe to add player positions
boxscore_complete_player = pd.merge(boxscore_trad_player_df, player_info_df, on='PLAYER_ID', how='left')

boxscore_complete_player = pd.merge(
    boxscore_complete_player,
    boxscore_adv_player_df,
    on=['GAME_ID', 'PLAYER_ID', 'TEAM_ID'],
    how='left',
    suffixes=['', '_adv']
)

# Merge the filtered game headers dataframe to add game information
game_info_df = game_headers_df_processed_filtered[['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST']]
boxscore_complete_player = pd.merge(boxscore_complete_player, game_info_df, on='GAME_ID', how='left')

# Filter out pre-season and all-star-games 
boxscore_complete_player = boxscore_complete_player[~boxscore_complete_player['game_type'].isin(['Pre-Season', 'All Star'])]

# Shawn Marion plays in two games in the 2007 season so we remove the one with less stats
boxscore_complete_player[(boxscore_complete_player['PLAYER_ID']==1890) & (boxscore_complete_player['GAME_DATE_EST']=='2007-12-19')]
shawn_marion_index_to_drop = boxscore_complete_player[(boxscore_complete_player['PLAYER_ID']==1890) & (boxscore_complete_player['GAME_ID']=='0020700367')].index
boxscore_complete_player = boxscore_complete_player.drop(shawn_marion_index_to_drop[0])

# Team Level DF ------------------------------------
boxscore_complete_team = pd.merge(
    boxscore_trad_team_df,
    boxscore_adv_team_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left',
    suffixes=['', '_adv']
)

boxscore_complete_team = pd.merge(boxscore_complete_team, game_home_away, how='left', on=['GAME_ID', 'TEAM_ID'])
boxscore_complete_team = pd.merge(boxscore_complete_team, game_info_df, on='GAME_ID', how='left')
boxscore_complete_team = boxscore_complete_team[~boxscore_complete_team['game_type'].isin(['Pre-Season', 'All Star'])]

del player_info_df, game_headers_df, game_headers_df_processed_filtered, boxscore_trad_player_df, boxscore_trad_team_df, boxscore_adv_player_df, game_home_away
#%%



boxscore_complete_player, boxscore_complete_team




# CREATE PLAYER LEVEL FEATURES -------------------------------------------------------------------------
#%%


boxscore_complete_player_processed = (boxscore_complete_player
    .assign(
        seconds_played = lambda x: x['MIN'].apply(get_sec),
        fantasy_points = calculate_fantasy_points(boxscore_complete_player),
        SEASON_ID = lambda x: x['SEASON'].astype('int'),
        Player_ID = lambda x: x['PLAYER_ID'].astype('str'),
        is_starter = np.where(boxscore_complete_player['START_POSITION']=="", False, True)
        )
    .sort_values(by=['GAME_DATE_EST'])
    .reset_index(drop=True)
    .drop(['MIN'], axis=1)
)


# calculate rolling games played 

boxscore_complete_player_processed['rolling_games_played_season'] = (
    boxscore_complete_player_processed
    .sort_values('GAME_DATE_EST')
    .groupby(['PLAYER_ID','SEASON'])['GAME_ID']
    .rolling(window=100)
    .count()
    .droplevel([0,1])
    .sort_index()
    .reset_index(drop=True)
)

boxscore_complete_player_processed['rolling_games_played_season_lagged'] = boxscore_complete_player_processed.groupby(['PLAYER_ID','SEASON'])['rolling_games_played_season'].shift(1)


boxscore_complete_player_processed['rolling_games_played_overall'] = (
    boxscore_complete_player_processed
    .sort_values('GAME_DATE_EST')
    .groupby(['PLAYER_ID'])['GAME_ID']
    .rolling(window=300000)
    .count()
    .droplevel([0])
    .sort_index()
    .reset_index(drop=True)
)

boxscore_complete_player_processed['rolling_games_played_overall_lagged'] = boxscore_complete_player_processed.groupby(['PLAYER_ID'])['rolling_games_played_overall'].shift(1)

boxscore_complete_player_processed = boxscore_complete_player_processed.drop(['rolling_games_played_season', 'rolling_games_played_overall'], axis=1)



# add seconds_played average lagged so we can filter out data in a few different ways
boxscore_complete_player_processed['seconds_played_rolling_3game_avg_lag'] = (
    boxscore_complete_player_processed
    .sort_values(['GAME_DATE_EST'])
    .groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'])['seconds_played']
    .rolling(window=3)
    .mean()
    .reset_index(level=[0,1,2], drop=True)
    )

boxscore_complete_player_processed['seconds_played_rolling_3game_avg_lag'] = boxscore_complete_player_processed.sort_values(['GAME_DATE_EST']).groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'])['seconds_played_rolling_3game_avg_lag'].shift(1)

boxscore_complete_player_processed['seconds_played_rolling_5game_avg_lag'] = (
    boxscore_complete_player_processed
    .sort_values(['GAME_DATE_EST'])
    .groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'])['seconds_played']
    .rolling(window=5)
    .mean()
    .reset_index(level=[0,1,2], drop=True)
    )

boxscore_complete_player_processed['seconds_played_rolling_5game_avg_lag'] = boxscore_complete_player_processed.sort_values(['GAME_DATE_EST']).groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'])['seconds_played_rolling_5game_avg_lag'].shift(1)



# define columns we'll use and filter df
boxscore_complete_player_processed['fantasy_points_actual'] = boxscore_complete_player_processed['fantasy_points']

## rel id columns + target variable ------------
rel_cols_player_no_lag = ['GAME_ID', 'SEASON_ID', 'GAME_DATE_EST', 'is_starter', 'PLAYER_ID', 'PLAYER_NAME', 'START_POSITION', 'POSITION', 'TEAM_ID', 'fantasy_points_actual']

## num columns -----------
rel_num_cols = ['seconds_played', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A','FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS',
       'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
       'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
       'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
       'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE', 'fantasy_points']

## additional columns ------------------------
additional_cols =  ['rolling_games_played_season_lagged', 'rolling_games_played_overall_lagged', 'seconds_played_rolling_3game_avg_lag','seconds_played_rolling_5game_avg_lag']


f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()
boxscore_complete_player_processed = boxscore_complete_player_processed[rel_cols_player_no_lag + rel_num_cols + additional_cols]
boxscore_complete_player_processed = create_lagged_player_stats(boxscore_complete_player_processed, rel_num_cols)


boxscore_complete_player_processed = (
    boxscore_complete_player_processed
    .sort_values(['GAME_DATE_EST'])
    .assign(
        TEAM_ID = lambda x: x['TEAM_ID'].astype(str)
    )
)


boxscore_complete_player_processed['fantasy_points_lagged_mean']  = (
    boxscore_complete_player_processed
    .groupby(['PLAYER_ID', 'SEASON_ID'])
    ['player_fantasy_points_lagged']
    .rolling(window=100, min_periods=1)
    .mean()
    .rename('fantasy_points_lagged_mean')
    .reset_index()
    .set_index('level_2')
    .sort_index()
    ['fantasy_points_lagged_mean']
)

misassigned_2000 = (boxscore_complete_player_processed['GAME_ID'] == '0020000452') & (boxscore_complete_player_processed['PLAYER_ID'].isin([754, 1533, 1683]))

0020200888


boxscore_complete_player_processed = boxscore_complete_player_processed[~misassigned_2000]

def reindex_by_date(df):
    dates = pd.date_range(df.GAME_DATE_EST.min(), df.GAME_DATE_EST.max())
    return df.reindex(dates).ffill()

player_season_calendar_list = []
player_season_ranking_calendar_base = boxscore_complete_player_processed[['PLAYER_ID', 'POSITION', 'fantasy_points_lagged_mean', 'SEASON_ID', 'GAME_DATE_EST']]

for season in boxscore_complete_player_processed.SEASON_ID.unique():

    df = player_season_ranking_calendar_base[player_season_ranking_calendar_base['SEASON_ID']==season]
    df.index = pd.DatetimeIndex(df.GAME_DATE_EST)

    df = df.groupby('PLAYER_ID').apply(reindex_by_date).reset_index(0, drop=True)
    player_season_calendar_list.append(df)

    print(season)



df = player_season_ranking_calendar_base[player_season_ranking_calendar_base['SEASON_ID']==2002]
df.index = pd.DatetimeIndex(df.GAME_DATE_EST)


df = df.reset_index(drop=True)
df[['GAME_DATE_EST', 'PLAYER_ID']].value_counts()



df = df.groupby('PLAYER_ID').apply(reindex_by_date).reset_index(0, drop=True)
player_season_calendar_list.append(df)


s
[754,1533, 1683]
boxscore_complete_player_processed[(boxscore_complete_player_processed['GAME_DATE_EST']=='2003-03-07') & (boxscore_complete_player_processed['PLAYER_ID'].isin([238]))].sort_values('PLAYER_ID')

boxscore_complete_player_processed[boxscore_complete_player_processed['GAME_ID']=='0020000453']
    print(season)



# create rankings then you can also bin these rankings as well


player_season_ranking_calendar_df = pd.concat(player_season_calendar_list)
player_season_ranking_calendar_df = player_season_ranking_calendar_df[['PLAYER_ID', 'POSITION', 'fantasy_points_lagged_mean', 'SEASON_ID']].reset_index(names='calendar_date')
player_season_ranking_calendar_df = player_season_ranking_calendar_df.dropna(subset='fantasy_points_lagged_mean')
player_season_ranking_calendar_df['fantasy_points_rank_overall'] = player_season_ranking_calendar_df.groupby('calendar_date')['fantasy_points_lagged_mean'].rank(ascending=False)
#player_season_ranking_calendar_df = player_season_ranking_calendar_df.drop(['POSITION', 'SEASON_ID', 'fantasy_points_lagged_mean'], axis=1)

player_test = player_season_ranking_calendar_df[player_season_ranking_calendar_df['PLAYER_ID'] == 3]
day_test = player_season_ranking_calendar_df[player_season_ranking_calendar_df['calendar_date'] == '2001-11-05']

day_test.sort_values(by='fantasy_points_rank_overall')

player_season_ranking_calendar_df.PLAYER_ID.unique()

#%%


# CREATE TEAM LEVEL FEATURES -------------------------------------------------------------------------
#%%

boxscore_complete_team['seconds_played'] = boxscore_complete_team['MIN'].apply(get_sec)

boxscore_complete_team_processed = (boxscore_complete_team
    .assign(
        seconds_played = lambda x: x['MIN'].apply(get_sec),
        fantasy_points_team = calculate_fantasy_points(boxscore_complete_team),
        SEASON_ID = lambda x: x['SEASON'].astype('int'),
        TEAM_ID = lambda x: x['TEAM_ID'].astype('str')
        )
    .drop('MIN', axis=1)
    )


# Get fantasy points allowed from other team
fantasy_points_allowed = boxscore_complete_team_processed[['GAME_ID', 'TEAM_ID', 'fantasy_points_team']]
boxscore_complete_team_processed = pd.merge(boxscore_complete_team_processed, fantasy_points_allowed, on='GAME_ID', suffixes=['', '_opposing'], how='left')
boxscore_complete_team_processed = boxscore_complete_team_processed[boxscore_complete_team_processed['TEAM_ID'] != boxscore_complete_team_processed['TEAM_ID_opposing']]

# Lag team fantasy points and opposing team fantasy points
boxscore_complete_team_processed = boxscore_complete_team_processed.sort_values(['GAME_DATE_EST'])

team_group_shift = boxscore_complete_team_processed.groupby(['TEAM_ID'])[['fantasy_points_team', 'fantasy_points_team_opposing']].shift(1)
team_group_shift.columns = ['fantasy_points_team_lagged', 'fantasy_points_team_allowed_lagged']

boxscore_complete_team_processed = pd.concat([boxscore_complete_team_processed, team_group_shift], axis=1)

rolling_avg_team = boxscore_complete_team_processed.groupby(['TEAM_ID', 'SEASON_ID'])[['fantasy_points_team', 'fantasy_points_team_opposing']].rolling(window=100, min_periods=1).mean()
rolling_avg_team = rolling_avg_team.rename(columns={'fantasy_points_team': 'fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing': 'fantasy_points_team_opposing_lagged_mean'})
rolling_avg_team = rolling_avg_team.reset_index().set_index('level_2').sort_index()

boxscore_complete_team_processed[['fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing_lagged_mean']] = rolling_avg_team[['fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing_lagged_mean']]


# Create team ranking of fantasy points scored & allowed
boxscore_complete_team_processed[boxscore_complete_team_processed.index.duplicated()]

def reindex_by_date(df):
    dates = pd.date_range(df.GAME_DATE_EST.min(), df.GAME_DATE_EST.max())
    return df.reindex(dates).ffill()

team_season_calendar_list = []

for season in boxscore_complete_team_processed['SEASON_ID'].unique():
    df = boxscore_complete_team_processed[boxscore_complete_team_processed['SEASON_ID']==season]
    df.index = pd.DatetimeIndex(df.GAME_DATE_EST)

    df = df.groupby('TEAM_ID').apply(reindex_by_date).reset_index(0, drop=True)
    team_season_calendar_list.append(df)

    print(season)


team_season_lagged_ranking_df = pd.concat(team_season_calendar_list)

team_season_lagged_ranking_df = team_season_lagged_ranking_df[['TEAM_ID', 'SEASON_ID', 'fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing_lagged_mean']].reset_index(names='calendar_date')
team_season_lagged_ranking_df = team_season_lagged_ranking_df.dropna(subset='fantasy_points_team_lagged_mean')

# Rank team's fantasy points scored
team_season_lagged_ranking_df['fantasy_points_rank_overall_lagged_team'] = team_season_lagged_ranking_df.groupby('calendar_date')['fantasy_points_team_lagged_mean'].rank(ascending=False)

# Rank how many points they give up to their opponent with the team giving up most points being #1
team_season_lagged_ranking_df['fantasy_points_rank_overall_lagged_team_allowed'] = team_season_lagged_ranking_df.groupby('calendar_date')['fantasy_points_team_opposing_lagged_mean'].rank() 
team_season_lagged_ranking_df = team_season_lagged_ranking_df[['calendar_date', 'TEAM_ID', 'fantasy_points_rank_overall_lagged_team', 'fantasy_points_rank_overall_lagged_team_allowed']]




# Create lagged team stats and basic stats from them, same as player plus additional fantasy points

rel_team_cols_no_lag = ['GAME_ID', 'TEAM_ID', 'GAME_DATE_EST', 'SEASON_ID',  'home_away', 'game_type']
rel_num_cols_team = rel_num_cols + ['fantasy_points_team']

boxscore_complete_team_processed = boxscore_complete_team_processed[rel_team_cols_no_lag + rel_num_cols_team]
boxscore_complete_team_processed = create_lagged_team_stats(boxscore_complete_team_processed, rel_num_cols_team)


# Add ranking to the team boxscore
boxscore_complete_team_processed = pd.merge(boxscore_complete_team_processed, team_season_lagged_ranking_df, left_on= ['GAME_DATE_EST', 'TEAM_ID'], right_on =['calendar_date', 'TEAM_ID'], how='left')
boxscore_complete_team_processed = boxscore_complete_team_processed.drop(['SEASON_ID', 'GAME_DATE_EST', 'SEASON_ID'], axis=1)

# join team boxscore on itself so we can get opposing team stats
boxscore_complete_team_processed = pd.merge(boxscore_complete_team_processed, boxscore_complete_team_processed, on='GAME_ID', how='left', suffixes=['', '_opposing'])
boxscore_complete_team_processed = boxscore_complete_team_processed[boxscore_complete_team_processed['TEAM_ID'] != boxscore_complete_team_processed['TEAM_ID_opposing']]



# Merge player and team dataframes ------------------------------------------------
combined_player_team_boxscore = pd.merge(boxscore_complete_player_processed, boxscore_complete_team_processed, how='left', on=['GAME_ID', 'TEAM_ID'])
#%%