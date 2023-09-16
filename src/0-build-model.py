
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

# This function converts minutes played into total seconds --------------------
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

# This function calculates fantasy points based on draftkings 
def calculate_fantasy_points(df):
        return df['PTS'] + df['FG3M']*0.5 + df['REB']*1.25 + df['AST']*1.5 + df['STL']*2 + df['BLK']*2 - df['TO']*0.5


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

# Player Level DF -------------------
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

del player_info_df, game_headers_df, game_headers_df, game_headers_df_processed_filtered, boxscore_trad_player_df, boxscore_trad_team_df, boxscore_adv_player_df, game_home_away
#%%



boxscore_complete_player, boxscore_complete_team