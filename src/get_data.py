import pandas as pd
import numpy as np
import datetime
import awswrangler as wr
import time as time
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

from nba_api.stats.static import teams, players

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players


players = players.get_players()
players_df = pd.DataFrame(players)
active_players = players_df[players_df['is_active'] == True]

active_playergamelog_list = []
active_players_length = len(active_players)
loop_place = 0


error_id_list = []

for id in active_players['id']:

    print(str(id) + ' starting')
    
    for i in range(2000,2023):
        try:
            gamelog = pd.concat(playergamelog.PlayerGameLog(player_id=id, season=i).get_data_frames())

            if gamelog.shape[0] != 0:
                active_playergamelog_list.append(gamelog)


        except Exception as e:
            error_id_list.append(id)

        time.sleep(1.01)
    
    loop_place += 1
    print(str(round((loop_place/active_players_length) *100, 2)) + '%')
   



active_playergamelog_df = pd.concat(active_playergamelog_list)
active_playergamelog_df["GAME_DATE"] = pd.to_datetime(active_playergamelog_df["GAME_DATE"], format="%b %d, %Y")

#active_playergamelog_df.to_parquet('projects/nba-daily-fantasy/data/active_playergamelog_2022_12_27.parquet')

error_id_df_2023_active = pd.DataFrame(error_id_list)
error_id_df_2023_active.to_csv('projects/nba-daily-fantasy/data/error_id_df_2023_active.csv')

active_playergamelog_df = pd.read_parquet('projects/nba-daily-fantasy/data/active_playergamelog_2022_12_27.parquet')

active_playergamelog_df_train = active_playergamelog_df[active_playergamelog_df['GAME_DATE'] <= "2018-06-01"]


teams.get_teams()
players = players.get_players()
players = pd.DataFrame(players)
active_players = players[players['is_active'] == True]

# PROPHET TEST ------------------------------------------------
active_playergamelog_df_train = active_playergamelog_df_train.merge(active_players, how='left', left_on='Player_ID', right_on = 'id')

player_sample_kd = active_playergamelog_df_train[active_playergamelog_df_train['full_name'] == 'Kevin Durant']

player_sample_kd_train_train = player_sample_kd[player_sample_kd['GAME_DATE'] <= "2017-06-01"]
player_sample_kd_train_valid = player_sample_kd[player_sample_kd['GAME_DATE'] > "2017-06-01"]


player_sample_kd_train_train_pts = player_sample_kd_train_train[['GAME_DATE','PTS']]
player_sample_kd_train_valid_pts = player_sample_kd_train_valid[['GAME_DATE','PTS']]

player_sample_kd_train_train_pts.columns = ['ds', 'y']
player_sample_kd_train_valid_pts.columns = ['ds', 'y']



m = Prophet()
m.fit(player_sample_kd_train_train_pts)

future_kd_train = m.make_future_dataframe(periods=365)
forecast = m.predict(player_sample_kd_train_train_pts)


forecast_w_actuals

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)


# we could just fit all the forecasters and see which's performance is better?
from sktime.registry import all_estimators

for forecaster in all_estimators(filter_tags={"scitype:y": ["multivariate", "both"]}):
    print(forecaster[0])


# SKTIME HIERARCHAL SERIES -----------------------------------------------------------------------
player_sample = active_playergamelog_df_train[(active_playergamelog_df_train['full_name'] == 'Kevin Durant') | (active_playergamelog_df_train['full_name'] == 'Russell Westbrook')]
player_sample_pts = player_sample[['full_name', 'GAME_DATE', 'PTS']]

player_start_n_end_dates = (player_sample_pts
    .groupby(['full_name'])['GAME_DATE']
    .agg({'min','max'})
    .reset_index()
)


player_sample_pts.set_index(['full_name', 'GAME_DATE'],  inplace=True)


player_sample_pts




# HOW DO WE FILL IN MISSING VALUES FOR SKTIME?
from sktime.transformations.series.impute import Imputer

# INSERT LOGIC THAT GETS FIRST DAY THEY PLAYED * A GIVEN MULTIPLE:

player_sample_pts_series = player_sample_pts['PTS']
y_pred = forecaster.fit(player_sample_pts_series, fh=[1, 2]).predict()
y_pred.GAME_DATE


player_sample_pts_y_hat = transformer.fit_transform(player_sample_pts_series)


# APPLY PROPHET MODEL HERE: 

from sktime.forecasting.fbprophet import Prophet
forecaster = Prophet()

forecaster.fit(player_sample_pts_y_hat)  
y_pred = forecaster.predict(fh=[1,2,3])






# NOW WE PANEL FORECAST? 