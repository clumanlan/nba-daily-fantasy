import pandas as pd
import numpy as np
import datetime
import awswrangler as wr
import time as time
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.fbprophet import Prophet

from nba_api.stats.static import teams, players

from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players

dk_salaries = pd.read_csv("projects/nba-daily-fantasy/data/dk_salaries_2023_01_16.csv")


# GET PLAYER INFO ------------------------------------------

players_list = players.get_players()
players_df = pd.DataFrame(players_list)
players_df['id'] = players_df['id'].astype(str)

common_player_info_complete_list = []
error_player_info_list = []


loop_place = 0
players_df_length = len(players_df)

for id in players_df['id']:
    
    loop_place += 1

    try: 
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=id)
        common_player_info_df = player_info.common_player_info.get_data_frame()

        common_player_info_complete_list.append(common_player_info_df)
        
    except Exception as e:
        error_player_info_list.append(id)


    print(id, '% complete: ', str(round((loop_place/players_df_length) *100, 2)) + '%')


    time.sleep(1.01)



common_player_info_complete_df = pd.concat(common_player_info_complete_list)

common_player_info_complete_df.to_parquet('projects/nba-daily-fantasy/data/common_player_info_complete_df.parquet')


error_player_info_df = pd.DataFrame(error_player_info_list, columns=['player_id_error'])

error_player_info_df.to_parquet('projects/nba-daily-fantasy/data/error_player_info_df.parquet')




common_player_info_complete_df_filtered = common_player_info_complete_df[['PERSON_ID', 'FROM_YEAR', 'TO_YEAR', 'DISPLAY_FIRST_LAST', 'NBA_FLAG']]
common_player_info_filtered_currently_active = common_player_info_complete_df_filtered[(common_player_info_complete_df_filtered['TO_YEAR'] == 2022) & (common_player_info_complete_df_filtered['NBA_FLAG'] == 'Y')]


active_playergamelog_list = []
players_info_length = len(common_player_info_filtered_currently_active)
loop_place = 0

error_player_gamelog_list = []

for id in common_player_info_filtered_currently_active['PERSON_ID']:

    print(str(id) + ' starting')
    
    for i in range(2000,2023):
        try:
            gamelog = pd.concat(playergamelog.PlayerGameLog(player_id=id, season=i).get_data_frames())

            if gamelog.shape[0] != 0:
                active_playergamelog_list.append(gamelog)
                print(str(id), str(i), ' processing')


        except Exception as e:
            error_player_gamelog_list.append(id)

        time.sleep(1.01)
    
    loop_place += 1
    print(str(round((loop_place/players_info_length) *100, 2)) + '%')
   


active_playergamelog_df = pd.concat(active_playergamelog_list)
active_playergamelog_df["GAME_DATE"] = pd.to_datetime(active_playergamelog_df["GAME_DATE"], format="%b %d, %Y")





active_playergamelog_df.to_parquet('projects/nba-daily-fantasy/data/active_playergamelog_2023_01_19.parquet')

error_id_df_2023_active = pd.DataFrame(error_player_gamelog_list)
error_id_df_2023_active.to_csv('projects/nba-daily-fantasy/data/error_id_df_2023_active.csv')

active_playergamelog_df = pd.read_parquet('projects/nba-daily-fantasy/data/active_playergamelog_2022_12_27.parquet')
 

# GET DATA GOING FORWARD WILL BE 



## IMPUTE MISSING VALUES WITH PROPHET IMPUTER --------------------------




# CHECK TO SEE IF THEY MATCH

dk_active_players = dk_salaries[['Name', 'Roster Position', 'Salary']]
active_player_prophet_unique = pd.Series(active_player_prophet_wnames['full_name'].unique(), name='full_name')


test = dk_active_players.merge(active_player_prophet_unique, left_on='Name', right_on='full_name', how='left')


active_player_prophet_wnames[active_player_prophet_wnames['full_name'].str.contains('James')]


## FILTER RELEVANT PLAYERS AND FORECAST VALUES ------------------------

active_player_prophet_wnames.set_index([''])
active_player_prophet_wnames.columns

active_player_prophet_filtered = active_player_prophet_wnames[active_player_prophet_wnames['full_name'].isin(dk_active_players['Name'])]

active_player_prophet.filter(full_name = dk_active_players)

forecaster.fit(active_player_prophet)  

fh_ten = np.arange(1, 10)
y_pred = forecaster.predict(fh=fh_ten)


forecaster.predict()
y_pred



# BUILD AN OPTIMIZER 





# COMPARE VALUES VERSUS DRAFTKINGS AND ROTOWIRE !!!!!
























# APPENDIX ----------------------------------------------------------------------


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


fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

# drift imputer 

drift_impute_trans = Imputer(method="drift") # maybe change to forecaster latser 
player_sample_kdwestbrook_drift = drift_impute_trans.fit_transform(player_sample_pts_reindex)
player_sample_kdwestbrook_drift.columns = ['kd_pts_drift', 'westbrook_pts_drift']


date_index = pd.date_range('1/1/2010', periods=3650, freq='D')

player_sample_pts_series = player_sample_pts.reindex(date_index)



# INSERT LOGIC THAT GETS FIRST DAY THEY PLAYED * A GIVEN MULTIPLE:

player_sample_pts_series = player_sample_pts['PTS']
y_pred = forecaster.fit(player_sample_pts_series, fh=[1, 2]).predict()
y_pred.GAME_DATE


player_sample_pts_y_hat = transformer.fit_transform(player_sample_pts_series)



# IMPUTE MISSING VALUES ---------------------------------------------------


player_sample = active_playergamelog_df_train[(active_playergamelog_df_train['full_name'] == 'Kevin Durant') | (active_playergamelog_df_train['full_name'] == 'Russell Westbrook')]
player_sample_pts = player_sample[['full_name', 'GAME_DATE', 'PTS']]

player_start_n_end_dates = (player_sample_pts
    .groupby(['full_name'])['GAME_DATE']
    .agg({'min','max'})
    .reset_index()
)

player_sample_pts.set_index(['full_name', 'GAME_DATE'],  inplace=True)


player_sample_pts_reindex = (player_sample_pts
        .unstack(level=0)
        .reindex(pd.date_range(start='2010-01-01',
                   end='2018-06-01', freq='D'))
)

player_sample_pts_reindex.stack('full_name', dropna=False).swaplevel(0,1).sort_index()


from sktime.transformations.series.impute import Imputer
from sktime.forecasting.fbprophet import Prophet
forecaster = Prophet()


forecaster_impute_trans = Imputer(method='forecaster', forecaster=forecaster)
player_sample_kdwestbrook_prophet = forecaster_impute_trans.fit_transform(player_sample_pts_reindex)
player_sample_kdwestbrook_prophet.columns = ['kd_pts_prophet', 'westbrook_pts_prophet']

player_sample_impute_merged = pd.concat([player_sample_kdwestbrook_drift, player_sample_kdwestbrook_prophet], axis=1)


sns.lineplot(data=player_sample_impute_merged[['westbrook_pts_drift', 'westbrook_pts_prophet']])
# APPLY PROPHET MODEL HERE: 

from sktime.forecasting.fbprophet import Prophet
forecaster = Prophet()

forecaster.fit(player_sample_pts_y_hat)  
y_pred = forecaster.predict(fh=[1,2,3])






# NOW WE PANEL FORECAST? 