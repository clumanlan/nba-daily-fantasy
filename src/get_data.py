import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import awswrangler as wr
import time as time
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.fbprophet import Prophet

from nba_api.stats.static import teams, players

from nba_api.stats.endpoints import playergamelog, commonplayerinfo, ScoreboardV2, BoxScoreAdvancedV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players

from datetime import date



def get_game_header_n_line_score(start_date):

    game_header_w_standings_list = []
    team_game_line_score_list = []
    error_dates_list = []

    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = date.today()

    current_date = start_date

    while current_date <= end_date:
        try:
            scoreboard = ScoreboardV2(game_date=current_date, league_id='00')

            game_header = scoreboard.game_header.get_data_frame()
            series_standings = scoreboard.series_standings.get_data_frame()
            series_standings.drop(['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST'], axis=1, inplace=True)

            game_header_w_standings = game_header.merge(series_standings, on='GAME_ID')
            game_header_w_standings_list.append(game_header_w_standings)

            # each line rpresents a game-teamid
            team_game_line_score = scoreboard.line_score.get_data_frame()
            team_game_line_score_list.append(team_game_line_score)
        
        except Exception as e:
            error_dates_list.append(current_date)
            print(f'error {current_date}')

        current_date += timedelta(days=1)
        print(current_date)

        time.sleep(1.1)

    game_header_w_standings_complete_df = pd.concat(game_header_w_standings_list)
    team_game_line_score_complete_df = pd.concat(team_game_line_score_list)
    error_dates_df = pd.concat(error_dates_list)

    game_header_w_standings_complete_df.reset_index(inplace=True)
    team_game_line_score_complete_df.reset_index(inplace=True)

    game_ids = game_header_w_standings_complete_df.GAME_ID

    # ONLY RETURN ONE DATAFRAME GAME IDS AND WRITE THE REST TO RELEVANT BUCKETS

    return game_header_w_standings_complete_df, team_game_line_score_complete_df, error_dates_df, game_ids





# GET BOXSCORE STATS ------------------------------------------------------------
## these are just additional boxscore stat metrics (e.g. offesnive rating, usage percentge) 
def get_player_n_team_boxscore_stats(game_ids):

    player_boxscore_stats_list = []
    team_boxscore_stats_list = []
    game_id_list = []
    error_game_id_list = []

    game_sample_len = len(game_ids)
    loop_place = 0

    for game_id in game_ids:
        print(f'Starting {game_id}')

        try:
            player_boxscore_stats = BoxScoreAdvancedV2(game_id=game_id).player_stats.get_data_frame()
            time.sleep(1.5)
            team_boxscore_stats = BoxScoreAdvancedV2(game_id=game_id).team_stats.get_data_frame()

            player_boxscore_stats_list.append(player_boxscore_stats)
            team_boxscore_stats_list.append(team_boxscore_stats)

            print(f'success {game_id}')
        
        except Exception as e:
            error_game_id_list.append(game_id)

            print(f'error {game_id}')
        
        loop_place += 1
        print(f'{(loop_place/game_sample_len)*100} % complete')
        time.sleep(1.5)

    player_boxscore_stats_df = pd.concat(player_boxscore_stats_list)
    player_ids = player_boxscore_stats_df.PLAYER_ID.unique()

    team_boxscore_stats_df = pd.concat(team_boxscore_stats_list)
    

    error_player_gamelog_df = pd.DataFrame(error_game_id_list)

    # AGAIN WE'LL WRITE EVERYTHING THEN JUST RETURN THE IDS IN HERE 

    return player_boxscore_stats_df, team_boxscore_stats_df, error_player_gamelog_df, player_ids


def get_player_gamelog(player_ids):

    active_playergamelog_list = []
    players_info_length = len(player_ids)
    loop_place = 0

    error_player_gamelog_list = []

    for id in player_ids:

        print(str(id) + ' starting')
        
        for i in range(1990,2023): # we want to account for player history before if a player played before 2002
            try:
                gamelog = pd.concat(playergamelog.PlayerGameLog(player_id=id, season=i).get_data_frames())

                if gamelog.shape[0] != 0:
                    active_playergamelog_list.append(gamelog)
                    print(str(id), str(i), ' processing')

            except Exception as e:
                error_player_gamelog_list.append(id)
                print(f'error on {id} and {i}')

            time.sleep(1.01)
        
        loop_place += 1
        print(str(round((loop_place/players_info_length) *100, 2)) + '%')
    
    active_playergamelog_df = pd.concat(active_playergamelog_list)
    active_playergamelog_df["GAME_DATE"] = pd.to_datetime(active_playergamelog_df["GAME_DATE"], format="%b %d, %Y")
    active_playergamelog_df.reset_index(inplace=True)

    # WE WRITE THESE TO S3 AGAIN 
    return active_playergamelog_df, error_player_gamelog_list


game_header_w_standings_complete_df, team_game_line_score_complete_df, error_dates_df, game_ids = get_game_header_n_line_score('2023-02-20')

player_boxscore_stats_df, team_boxscore_stats_df, error_player_gamelog_df, player_ids = get_player_n_team_boxscore_stats(game_ids)

active_playergamelog_df, error_player_gamelog_list = get_player_gamelog(player_ids)

player_ids


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




#active_playergamelog_df.to_parquet('projects/nba-daily-fantasy/data/active_playergamelog_2023_01_19.parquet')

error_id_df_2023_active.to_csv('projects/nba-daily-fantasy/data/error_id_df_2023_active.csv')

# GET ALL GAME IDS FOR GIVEN DATES ------------------------------------------------





#THIS LOOKS LIKE GAME_IDS ALREADY IN TEHRE VERSUS

player_boxscore_stats = pd.read_parquet('projects/nba-daily-fantasy/data/player_boxscore_stats_1.parquet')
game_ids_already_pulled = player_boxscore_stats.GAME_ID.unique()























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