import pandas as pd
import numpy as np
import datetime
import awswrangler as wr
import time as time
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players



# GET PLAYER INFO ------------------------------------------

def get_players():
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

    error_player_info_df = pd.DataFrame(error_player_info_list, columns=['player_id_error'])

    common_player_info_complete_df = pd.concat(common_player_info_complete_list)

    return common_player_info_complete_df




def filter_common_player_info_df(df):

    df_filtered = df[['PERSON_ID', 'FROM_YEAR', 'TO_YEAR', 'DISPLAY_FIRST_LAST', 'NBA_FLAG']]
    df_filtered = df_filtered[(df_filtered['TO_YEAR'] == 2022) & (df_filtered['NBA_FLAG'] == 'Y')]

    return df_filtered






def get_last_three_days_of_data():

    active_playergamelog_list = []
    players_info_length = len(common_player_info_complete_filtered)
    loop_place = 0

    error_player_gamelog_list = []

    last_three =  datetime.date.today()  - datetime.timedelta(days=3)


    for id in common_player_info_complete_filtered['PERSON_ID']:

        print(str(id) + ' starting')
        
        try:
            gamelog = pd.concat(playergamelog.PlayerGameLog(player_id=id, date_from_nullable=last_three.strftime('%m/%d/%Y')).get_data_frames())

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

    return active_playergamelog_df


common_player_info_complete = pd.read_parquet('projects/nba-daily-fantasy/data/common_player_info_complete_df.parquet')

common_player_info_complete_filtered = filter_common_player_info_df(common_player_info_complete)
