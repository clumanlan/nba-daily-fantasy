import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import awswrangler as wr
import time as time
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, ScoreboardV2, BoxScoreAdvancedV2, BoxScoreTraditionalV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players
from datetime import date
import awswrangler as wr

# Get data from nba api endpoints: game headers, boxscore traditional (team & player level), boxscore advanced (team & player level) and player info


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

    game_header_w_standings_complete_df.reset_index(inplace=True)
    team_game_line_score_complete_df.reset_index(inplace=True)

    game_ids = game_header_w_standings_complete_df.GAME_ID.unique()
    game_ids_df = pd.DataFrame(game_ids, columns=['game_id'])

    print('Writing Game Header & Line Score to S3.....................')


    wr.s3.to_parquet(
        df=game_header_w_standings_complete_df,
        path="s3://nbadk-model/game_stats/game_header/game_header_initial.parquet"
    )

    wr.s3.to_parquet(
        df=team_game_line_score_complete_df,
        path="s3://nbadk-model/team_stats/game_line_score/game_line_score_initial.parquet"
    )

    wr.s3.to_parquet(
        df=game_ids_df,
        path="s3://nbadk-model/api_ids/game_ids/game_id_initial.parquet"
    )
    
    return game_ids

# GET BOXSCORE TRADITIONAL STATS -------------------------------------------------
def get_boxscore_traditional(game_ids):

    boxscore_trad_player_list = []
    boxscore_trad_team_list = []
    boxscore_trad_error_list = []
    game_len = game_ids.shape[0]
    loop_place = 0

    for game_id in game_ids:
        try:
            boxscore_trad = BoxScoreTraditionalV2(game_id=game_id)

            boxscore_trad_player = boxscore_trad.player_stats.get_data_frame()
            boxscore_trad_team = boxscore_trad.team_stats.get_data_frame()

            boxscore_trad_player_list.append(boxscore_trad_player)
            boxscore_trad_team_list.append(boxscore_trad_team)

            print(game_id)
        
        except Exception as e:
            boxscore_trad_error_list.append(game_id)

            print(f'error {game_id}')
        
        time.sleep(.60)
        loop_place += 1
        print(f'{(loop_place/game_len)*100} % complete')

    boxscore_traditional_player_df = pd.concat(boxscore_trad_player_list)
    boxscore_traditional_team_df = pd.concat(boxscore_trad_team_list)

    boxscore_traditional_player_df['MIN'] = boxscore_traditional_player_df['MIN'].astype(str)
    boxscore_traditional_team_df['MIN'] = boxscore_traditional_team_df['MIN'].astype(str)

    print('Writing Boxscore Traditional to S3.....................')

    wr.s3.to_parquet(
            df=boxscore_traditional_player_df,
            path="s3://nbadk-model/player_stats/boxscore_traditional/boxscore_traditional_player_initial.parquet"
        )

    wr.s3.to_parquet(
        df=boxscore_traditional_team_df,
        path="s3://nbadk-model/team_stats/boxscore_traditional/boxscore_traditional_team_initial.parquet"
        )

    return boxscore_traditional_player_df.PLAYER_ID.unique(), boxscore_trad_error_list


# GET BOXSCORE ADVANCED STATS  -----------------------------------------------------------------
## these are just additional boxscore stat metrics (e.g. offesnive rating, usage percentge) 
def get_boxscore_advanced(game_ids):

    player_boxscore_stats_list = []
    team_boxscore_stats_list = []
    error_game_id_list = []

    game_len = len(game_ids)
    loop_place = 0

    for game_id in game_ids:
        print(f'Starting {game_id}')

        try:
            boxscore_stats_adv = BoxScoreAdvancedV2(game_id=game_id)

            player_boxscore_stats = boxscore_stats_adv.player_stats.get_data_frame()
            team_boxscore_stats = boxscore_stats_adv.team_stats.get_data_frame()

            player_boxscore_stats_list.append(player_boxscore_stats)
            team_boxscore_stats_list.append(team_boxscore_stats)

            print(f'success {game_id}')
        
        except Exception as e:
            error_game_id_list.append(game_id)

            print(f'error {game_id}')
        
        loop_place += 1
        print(f'{(loop_place/game_len)*100} % complete')
        time.sleep(1)

    player_boxscore_advanced_stats_df = pd.concat(player_boxscore_stats_list)
    team_boxscore_stats_advanced_df = pd.concat(team_boxscore_stats_list)
    
    player_ids = player_boxscore_advanced_stats_df.PLAYER_ID.unique()
    player_ids_df = pd.DataFrame(player_ids, columns=['player_id'])

    print('Writing Boxscore Advanced to S3.....................')

    wr.s3.to_parquet(
        df=player_boxscore_advanced_stats_df,
        path="s3://nbadk-model/player_stats/boxscore_advanced/player_boxscore_advanced_stats_initial.parquet"
    )

    wr.s3.to_parquet(
        df=team_boxscore_stats_advanced_df,
        path="s3://nbadk-model/team_stats/boxscore_advanced/team_boxscore_advanced_stats_initial.parquet"
    )

    return error_game_id_list



# GET PLAYER INFO ------------------------------------------

def get_player_info(player_ids):
    common_player_info_complete_list = []
    error_player_info_list = []


    loop_place = 0
    players_df_length = len(player_ids)

    for id in player_ids:
        
        loop_place += 1

        try: 
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=id)
            common_player_info_df = player_info.common_player_info.get_data_frame()

            common_player_info_complete_list.append(common_player_info_df)
            
        except Exception as e:
            error_player_info_list.append(id)
            print(f'error {id}')


        print(id, '% complete: ', str(round((loop_place/players_df_length) *100, 2)) + '%')


        time.sleep(.67)


    common_player_info_complete_df = pd.concat(common_player_info_complete_list)

    time.sleep(1.1)

    print('Writing Common Player Info to S3.....................')

    wr.s3.to_parquet(
        df=common_player_info_complete_df,
        path="s3://nbadk-model/player_info/player_info_initial_2.parquet"
        )

    error_player_info_df = pd.DataFrame(error_player_info_list, columns=['player_id_error'])

    return error_player_info_df

def main():
    game_ids = get_game_header_n_line_score(start_date='2001-01-01')
    player_ids, boxscore_trad_error_list = get_boxscore_traditional(game_ids)
    boxscore_adv_error_list = get_boxscore_advanced(game_ids)
    error_player_info_df = get_player_info(player_ids)

if __name__ == '__main__':
    main()