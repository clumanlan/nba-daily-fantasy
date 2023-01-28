from sktime.transformations.series.impute import Imputer
from sktime.forecasting.fbprophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
from sktime.forecasting.compose import MultiplexForecaster
from nba_api.stats.static import teams, players
import time
from pulp import *
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster

active_playergamelog_df = pd.read_parquet('projects/nba-daily-fantasy/data/active_playergamelog_2023_01_19.parquet')
active_playergamelog_df = active_playergamelog_df.drop(['level_0', 'index'], axis=1)


# FILTER FOR DATA ONLY RELEVANT TO TODAYS PLAYERS -------------------------------
current_dk_players = pd.read_csv('projects/nba-daily-fantasy/data/DKSalaries_2023_01_28.csv')
current_dk_players = current_dk_players[['Roster Position', 'Name', 'ID', 'Salary']]
current_dk_players.rename(columns={"ID":'dk_ID'}, inplace=True)
current_dk_players['Name'] = current_dk_players['Name'].replace({'Nicolas Claxton': 'Nic Claxton'})

playergamelog_current = active_playergamelog_df_reindex[active_playergamelog_df_reindex['full_name'].isin(current_current_dk_players['Name'])]



# BUILD DAILY INDEX FROM DATA --------------------------------------------

def create_daily_index(df):

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    active_player_first_game = df.groupby(['Player_ID'])[['GAME_DATE']].min()
    active_player_first_game.columns = ['first_GAME_DATE']

    df.reset_index(inplace=True)
    df_filtered = df.drop(['Game_ID', 'VIDEO_AVAILABLE'], axis=1)

    df_filtered.set_index(['Player_ID', 'GAME_DATE'], inplace=True)

    active_player_gamelog_reindex = (df_filtered
            .unstack(level=0)
            .reindex(pd.date_range(start='2003-01-01',
                    end=datetime.now() - timedelta(1), freq='D')) # creates a daily index that fills all 365 days
    )
    

    active_player_gamelog_reindex = active_player_gamelog_reindex.stack('Player_ID', dropna=False).swaplevel(0,1).sort_index()

    ## since date range above is general, filter down for only game days when they're active  

    active_player_gamelog_reindex = active_player_gamelog_reindex.reset_index()
    active_player_gamelog_reindex.rename(columns={'level_1': 'game_date'}, inplace=True)
    active_player_gamelog_reindex = active_player_gamelog_reindex.merge(active_player_first_game, how='left', left_on='Player_ID', right_on='Player_ID')
    active_player_gamelog_reindex_filtered = active_player_gamelog_reindex[active_player_gamelog_reindex['game_date'] >= active_player_gamelog_reindex['first_GAME_DATE']]

    active_player_gamelog_reindex_filtered.drop(['first_GAME_DATE'], axis=1, inplace=True)
    active_player_gamelog_reindex_filtered['Player_ID'] = active_player_gamelog_reindex_filtered['Player_ID'].astype(str)

    return active_player_gamelog_reindex_filtered



def add_names(df):

    df['Player_ID'] = df['Player_ID'].astype(str)

    players_list = players.get_players()
    players_df = pd.DataFrame(players_list)
    players_df['id'] = players_df['id'].astype(str)
    players_df = players_df[['id', 'full_name']]


    df_wnames = df.merge(players_df, left_on='Player_ID', right_on='id', how='left')
    df_wnames.drop(['id'], axis=1, inplace=True)

    return df_wnames



def calculate_fantasy_points(df):
        return df['PTS'] + df['FG3M']*0.5 + df['REB']*1.25 + df['AST']*1.5 + df['STL']*2 + df['BLK']*2 - df['TOV']*0.5


active_playergamelog_df_reindex = create_daily_index(active_playergamelog_df)
active_playergamelog_df_reindex = add_names(active_playergamelog_df_reindex)
active_playergamelog_df_reindex['fp'] = calculate_fantasy_points(active_playergamelog_df_reindex)

active_playergamelog_df_reindex.drop(['index'], axis=1, inplace=True)




# SPLIT TO TRAIN TEST SET ---------------------------------

player_gamelog_train = active_playergamelog_df_reindex[active_playergamelog_df_reindex['game_date'] <= '2017-07-28']
player_gamelog_test = active_playergamelog_df_reindex[active_playergamelog_df_reindex['game_date'] > '2017-07-28']


train_train = player_gamelog_train[player_gamelog_train['game_date'] <= '2016-07-28']
train_valid = player_gamelog_train[player_gamelog_train['game_date'] > '2016-07-28']

train_train_x = train_train.set_index(['Player_ID', 'game_date']).drop(['fp'], axis=1)
train_valid_x = train_valid.set_index(['Player_ID', 'game_date']).drop(['fp'], axis=1)


train_train_y_all = train_train.set_index(['Player_ID', 'game_date'])[['fp']]
train_valid_y_all = train_valid.set_index(['Player_ID', 'game_date'])[['fp']]

train_train_y = train_train[~train_train['PTS'].isnull()].set_index(['Player_ID', 'game_date'])[['fp']]
train_valid_y = train_valid[~train_valid['PTS'].isnull()].set_index(['Player_ID', 'game_date'])[['fp']]





fh = ForecastingHorizon(range(0,train_valid_y_all.shape[0]), is_relative=False)




# CREATE A VALIDATION METHOD -----------------------------------------------
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series

y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=36)


fh = ForecastingHorizon(y_test.index, is_relative=False)

forecaster = NaiveForecaster(strategy="last", sp=12)

forecaster.fit(y_train)

# y_pred will contain the predictions
y_pred = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

fh = ForecastingHorizon(train_valid_y.index, is_relative=False)

forecaster = NaiveForecaster(strategy="last", sp=12)

## SPECIFY QUANTIATIVE METRIC 








# BASE MODEL -------------------

player_gamelog_train_base = playergamelog_current[['Player_ID', 'full_name', 'game_date', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']]

player_gamelog_train_base.set_index(['Player_ID', 'full_name', 'game_date'], inplace=True)








# FILTER DATA DOWN HERE ----------------

# FILTER FOR PLAYERS TODAY -----------------------




start_time = time.time()

forecaster = Prophet()

prophet_impute_missing = Imputer(method='forecaster', forecaster=forecaster) # impute missing values with prophet

active_player_prophet = prophet_impute_missing.fit_transform(player_gamelog_train_base)

print("Prophet took Impute ", (time.time() - start_time)/60, " minutes to run")

active_player_prophet.to_parquet('projects/nba-daily-fantasy/data/prophet_base_forecast_2022_01_28.parquet')



# PREDICT FORECAST HORIZION -------------------------------

forecaster.fit(active_player_prophet)  

fh_five = np.arange(1, 5)
y_pred_base = forecaster.predict(fh=fh_five)

y_pred_base
y_pred_base.reset_index(inplace=True)

y_pred_base_filtered = y_pred_base[y_pred_base['game_date'] == '2023-01-28']


y_pred_base_filtered['fp'] = calculate_fantasy_points(y_pred_base_filtered)

y_pred_base_filtered = y_pred_base_filtered[~y_pred_base_filtered['full_name'].isin(['Danilo Gallinari', 'John Wall', 'Landry Shamet'])]

# OPTIMIZER -------------------------------------------------------------
dk_players = current_dk_players[current_dk_players['Name'].isin(y_pred_base_filtered['full_name'])]

dk_salaries_dict = dict(zip(dk_players.Name, dk_players.Salary))
dk_points_dict = dict(zip(y_pred_base_filtered.full_name, y_pred_base_filtered.fp))





positions = {'PG','G', 'SG', 'SF', 'PF', 'C', 'F', 'UTIL'}


dk_assignments = dict(zip(dk_players.Name, dk_players['Roster Position'].str.split('/')))

dk_assignments_set = {(k,v) for k in dk_assignments.keys() for v in dk_assignments[k]}



prob = LpProblem("Team", LpMaximize)

assign = LpVariable.dicts("assignment", dk_assignments_set, cat="Binary")
	
#maximize fantasy points
prob += sum(assign[player, pos] * dk_points_dict[player] for (player, pos) in dk_assignments_set)

prob += sum(assign[player, pos] * dk_salaries_dict[player] for (player, pos) in dk_assignments_set) <= 50000

# prob += pulp.lpSum([player_vars[i] for i in player_ids]) == 9

# only assign each player once
for player in dk_salaries_dict.keys():
    prob += sum(assign[player, pos] for pos in positions if (player, pos) in dk_assignments_set) <=1

prob += sum(assign[player, 'PG'] for player in dk_salaries_dict.keys() if (player, 'PG') in dk_assignments_set) == 1
prob += sum(assign[player, 'SG'] for player in dk_salaries_dict.keys() if (player, 'SG') in dk_assignments_set) == 1
prob += sum(assign[player, 'SF'] for player in dk_salaries_dict.keys() if (player, 'SF') in dk_assignments_set) == 1
prob += sum(assign[player, 'PF'] for player in dk_salaries_dict.keys() if (player, 'PF') in dk_assignments_set) == 1
prob += sum(assign[player, 'C'] for player in dk_salaries_dict.keys() if (player, 'C') in dk_assignments_set) == 1

prob += sum(assign[player, 'G'] for player in dk_salaries_dict.keys() if (player, 'G') in dk_assignments_set) == 1
prob += sum(assign[player, 'F'] for player in dk_salaries_dict.keys() if (player, 'F') in dk_assignments_set) == 1
prob += sum(assign[player, 'UTIL'] for player in dk_salaries_dict.keys() if (player, 'UTIL') in dk_assignments_set) == 1


prob.solve()


print('Status:', LpStatus[prob.status])

prob_name_list = []
prob_value_list = []
for v in prob.variables():
    print(v.name, '=', v.varValue)
    prob_name_list.append(v.name)
    prob_value_list.append(v.varValue)


test = pd.DataFrame({'name': prob_name_list, 'value':prob_value_list})

test[test['value'] == 1]