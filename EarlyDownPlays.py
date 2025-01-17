# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:56:24 2024

@author: NAWri
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import itertools

seasons = [2021,2022,2023]
pbpdata = nfl.import_pbp_data(seasons)

# Filter for neutral game situations and downs (1st, 2nd, and 3rd)
pbp_filtered = pbpdata[
    (pbpdata['down'].isin([1, 2, 3])) & 
    (pbpdata['score_differential'].abs() <= 16)
]

# Sort by game_id, series, and play_id for proper ordering
pbp_filtered = pbp_filtered.sort_values(by=['game_id', 'series', 'play_id']).copy()

# Filter to include only series with all three downs
def has_all_downs(group):
    # Check if the set of downs in the group includes 1st, 2nd, and 3rd down
    return set([1, 2, 3]).issubset(group['down'])

# Apply the filter by grouping by game_id and series, keeping only groups with all three downs
series_with_all_downs = pbp_filtered.groupby(['game_id', 'series']).filter(has_all_downs)

# Check the resulting DataFrame
print(series_with_all_downs[['game_id', 'series', 'down', 'play_type']].head(20))

#---------------------------------------------------------------------------------------

# Identify the play type of the first down in each series
# Group by game_id and series, and get the first play type in each group
pbp_filtered['first_down_play_type'] = pbp_filtered.groupby(['game_id', 'series'])['play_type'].transform('first')

# Filter into two dataframes based on the first down play type
# All series where the first down was a run
run_start_series = pbp_filtered[pbp_filtered['first_down_play_type'] == 'run']

# All series where the first down was a pass
pass_start_series = pbp_filtered[pbp_filtered['first_down_play_type'] == 'pass']

# Check the first few rows of each to verify
print("Run-start series:")
print(run_start_series[['game_id', 'series', 'down', 'play_type']].head(20))

print("\nPass-start series:")
print(pass_start_series[['game_id', 'series', 'down', 'play_type']].head(20))

#-----------------------------------------------------------------------------------
#--------------------------- 2nd Down distance to go averages -------------------------------
#-----------------------------------------------------------------------------------

# Filter for second downs in the run and pass start dataframes
second_down_run = run_start_series[run_start_series['down'] == 2]
second_down_pass = pass_start_series[pass_start_series['down'] == 2]

# Calculate the average distance to go on second down for each dataframe
avg_distance_to_go_run = second_down_run['ydstogo'].mean()
avg_distance_to_go_pass = second_down_pass['ydstogo'].mean()

print(f"Average Distance to Go on Second Down - Run Series Start: {avg_distance_to_go_run:.2f} yards")
print(f"Average Distance to Go on Second Down - Pass Series Start: {avg_distance_to_go_pass:.2f} yards")

import matplotlib.pyplot as plt

# Data for plotting
labels = ['Run Start', 'Pass Start']
averages = [avg_distance_to_go_run, avg_distance_to_go_pass]

# Create a bar graph
plt.figure(figsize=(8, 5))
plt.bar(labels, averages, color=['blue', 'orange'])
plt.ylabel('Average Distance to Go (yards)')
plt.title('Average Distance to Go on Second Down')
plt.ylim(0, max(averages) + 5)  # Adjust y-axis limit for better visualization
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--')

# Display the value on top of the bars
for i, v in enumerate(averages):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center')

# Show the plot
plt.show()

#---------------------------------------------------------------------------------------
#------------------------ Splitting by Second Down Play Type ---------------------------
#---------------------------------------------------------------------------------------
# Identify the play type of the second down in each series
# Removing series influenced by penalty

pass_filtered = pass_start_series[pass_start_series['play_type'] != 'no_play']
run_filtered = run_start_series[run_start_series['play_type'] != 'no_play']

pass_filtered = pass_filtered.groupby(['game_id', 'series']).filter(lambda x: len(x) == 3)

# Identify the second down play type for the pass filtered DataFrame
pass_filtered['second_down_play_type'] = pass_filtered.groupby(['game_id', 'series'])['play_type'].shift(-1)

# Create an empty list to store categorized DataFrames
pass_combination_dfs = []

# Categorize series into appropriate DataFrames based on second down play type
for _, group in pass_filtered.groupby(['game_id', 'series']):
    second_down_play = group[group['down'] == 2]['play_type'].values[0]  # Get the play type on second down
    if second_down_play == 'run':
        pass_combination_dfs.append(('pass_run', group))
    elif second_down_play == 'pass':
        pass_combination_dfs.append(('pass_pass', group))

# Combine categorized DataFrames into separate DataFrames
pass_run_df = pd.concat([df for combo, df in pass_combination_dfs if combo == 'pass_run'], ignore_index=True)
pass_pass_df = pd.concat([df for combo, df in pass_combination_dfs if combo == 'pass_pass'], ignore_index=True)

# Print the counts for the pass-start DataFrames
print(f"Pass-Run Series Count: {len(pass_run_df)}")
print(f"Pass-Pass Series Count: {len(pass_pass_df)}")

# Repeat similar steps for the run-start DataFrame
run_filtered = run_start_series[run_start_series['play_type'] != 'no_play']
run_filtered = run_filtered.groupby(['game_id', 'series']).filter(lambda x: len(x) == 3)

# Create an empty list to store categorized DataFrames
run_combination_dfs = []

# Categorize series into appropriate DataFrames based on second down play type
for _, group in run_filtered.groupby(['game_id', 'series']):
    second_down_play = group[group['down'] == 2]['play_type'].values[0]  # Get the play type on second down
    if second_down_play == 'run':
        run_combination_dfs.append(('run_run', group))
    elif second_down_play == 'pass':
        run_combination_dfs.append(('run_pass', group))

# Combine categorized DataFrames into separate DataFrames
run_run_df = pd.concat([df for combo, df in run_combination_dfs if combo == 'run_run'], ignore_index=True)
run_pass_df = pd.concat([df for combo, df in run_combination_dfs if combo == 'run_pass'], ignore_index=True)

# Print the counts for the run-start DataFrames
print(f"Run-Run Series Count: {len(run_run_df)}")
print(f"Run-Pass Series Count: {len(run_pass_df)}")

#-------------------------------------------------------------------------------------------------------------
#------------------------------------ Distance to go by playcalling combo ------------------------------------
#-------------------------------------------------------------------------------------------------------------

#passpass
thirddownpasspass = pass_pass_df[pass_pass_df['down'] == 3]
passpassavgdist = thirddownpasspass['ydstogo'].mean()

#passrun
thirddownpassrun = pass_run_df[pass_run_df['down'] == 3]
passrunavgdist = thirddownpassrun['ydstogo'].mean()

#runrun
thirddownrunrun = run_run_df[run_run_df['down'] == 3]
runrunavgdist = thirddownrunrun['ydstogo'].mean()

#runpass
thirddownrunpass = run_pass_df[run_pass_df['down'] == 3]
runpassavgdist = thirddownrunpass['ydstogo'].mean()

#-------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Data for the bar graph
play_combinations = ['Pass-Pass', 'Pass-Run', 'Run-Run', 'Run-Pass']
average_distances = [passpassavgdist, passrunavgdist, runrunavgdist, runpassavgdist]

# Plotting
plt.figure(figsize=(8, 6))
bars = plt.bar(play_combinations, average_distances, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('Play Call Combination')
plt.ylabel('Average Distance to Go on 3rd Down (Yards)')
plt.title('Average Distance to Go on 3rd Down for Different Play Call Combinations')
plt.ylim(0, max(average_distances) + 2)  # Adjust y-axis for visibility

# Add text labels within each bar
for bar, avg_dist in zip(bars, average_distances):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5, 
             f'{avg_dist:.2f}', ha='center', va='top', color='white', fontsize=10)

plt.show()

#---------------------------------------------------------------------------------------------------------------
#--------------------------------- Third Down Success Rates by Playcall Combo ----------------------------------
#---------------------------------------------------------------------------------------------------------------

# Calculate third down conversion success rates
passpass_conversion_rate = thirddownpasspass['success'].mean() * 100
passrun_conversion_rate = thirddownpassrun['success'].mean() * 100
runrun_conversion_rate = thirddownrunrun['success'].mean() * 100
runpass_conversion_rate = thirddownrunpass['success'].mean() * 100

# Display results
print(f"Pass-Pass Conversion Rate: {passpass_conversion_rate:.2f}%")
print(f"Pass-Run Conversion Rate: {passrun_conversion_rate:.2f}%")
print(f"Run-Run Conversion Rate: {runrun_conversion_rate:.2f}%")
print(f"Run-Pass Conversion Rate: {runpass_conversion_rate:.2f}%")

import matplotlib.pyplot as plt

# Data for the bar graph
play_combinations = ['Pass-Pass', 'Pass-Run', 'Run-Run', 'Run-Pass']
conversion_rates = [passpass_conversion_rate, passrun_conversion_rate, runrun_conversion_rate, runpass_conversion_rate]

# Plotting
plt.figure(figsize=(8, 6))
bars = plt.bar(play_combinations, conversion_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('Play Call Combination')
plt.ylabel('3rd Down Conversion Success Rate (%)')
plt.title('3rd Down Conversion Success Rates for Different Play Call Combinations')
plt.ylim(0, max(conversion_rates) + 5)  # Adjust y-axis for visibility

# Add text labels within each bar
for bar, rate in zip(bars, conversion_rates):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1, 
             f'{rate:.2f}%', ha='center', va='top', color='white', fontsize=10)

plt.show()

#-----------------------------------------------------------------------------------------------------
#------------------------------- Team Playcall Combo Tendencies --------------------------------------
#-----------------------------------------------------------------------------------------------------

# Add a play call combination identifier to each dataframe
pass_pass_df['play_combo'] = 'Pass-Pass'
pass_run_df['play_combo'] = 'Pass-Run'
run_run_df['play_combo'] = 'Run-Run'
run_pass_df['play_combo'] = 'Run-Pass'

# Concatenate all dataframes into a single dataframe
playcall_combos = pd.concat([pass_pass_df, pass_run_df, run_run_df, run_pass_df], ignore_index=True)

# Preview the concatenated dataframe to ensure it worked
print(playcall_combos[['posteam', 'play_combo', 'down', 'ydstogo']].head())

# Calculate the frequency of each play call combination per team
team_tendencies = (
    playcall_combos.groupby(['posteam', 'play_combo'])
    .size()
    .reset_index(name='count')
)

# Calculate total play count for each team to get percentages
team_totals = team_tendencies.groupby('posteam')['count'].transform('sum')
team_tendencies['percentage'] = team_tendencies['count'] / team_totals * 100

# Pivot the data to have each play call combination as a separate column for easier plotting
team_tendencies_pivot = team_tendencies.pivot(index='posteam', columns='play_combo', values='percentage').fillna(0)

# Preview the pivoted dataframe to verify
print(team_tendencies_pivot.head())

team_tendencies_pivot.to_csv("C:/Users/NAWri/Documents/BGA/EarlyDownPlaycalling/team_tendencies_pivot.csv")
"""
import pandas as pd
import matplotlib.pyplot as plt

team_tendencies_pivot.pivot(index='posteam', columns=['Pass-Pass','Pass-Run','Run-Pass','Run-Run'], values='CallPercentage').plot(kind='bar', stacked=True)

plt.xlabel('Team')
plt.ylabel('Playcall Values')
plt.title('Playcall Tendencies by Team')
plt.legend(title='Playcall Combo')
plt.show()
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load the DataFrame from the specified path
team_tendencies = pd.read_csv("C:/Users/NAWri/Documents/BGA/EarlyDownPlaycalling/team_tendencies_pivot.csv")

# Define play call columns in the format from your dataset
play_call_cols = ['Pass-Pass', 'Pass-Run', 'Run-Pass', 'Run-Run']

# Normalize each play-call combination to percentages
team_tendencies[play_call_cols] = team_tendencies[play_call_cols].div(team_tendencies[play_call_cols].sum(axis=1), axis=0) * 100

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each play-call combination as a stacked bar with explicit labels for legend
bottom = None
for play in play_call_cols:
    ax.bar(team_tendencies['posteam'], team_tendencies[play], label=play, bottom=bottom)
    bottom = team_tendencies[play] if bottom is None else bottom + team_tendencies[play]

# Adding percentage labels on each segment of the bar
for i, team in enumerate(team_tendencies['posteam']):
    cumulative = 0
    for play in play_call_cols:
        percentage = team_tendencies.loc[i, play]
        if percentage > 0:  # Only label if the segment is non-zero
            ax.text(i, cumulative + percentage / 2, f'{percentage:.1f}%', ha='center', va='center', color='white', fontsize=6)
        cumulative += percentage

# Final plot touches
ax.set_title('Play Call Tendencies by Team (in %)')
ax.set_xlabel('Team')
ax.set_ylabel('Percentage of Play Calls')
plt.xticks(rotation=90)
ax.legend(title='Play Call Type')
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------
#-------------- Win Percentage and Conversion Rate Correlations to Play Calling --------------------
#---------------------------------------------------------------------------------------------------
import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

seasons = [2021,2022,2023]
pbpdata = nfl.import_pbp_data(seasons)

# Filter for third down plays
third_downs = pbpdata[pbpdata['down'] == 3]

conversion_counts = third_downs.groupby('posteam').agg(
    conversions=('third_down_converted', 'sum'),  # Sum of successful conversions
    attempts=('third_down_converted', 'count')  # Count of third down attempts
).reset_index()
conversion_counts['conversion_rate'] = conversion_counts['conversions'] / conversion_counts['attempts']

# Calculate the conversion rate as a whole number percentage
conversion_counts['conversion_rate'] = (conversion_counts['conversions'] / conversion_counts['attempts']) * 100  # Convert to percentage

# Create the bar graph for conversion rates
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x='posteam', y='conversion_rate', data=conversion_counts, palette='viridis', order=conversion_counts['posteam'])

# Get the positions of the bars
bar_positions = [p.get_x() + p.get_width() / 2 for p in bar_plot.patches]

# Add percentage labels on top of the bars
for index, bar in enumerate(bar_plot.patches):
    y_value = conversion_counts['conversion_rate'].iloc[index]  # Get the conversion rate
    # Set text slightly above the bar
    bar_plot.text(bar_positions[index], y_value + 1, f'{y_value:.1f}%', 
                  color='black', ha='center', va='bottom', fontsize=8)

plt.title('NFL Third Down Conversion Rates (2021-2023)', fontsize=16)
plt.ylabel('Conversion Rate (%)', fontsize=14)
plt.xlabel('NFL Teams', fontsize=14)
plt.ylim(0, 100)  # Set the limit for the y-axis to go from 0 to 100%
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y')  # Add gridlines for better readability
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()
#-------------------------------------------------------------------------------------------
# Merge DataFrames on the 'posteam' column
merged_data = pd.merge(conversion_counts, team_tendencies, on='posteam')
# Calculate the correlation of playcalling tendencies with the conversion rate
correlations = merged_data[['conversion_rate', 'Pass-Pass', 'Pass-Run', 'Run-Run', 'Run-Pass']].corr()

# Extract the correlation of conversion rate with playcalling tendencies
conversion_corr = correlations['conversion_rate'].drop('conversion_rate')

# Optionally, visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix between Playcalling Tendencies and Conversion Rates')
plt.show()
#-------------------------------------------------------------------------------------------
# Load game data to calculate win percentages
games = nfl.import_games(seasons)

# Calculate win counts and total games
win_counts = games.groupby('winning_team').size().reset_index(name='wins')
games['home_team'] = games['home_team'].fillna('')
games['visitor_team'] = games['visitor_team'].fillna('')
total_games = games.groupby('home_team').size().reset_index(name='home_games')
total_games = total_games.merge(games.groupby('visitor_team').size().reset_index(name='away_games'), 
                                 left_on='home_team', right_on='visitor_team', how='outer')
total_games['total_games'] = total_games['home_games'] + total_games['away_games']
total_games.fillna(0, inplace=True)

# Merge win counts with total games
win_counts = win_counts.merge(total_games[['home_team', 'total_games']], 
                              left_on='winning_team', right_on='home_team', how='right')
win_counts['win_percentage'] = win_counts['wins'] / win_counts['total_games']
win_counts = win_counts[['home_team', 'win_percentage']].rename(columns={'home_team': 'team'})

# Combine conversion rates and win percentages
results = conversion_counts.merge(win_counts, left_on='posteam', right_on='team', how='left')

# Rename columns for clarity
results = results[['posteam', 'conversion_rate', 'win_percentage']]
results.rename(columns={'posteam': 'team'}, inplace=True)

print(results)
