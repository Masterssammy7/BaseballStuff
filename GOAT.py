import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load Batting and Salaries data using csv module
batting_data = []
salaries_data = []

with open('Batting - Batting.csv.csv', newline='') as batting_file:
    reader = csv.DictReader(batting_file)
    for row in reader:
        batting_data.append(row)

with open('Salaries - Salaries.csv', newline='') as salaries_file:
    reader = csv.DictReader(salaries_file)
    for row in reader:
        salaries_data.append(row)

# Convert to numpy arrays for easier manipulation
batting_data = np.array(batting_data)
salaries_data = np.array(salaries_data)

# Extract relevant data (playerID, batting average, salary, lgID as league, and yearID as year)
batting_player_ids = np.array([row['playerID'] for row in batting_data])
salary_player_ids = np.array([row['playerID'] for row in salaries_data])

batting_years = np.array([row['yearID'] for row in batting_data])
salary_years = np.array([row['yearID'] for row in salaries_data])

# Ensure alignment by matching playerID and yearID
aligned_indices = []
for i, (player, year) in enumerate(zip(batting_player_ids, batting_years)):
    match_idx = np.where((salary_player_ids == player) & (salary_years == year))[0]
    if len(match_idx) > 0:
        aligned_indices.append((i, match_idx[0]))

# Filter the data based on aligned indices
batting_player_ids = np.array([batting_player_ids[i] for i, j in aligned_indices])
batting_avg = np.array([float(batting_data[i]['H']) / float(batting_data[i]['AB']) if float(batting_data[i]['AB']) > 0 else 0 for i, j in aligned_indices])
salary = np.array([float(salaries_data[j]['salary']) for i, j in aligned_indices])
league = np.array([batting_data[i]['lgID'] for i, j in aligned_indices])
year = np.array([batting_data[i]['yearID'] for i, j in aligned_indices])
team = np.array([batting_data[i]['teamID'] for i, j in aligned_indices])

# Step 1: Calculate Year-Over-Year Salary Change for Each Team
unique_teams = np.unique(team)
team_salary_changes = []
team_performance_changes = []

for t in unique_teams:
    team_indices = np.where(team == t)
    team_years = year[team_indices].astype(int)
    team_salaries = salary[team_indices]
    team_batting_avg = batting_avg[team_indices]
    
    # Sort data by year to calculate year-over-year change
    sorted_indices = np.argsort(team_years)
    team_years_sorted = team_years[sorted_indices]
    team_salaries_sorted = team_salaries[sorted_indices]
    team_batting_avg_sorted = team_batting_avg[sorted_indices]
    
    # Calculate salary change and performance change year-over-year
    salary_change = np.diff(team_salaries_sorted) / team_salaries_sorted[:-1]
    performance_change = np.diff(team_batting_avg_sorted) / team_batting_avg_sorted[:-1]
    
    team_salary_changes.append(salary_change)
    team_performance_changes.append(performance_change)

# Step 2: Fit a Best-Fit Trend Line
salary_changes_flat = np.concatenate(team_salary_changes)
performance_changes_flat = np.concatenate(team_performance_changes)

# Perform a linear regression to find the trend
slope, intercept, r_value, p_value, std_err = stats.linregress(salary_changes_flat, performance_changes_flat)

# Plot the salary change vs. performance change with the trend line
plt.figure(figsize=(10, 6))
plt.scatter(salary_changes_flat, performance_changes_flat, alpha=0.6, label='Team Data')
plt.plot(salary_changes_flat, intercept + slope * salary_changes_flat, 'r', label=f'Trend Line (slope={slope:.2f})')

plt.title('Salary Change vs. Performance Change (Next Year)')
plt.xlabel('Salary Change (%)')
plt.ylabel('Performance Change (Batting Avg)')
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Identify the Team with the Most Top Players
# Count the number of times each team had the top player by batting average in any year
top_players_by_year = {}

for y in np.unique(year):
    yearly_indices = np.where(year == y)
    best_player_idx = yearly_indices[0][np.argmax(batting_avg[yearly_indices])]
    best_team = team[best_player_idx]
    
    if best_team in top_players_by_year:
        top_players_by_year[best_team] += 1
    else:
        top_players_by_year[best_team] = 1

# Find the team with the most top players
top_team = max(top_players_by_year, key=top_players_by_year.get)

# Step 4: Plot Best Fit Trend for the Top Team's Performance Over the Years
top_team_indices = np.where(team == top_team)
top_team_years = year[top_team_indices].astype(int)
top_team_batting_avg = batting_avg[top_team_indices]
top_team_salaries = salary[top_team_indices]

# Sort data by year
sorted_indices = np.argsort(top_team_years)
top_team_years_sorted = top_team_years[sorted_indices]
top_team_batting_avg_sorted = top_team_batting_avg[sorted_indices]
top_team_salaries_sorted = top_team_salaries[sorted_indices]

# Fit a trend line for the top team's batting average over the years
slope_top_team, intercept_top_team, r_value_top_team, p_value_top_team, std_err_top_team = stats.linregress(top_team_years_sorted, top_team_batting_avg_sorted)

# Plot the top team's performance over the years
plt.figure(figsize=(10, 6))
plt.scatter(top_team_years_sorted, top_team_batting_avg_sorted, label=f'{top_team} Performance', color='blue', alpha=0.7)
plt.plot(top_team_years_sorted, intercept_top_team + slope_top_team * top_team_years_sorted, 'r', label=f'Trend Line (slope={slope_top_team:.2f})')

plt.title(f'{top_team} Performance Over the Years')
plt.xlabel('Year')
plt.ylabel('Batting Average')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Scatter Plot for Each Team's Performance by Year
plt.figure(figsize=(12, 8))

# Unique teams for color differentiation
unique_teams = np.unique(team)

# Scatter plot by team, year, and performance
for t in unique_teams:
    team_indices = np.where(team == t)
    plt.scatter(year[team_indices], batting_avg[team_indices], s=salary[team_indices] / 100000, label=t, alpha=0.6)

plt.title('Team Performance Over the Years (Batting Average vs Year)')
plt.xlabel('Year')
plt.ylabel('Batting Average')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), title='Teams')
plt.grid(True)
plt.show()
