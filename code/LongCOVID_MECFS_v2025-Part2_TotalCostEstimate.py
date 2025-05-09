# ==================================================================================================
# The rising cost of Long COVID and ME/CFS in Germany (2020 - 2024)
# by James Daniell, Johannes Brand, Dirk Paessler, Joerg Heydecke, Simon Schoening and Amy McLennan
# ==================================================================================================
#
# Long COVID and ME/CFS cost calculation
# 
# Created: 
# 2025-01-31
# 
# Author:
# James Daniell
#
# -*- coding: utf-8 -*- 	
# ==================================================================================================


import pandas as pd
import numpy as np
import os
import time

np.random.seed(42) #meaning of life random seed :)

mecfs_recovery_rate = 0.05 #0.1
daily_recovery_rate = mecfs_recovery_rate / 365  # Daily recovery rate

# Define permutation options and select a specific permutation index
age_groups_options = ["[0.23899,0.171371,0.196573,0.393066]"] # Population distribution 2
#age_groups_options = ["[0.184640, 0.238318, 0.273365, 0.303677]"] # Population distribution 1
reinfection_options = [True]
long_covid_disability_options = [0.4] # low severity: 0.24, high severity: 0.4
mecfs_disability_options = [0.58] # low severity: 0.31, high severity: 0.58

# Define permutations and select a specific permutation index
permutations = [
    (age, reinfections, lc_dis, mecfs_dis)
    for age in age_groups_options
    for reinfections in reinfection_options
    for lc_dis in long_covid_disability_options
    for mecfs_dis in mecfs_disability_options
]
permutation_index = 0  # Change this as needed for different runs
selected_permutation = permutations[permutation_index]

# Extract specific values from the selected permutation
selected_age_group = selected_permutation[0]
selected_reinfections = selected_permutation[1]
selected_long_covid_disability = selected_permutation[2]
selected_mecfs_disability = selected_permutation[3]

# Create a formatted file name for the selected permutation, including brackets and commas
#selected_permutation_str = f"age_{selected_age_group}_reinfections_{selected_reinfections}_lc_dis_{selected_long_covid_disability}_mecfs_dis_{selected_mecfs_disability}"
column_names = ['start_date', 'age_group', 'disability_rating', 'severity_days', 'recovery_days', 
                          'severity_level', 'will_transition_to_mecfs', 'recovered']




# Define file paths
folder_path = 'F:/Corona/'
cases_file_path = f'{folder_path}input-cases-paessler.csv'
#long_covid_file_path = f'{folder_path}long_covid_case_trackerv2025V2_{selected_permutation_str}.csv'
long_covid_file_path = f'{folder_path}long_covid_cases_popavg_024_031_v15.csv'
# Verify the constructed file path
print(f"Attempting to load Long COVID case tracker from: {long_covid_file_path}")

# Check if the constructed file path exists, or raise a FileNotFoundError with a clear message
if not os.path.isfile(long_covid_file_path):
    print("Error: The constructed file path does not exist.")
    print("Ensure the filename matches exactly with the format in your folder.")
    # List files in the directory to assist in troubleshooting
    print("Available files in directory:")
    for filename in os.listdir(folder_path):
        print(filename)
    raise FileNotFoundError(f"No file found at path: {long_covid_file_path}")

# If the file exists, proceed with loading
df_cases = pd.read_csv(cases_file_path, parse_dates=['time_iso8601'])

chunk_size = 100000  # Adjust this based on memory availability
chunks = pd.read_csv(long_covid_file_path, dtype={'age_group': str}, chunksize=chunk_size)

# Concatenate chunks into a DataFrame
df_long_covid = pd.concat(chunks, ignore_index=True)

df_long_covid.columns = ['start_date', 'age_group', 'disability_rating', 'severity_days', 'recovery_days', 
                          'severity_level', 'will_transition_to_mecfs', 'recovered']

print(f"✅ Successfully loaded Long COVID data ({df_long_covid.shape[0]} rows).")


# Ensure date columns are correct
df_cases['time_iso8601'] = pd.to_datetime(df_cases['time_iso8601'])


# Define the first date in df_cases
first_date = df_cases['time_iso8601'].min()

df_long_covid['start_date'] = pd.to_datetime(df_long_covid['start_date'], errors='coerce')
print(df_long_covid['start_date'].dtype)  # Should be datetime64

# Ensure 'start_day' is numeric

df_long_covid['start_day'] = (df_long_covid['start_date'] - first_date).dt.days

# Check for invalid entries
if df_long_covid['start_day'].isnull().any():
    raise ValueError("The 'start_day' column contains invalid entries that cannot be converted to numeric.")

# Calculate 'start_date' and 'recovery_date'
# Check if 'recovery_days' exists
if 'recovery_days' not in df_long_covid.columns:
    print("'recovery_days' column is missing. Generating default values using log-normal distribution.")
    
def generate_recovery_days():


    """
    Generates severity duration (in days) based on a lognormal distribution:
    - 87% of cases are within 365 days.
    - Mode (most common duration) is ~60 days.
    - Heavy tail extends up to 1460 days.
    """
    """
    Assigns Long COVID severity (duration in days).
    
    - 77% of cases are within 365 days, using a lognormal distribution.
    - 23% of cases extend beyond 365 days, using an exponential tail up to 10 years.
    - Cases beyond 365 days recover at 15% per year.
    """
    
    short_term_prob = 0.8  # 80% probability of short-term duration

    if np.random.uniform() < short_term_prob:
        # Short-term distribution (lognormal, constrained to 28-365 days), peaking at ~80-100 days
        mean = np.log(100)  # Adjusted mean to peak near 80-100 days as per insurance info
        sigma = 0.5  # Moderate spread to sharpen the peak
        severity_days = np.random.lognormal(mean, sigma)
        severity_days = np.clip(severity_days, 28, 365)
    else:
        # Long-term cases with a gradual decay (Weibull distribution)
        shape = 1.2  # Controls decay rate (closer to linear than exponential)
        scale = 2500  # Extended scale for a smoother decline
        severity_days = 365 + np.random.weibull(shape) * scale
        severity_days = np.clip(severity_days, 366, 3650) #fitting known decline
    
    return int(severity_days)

    # Apply the function to each row
df_long_covid['recovery_days'] = [generate_recovery_days() for _ in range(len(df_long_covid))]

# Calculate recovery_date
df_long_covid['recovery_date'] = df_long_covid['start_date'] + pd.to_timedelta(df_long_covid['recovery_days'], unit='D')

# Initialize MECFS cases with transitions and initial cases
df_mecfs = df_long_covid[df_long_covid['will_transition_to_mecfs'] == True].copy()
df_mecfs['start_date'] = df_mecfs['recovery_date']
df_mecfs['end_date'] = df_mecfs['start_date'] + pd.to_timedelta(365 * 5, unit='D')  # 5-year duration

# Add the initial 400,000 MECFS cases at start date
initial_mecfs_cases = 400000  #KBV data - overestimation vs. underestimation
# Define calculation mode: "default", "gdp", or "bruttowertschöpfung"
calculation_mode = "default"  # Change as needed

 
    
# Dynamically update losses based on calculation mode
if calculation_mode == "default":
    society_loss_values = {
        '0-19': 68, '20-39': 124, '40-59': 137, '60+': 94
    }
    personal_loss_values = {
        '0-19': 111, '20-39': 178, '40-59': 184, '60+': 118
    }
    employer_loss_values = {
        '0-19': 25, '20-39': 93, '40-59': 108, '60+': 49
    }
elif calculation_mode == "gdp":
    society_loss_values = {age_group: 126.75 for age_group in ['0-19', '20-39', '40-59', '60+']}
    personal_loss_values = {age_group: 0 for age_group in ['0-19', '20-39', '40-59', '60+']}
    employer_loss_values = {age_group: 0 for age_group in ['0-19', '20-39', '40-59', '60+']}
elif calculation_mode == "bruttowertschöpfung": #this works out very closely to the willingness to pay QALY of 85000 euros per year
    bruttowertschöpfung_value = 207 / 0.8889  # Calculate the daily value
    society_loss_values = {age_group: bruttowertschöpfung_value for age_group in ['0-19', '20-39', '40-59', '60+']}
    personal_loss_values = {age_group: 0 for age_group in ['0-19', '20-39', '40-59', '60+']}
    employer_loss_values = {age_group: 0 for age_group in ['0-19', '20-39', '40-59', '60+']}
else:
    raise ValueError("Invalid calculation mode. Choose 'default', 'gdp', or 'bruttowertschöpfung'.")
#https://www.medrxiv.org/content/10.1101/2023.10.09.23296505v1.full.pdf
#https://pubmed.ncbi.nlm.nih.gov/38011828/
#https://researchers.cdu.edu.au/en/publications/the-economic-impacts-of-myalgic-encephalomyelitischronic-fatigue-

# Define age groups with dynamically updated values
age_groups = {
    '0-19': {
        'personal_loss': personal_loss_values['0-19'],
        'employer_loss': employer_loss_values['0-19'],
        'society_loss': society_loss_values['0-19']
    },
    '20-39': {
        'personal_loss': personal_loss_values['20-39'],
        'employer_loss': employer_loss_values['20-39'],
        'society_loss': society_loss_values['20-39']
    },
    '40-59': {
        'personal_loss': personal_loss_values['40-59'],
        'employer_loss': employer_loss_values['40-59'],
        'society_loss': society_loss_values['40-59']
    },
    '60+': {
        'personal_loss': personal_loss_values['60+'],
        'employer_loss': employer_loss_values['60+'],
        'society_loss': society_loss_values['60+']
    }
}
# Calculate daily cost based on age group and disability rating
# Function to calculate the daily cost
def calculate_cost_per_day(age_group_costs, disability_rating):
    return (
        (age_group_costs['personal_loss'] + age_group_costs['employer_loss'] + age_group_costs['society_loss'])
        * disability_rating
    )

# Calculate daily cost for Long COVID and MECFS cases
df_long_covid['daily_cost'] = df_long_covid.apply(
    lambda row: calculate_cost_per_day(age_groups[row['age_group']], row['disability_rating']), axis=1
)

df_long_covid['daily_cost'] = df_long_covid['daily_cost'].astype(float)

# Prepare MECFS cases (initial and transitioned from Long COVID)
df_long_covid['start_date'] = first_date + pd.to_timedelta(df_long_covid['start_day'], unit='D')
df_long_covid['recovery_date'] = df_long_covid['start_date'] + pd.to_timedelta(df_long_covid['recovery_days'], unit='D')
df_mecfs = df_long_covid[df_long_covid['will_transition_to_mecfs']].copy()
df_mecfs['start_date'] = df_mecfs['recovery_date']
df_mecfs['end_date'] = df_mecfs['start_date'] + pd.to_timedelta(365 * 5, unit='D')


# Assign daily costs to initial MECFS 
initial_mecfs_tracker = []
for _ in range(initial_mecfs_cases):
    age_group = np.random.choice(list(age_groups.keys()))
    disability_rating = np.random.normal(selected_mecfs_disability, 0.15)  # MECFS disability rating based on permutation
    daily_cost = calculate_cost_per_day(age_groups[age_group], disability_rating)
    initial_mecfs_tracker.append({
        'age_group': age_group,
        'disability_rating': disability_rating,
        'daily_cost': daily_cost,
        'start_date': first_date,
        'end_date': first_date + pd.Timedelta(days=365 * 5)
    })

# Convert initial MECFS tracker to DataFrame and combine with df_mecfs
df_initial_mecfs = pd.DataFrame(initial_mecfs_tracker)
df_mecfs = pd.concat([df_mecfs, df_initial_mecfs], ignore_index=True)

# Calculate daily costs if missing
if 'daily_cost' not in df_mecfs.columns:
    df_mecfs['daily_cost'] = df_mecfs.apply(
        lambda row: calculate_cost_per_day(age_groups[row['age_group']], row['disability_rating']), axis=1
    )

# # Convert initial MECFS tracker to DataFrame and combine with df_mecfs
# df_initial_mecfs = pd.DataFrame(initial_mecfs_tracker)
# df_mecfs = pd.concat([df_mecfs, df_initial_mecfs], ignore_index=True)

# # Calculate daily costs if missing
# if 'daily_cost' not in df_mecfs.columns:
#     df_mecfs['daily_cost'] = df_mecfs.apply(
#         lambda row: calculate_cost_per_day(age_groups[row['age_group']], row['disability_rating']), axis=1
#     )

# Initialize cumulative cost and active case counters
cumulative_long_covid_cost = 0
cumulative_mecfs_cost = 0
active_long_covid_count = 0
active_mecfs_count = len(initial_mecfs_tracker)  # Start with initial 400,000 MECFS cases

# Lists to store daily results
new_daily_long_covid_cases = []
long_covid_daily_costs = []
cumulative_long_covid_costs = []
active_long_covid_cases = []
recovered_long_covid_cases = []

new_daily_mecfs_cases = []
mecfs_daily_costs = []
cumulative_mecfs_costs = []
active_mecfs_cases = []
recovered_mecfs_cases = []

# Indices for iterating through cases
current_long_covid_index = 0
current_mecfs_index = 0


if 'start_date' not in df_mecfs.columns:
    df_mecfs = pd.DataFrame(columns=['age_group', 'disability_rating', 'daily_cost', 'start_date', 'end_date', 'from_long_covid'])

# Track active MECFS cases
active_mecfs_count = len(initial_mecfs_tracker)  # Start with initial 400,000 MECFS cases

for index, row in df_cases.iterrows():
    current_date = row['time_iso8601']
    start_time = time.time()
    # **Long COVID Cases**
    new_long_covid_cases_today = 0
    recovered_long_covid_cases_today = 0
    total_cost_long_covid_for_day = 0
    total_mecfs_daily_cost=0

    # Add new Long COVID cases for the current day
    while current_long_covid_index < len(df_long_covid):
        case = df_long_covid.iloc[current_long_covid_index]

        if case['start_date'] <= current_date:
            active_long_covid_count += 1  # Increase active case count
            new_long_covid_cases_today += 1
            current_long_covid_index += 1
        else:
            break  # Exit early if no more cases match

    # Process active Long COVID cases
    total_cost_long_covid_for_day = df_long_covid[
        (df_long_covid['start_date'] <= current_date) & 
        (df_long_covid['recovery_date'] > current_date)
    ]['daily_cost'].sum()

    # Count recovered Long COVID cases where recovery_date == current_date
    recovered_long_covid_cases_today = df_long_covid[df_long_covid['recovery_date'] == current_date].shape[0]
    
    # Update active cases by subtracting recoveries
    active_long_covid_count = max(0, active_long_covid_count - recovered_long_covid_cases_today)
    
    # Accumulate cost
    cumulative_long_covid_cost += total_cost_long_covid_for_day
    
    print(f"Date: {current_date}, New Long COVID Cases: {new_long_covid_cases_today}, Recovered Long COVID Cases: {recovered_long_covid_cases_today}, Active Long COVID Cases: {active_long_covid_count}")
        
    print(f"Long Covid Output saved in {time.time() - start_time:.2f} seconds.")
    
    start_time = time.time()
    # **MECFS Cases**
    # **Step 1: Add New MECFS Cases**
    new_mecfs_cases_today = df_long_covid[
        (df_long_covid['will_transition_to_mecfs']) & 
        (df_long_covid['recovery_date'] == current_date)
    ].shape[0]  # Count new cases

    active_mecfs_count += new_mecfs_cases_today  # Update active MECFS count

    # **Step 2: Compute MECFS Recoveries**
    active_mecfs_today = max(0, active_mecfs_count - initial_mecfs_cases)  # Excluding initial 400,000
    expected_recoveries = active_mecfs_today * 0.000137

    integer_part = int(expected_recoveries)
    fractional_part = expected_recoveries - integer_part
    recoveries_today = integer_part + (1 if np.random.uniform() < fractional_part else 0)
    recoveries_today = min(recoveries_today, active_mecfs_count)  # Ensure recoveries do not exceed active cases

    active_mecfs_count -= recoveries_today  # Subtract recoveries

  
    # **Step 3: Compute MECFS Daily Cost Correctly**
    # Get costs for newly transitioned MECFS cases
    new_mecfs_case_costs = df_long_covid[
        (df_long_covid['will_transition_to_mecfs']) & 
        (df_long_covid['recovery_date'] == current_date)
    ]['daily_cost'].sum()
    
    # Compute total cost from existing active cases
    active_mecfs_case_costs = df_mecfs[
        (df_mecfs['start_date'] <= current_date) & 
        (df_mecfs['end_date'] > current_date)
    ]['daily_cost'].sum()
    
    # Compute recovery cost deduction
    recovered_mecfs_case_costs = df_mecfs[
        (df_mecfs['end_date'] == current_date)
    ]['daily_cost'].sum()
    
    # Total daily cost: Active MECFS cases + New MECFS cases - Recovered MECFS cases
    total_mecfs_daily_cost = active_mecfs_case_costs + new_mecfs_case_costs - recovered_mecfs_case_costs
    
    # Ensure cost does not go negative
    total_mecfs_daily_cost = max(0, total_mecfs_daily_cost)
    
    # Accumulate into cumulative cost
    cumulative_mecfs_cost += total_mecfs_daily_cost
    
    # **Store Results**
    new_daily_long_covid_cases.append(new_long_covid_cases_today)
    long_covid_daily_costs.append(total_cost_long_covid_for_day)
    cumulative_long_covid_costs.append(cumulative_long_covid_cost)
    active_long_covid_cases.append(active_long_covid_count)
    recovered_long_covid_cases.append(recovered_long_covid_cases_today)

    new_daily_mecfs_cases.append(new_mecfs_cases_today)
    mecfs_daily_costs.append(total_mecfs_daily_cost)
    cumulative_mecfs_costs.append(cumulative_mecfs_cost)
    active_mecfs_cases.append(active_mecfs_count)
    recovered_mecfs_cases.append(recoveries_today)
    print(f"Append Output saved in {time.time() - start_time:.2f} seconds.")
    print(f"Date: {current_date}, New MECFS Cases: {new_mecfs_cases_today}, MECFS Recoveries: {recoveries_today}, Active MECFS: {active_mecfs_count}")

    print(f"✅ Daily MECFS processing completed in {time.time() - start_time:.2f} seconds.")
    
# **Update df_cases with computed results**
df_cases['New Daily Long COVID Cases'] = new_daily_long_covid_cases
df_cases['Long COVID Daily Cost'] = long_covid_daily_costs
df_cases['Cumulative Long COVID Cost'] = cumulative_long_covid_costs
df_cases['Active Long COVID Cases'] = active_long_covid_cases
df_cases['Recovered Long COVID Cases'] = recovered_long_covid_cases

df_cases['New Daily MECFS Cases'] = new_daily_mecfs_cases
df_cases['MECFS Daily Cost'] = mecfs_daily_costs
df_cases['Cumulative MECFS Cost'] = cumulative_mecfs_costs
df_cases['Active MECFS Cases'] = active_mecfs_cases
df_cases['Recovered MECFS Cases'] = recovered_mecfs_cases

# **Save results**
output_file = 'F:/Corona/long_covid_and_mecfs_daily_costsv2025v21_popold_l040_m058_v1d8.csv'
df_cases.to_csv(output_file, index=False)

print(f"Processing completed. Output saved to {output_file}")
