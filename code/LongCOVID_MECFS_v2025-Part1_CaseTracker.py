# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:33:07 2025

@author: JamesBeachPC
"""

# -*- coding: utf-8 -*- 
"""
Optimized Long COVID Case Tracker with Age Groups
"""

import pandas as pd
import numpy as np
import itertools
import os
import time

np.random.seed(42) #meaning of life random seed :)

# File paths
base_folder = 'F:/Corona/'
cases_file_path = os.path.join(base_folder, 'input-cases-paessler.csv')
vaccinations_file_path = os.path.join(base_folder, 'germany_vaccinations_timeseries_v3.csv')

# Load case data
print("Loading case data...")
start_time = time.time()
df_cases = pd.read_csv(cases_file_path)
df_cases['time_iso8601'] = pd.to_datetime(df_cases['time_iso8601'])
print(f"Case data loaded in {time.time() - start_time:.2f} seconds.")

# Load vaccination data
print("Loading vaccination data...")
start_time = time.time()
df_vaccinations = pd.read_csv(vaccinations_file_path, delimiter='\t')
df_vaccinations['date'] = pd.to_datetime(df_vaccinations['date'])
print(f"Vaccination data loaded in {time.time() - start_time:.2f} seconds.")

# Constants
initial_mecfs_cases = 400_000 #KBV data - overestimation and underestimation
total_population = 84_548_219

# Probabilities & disability rating
long_covid_prob_initial = (0.095, 0.11)  # Initial probability range
long_covid_prob_final = (0.06, 0.08)    # Final probability range
mecfs_prob_range = (0.0675,0.0825)           # MECFS transition probability (11.6%) #0.05-0.06% #0.0675,0.0825

disability_rating_means = {'long_covid': 0.24, 'mecfs': 0.31}
#disability_rating_means = {'long_covid': 0.4, 'mecfs': 0.58}
disability_std = 0.1

# MECFS recovery rate
mecfs_recovery_rate = 0.05 #0.1
daily_recovery_rate = mecfs_recovery_rate / 365  # Daily recovery rate

#From 6 months on.... 

# Age Groups and Probabilities
age_groups = ['0-19', '20-39', '40-59', '60+']
age_group_prob_sets = [[0.184640, 0.238318, 0.273365, 0.303677]]  # Probabilities must sum to 1
#age_group_prob_sets = [[0.23899,0.171371,0.196573,0.393066]]

# Function: Probability sliding over time
def get_sliding_probability(current_date):
    start_date = pd.to_datetime("2020-03-01")
    end_date = pd.to_datetime("2024-12-31")
    time_progress = min(max((current_date - start_date) / (end_date - start_date), 0), 1)
    prob_min_today = long_covid_prob_initial[0] + time_progress * (long_covid_prob_final[0] - long_covid_prob_initial[0])
    prob_max_today = long_covid_prob_initial[1] + time_progress * (long_covid_prob_final[1] - long_covid_prob_initial[1])
    return prob_min_today, prob_max_today

# Function: Assign a disability rating
def assign_disability_rating(mean, std, max_value=1.0):
    return np.clip(np.random.normal(mean, std), 0, max_value)

# # Function: Assign severity (duration of Long COVID)
# def assign_severity():
#     min_days, max_days = 7, 1460  # Min 7 days, max 4 years (1460 days)
#     p = np.random.uniform(0, 1)  # Random value between 0 and 1
#     severity_days = min_days + (max_days - min_days) * (p ** 3)  # Power 3 creates a heavy tail
#     return int(severity_days)

def assign_severity():
    """
    Assigns Long COVID severity (duration in days).
    
    - 77% of cases are within 365 days, using a lognormal distribution.
    - 23% of cases extend beyond 365 days, using an exponential tail up to 10 years.
    - Cases beyond 365 days recover at 15% per year.
    """
    short_term_prob=0.80
    
    if np.random.uniform() < short_term_prob:
        # Short-term distribution (lognormal, constrained to 28-365 days), peaking at ~80-100 days
        mean = np.log(100)  # Adjusted mean to peak near 80-100 days
        sigma = 0.5  # Moderate spread to sharpen the peak
        severity_days = np.random.lognormal(mean, sigma)
        severity_days = np.clip(severity_days, 28, 365)
    else:
        # Long-term cases with a gradual decay (Weibull distribution)
        shape = 1.2  # Controls decay rate (closer to linear than exponential)
        scale = 2500  # Extended scale for a smoother decline
        severity_days = 365 + np.random.weibull(shape) * scale
        severity_days = np.clip(severity_days, 366, 3650)
    
    return int(severity_days)

# Function: Assign age group based on probability set
def assign_age_group(age_group_probs):
    return np.random.choice(age_groups, p=age_group_probs)

# Function: Assign vaccination status on infection date
def assign_vaccination_status(infection_date):
    # Vaccination campaign start date
    vaccination_start_date = pd.to_datetime("2021-01-01")
    
    # Before the vaccination campaign starts, all vaccinations should be 0
    if infection_date < vaccination_start_date:
        return 0

    # Fetch closest vaccination data
    closest_vac_data = df_vaccinations[df_vaccinations['date'] <= infection_date]
    if closest_vac_data.empty:
        return 0
    num_vaccinations = closest_vac_data.iloc[-1]['impfungen_kumulativ'] / total_population
    return min(int(num_vaccinations), 5)  # Cap at 5 doses

# Function: Track reinfections
def track_reinfections(previous_infections, cumulative_cases):
    reinfection_prob = cumulative_cases / total_population
    return previous_infections + (1 if np.random.uniform() < reinfection_prob else 0)

def get_sliding_probability(current_date, reinfections, vaccinations, adjust_reinfections):
    start_date = pd.to_datetime("2020-03-01")
    end_date = pd.to_datetime("2024-12-31")
    time_progress = min(max((current_date - start_date) / (end_date - start_date), 0), 1)
    
    # Base probability that changes over time
    prob_min_today = long_covid_prob_initial[0] + time_progress * (long_covid_prob_final[0] - long_covid_prob_initial[0])
    prob_max_today = long_covid_prob_initial[1] + time_progress * (long_covid_prob_final[1] - long_covid_prob_initial[1])
    
    # Adjust for reinfections and vaccinations
    if adjust_reinfections:
        if reinfections >= 2 and vaccinations >= 2:
            prob_min_today *= 0.075 #0.2
            prob_max_today *= 0.075
        elif reinfections >= 1 and vaccinations >= 2:
            prob_min_today *= 0.15 #0.3
            prob_max_today *= 0.15                
        elif reinfections >= 0 and vaccinations >= 2:
            prob_min_today *= 0.8
            prob_max_today *= 0.8
        elif reinfections == 0 and vaccinations < 2:
            prob_min_today *= 1.0
            prob_max_today *= 1.0
    
    return prob_min_today, prob_max_today

# Initialize storage for Long COVID cases
# Initialize storage for Long COVID cases
# Initialize storage for Long COVID cases
# Initialize storage for Long COVID and MECFS cases
long_covid_cases = []
mecfs_cases = []
cumulative_cases = 0

# Active case counters
active_long_covid_count = 0  # Tracks total Long COVID cases
active_mecfs_count = 0  # Tracks total MECFS cases (excluding recoveries)

print("Processing case data...")
process_start_time = time.time()

long_covid_case_tracker = []  
cumulative_cases = 0  
total_long_covid = 0  
total_mecfs = 0  

for row in df_cases.itertuples(index=False):
    current_date = row.time_iso8601
    daily_cases = row.sum_cases
    cumulative_cases += daily_cases

    if daily_cases == 0:
        continue  # Skip processing if no cases

    # Precompute daily values
    daily_vaccination_status = assign_vaccination_status(current_date)
    daily_reinfection_count = track_reinfections(0, cumulative_cases)

    # Get probability adjustments
    prob_min, prob_max = get_sliding_probability(
        current_date,
        reinfections=daily_reinfection_count,
        vaccinations=daily_vaccination_status,
        adjust_reinfections=True
    )

    print(f"{current_date}: Processing {daily_cases} cases")
    print(f"Long COVID Probability Range: {prob_min:.4f} - {prob_max:.4f}")

    # Compute Long COVID cases using NumPy
    prob_today = np.random.uniform(prob_min, prob_max)
    num_long_covid = np.sum(np.random.uniform(0, 1, daily_cases) < prob_today)

    mecfs_prob_today = np.random.uniform(mecfs_prob_range[0], mecfs_prob_range[1])

    daily_long_covid_cases = [
        {
            'start_day': current_date,
            'age_group': assign_age_group(age_group_prob_sets[0]),
            'disability_rating': assign_disability_rating(disability_rating_means['long_covid'], disability_std),
            'severity_days': assign_severity(),
            'vaccinations': daily_vaccination_status,
            'reinfections': daily_reinfection_count,
            'will_transition_to_mecfs': np.random.uniform() < mecfs_prob_today,
            'recovered': False
        }
        for _ in range(num_long_covid)
    ]

    long_covid_case_tracker.extend(daily_long_covid_cases)
    active_long_covid_count += num_long_covid  # **Update active Long COVID counter**

    # MECFS Transitions (From Long COVID)
    mecfs_transitions = [case for case in daily_long_covid_cases if case['will_transition_to_mecfs']]
    num_mecfs_transitions = len(mecfs_transitions)

    print(f"{num_mecfs_transitions} new MECFS cases added today.")

    daily_mecfs_cases = [
        {
            'start_date': current_date,
            'age_group': case['age_group'],
            'disability_rating': assign_disability_rating(disability_rating_means['mecfs'], disability_std),
            'vaccinations': case['vaccinations'],
            'reinfections': case['reinfections'],
            'recovered': False,
            'from_long_covid': True  
        }
        for case in mecfs_transitions
    ]

    mecfs_cases.extend(daily_mecfs_cases)  # Accumulates cases instead of resetting
    active_mecfs_count += num_mecfs_transitions  # **Update active MECFS counter**

    # MECFS Recovery Calculation
    active_mecfs_today = max(0, active_mecfs_count)  # Only count post-400k cases
    expected_recoveries = active_mecfs_today * 0.000137

    integer_part = int(expected_recoveries)
    fractional_part = expected_recoveries - integer_part
    recoveries_today = integer_part + (1 if np.random.uniform() < fractional_part else 0)

    recoveries_today = min(recoveries_today, active_mecfs_count)

    # Remove recovered cases
    if recoveries_today > 0:
        mecfs_cases = mecfs_cases[:-recoveries_today]  
        active_mecfs_count -= recoveries_today  # **Decrease active MECFS counter**

    print(f"{recoveries_today} MECFS cases recovered today.")


    # Batch Save to File
    if len(long_covid_case_tracker) > 10_000:
        df_temp = pd.DataFrame(long_covid_case_tracker)
        df_temp.to_csv(os.path.join(base_folder, 'long_covid_cases_popavg_024_031_v15.csv'), mode='a', header=False, index=False)
        long_covid_case_tracker.clear()

    if len(mecfs_cases) > 10_000:
        df_temp = pd.DataFrame(mecfs_cases)
        df_temp.to_csv(os.path.join(base_folder, 'mecfs_cases_popavg_024_031_v15.csv'), mode='a', header=False, index=False)
        mecfs_cases.clear()

