#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Install the necessary package
get_ipython().system('pip install pyswarms')

# Import required libraries
import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO

# Define cost parameters for various work types
full_time_rates = np.array([30, 35, 40, 45, 50])                # Full-time hourly rates
overtime_rates = 1.5 * full_time_rates                            # Overtime rates (1.5 times the full-time rate)
part_time_rates = np.array([25, 27, 29, 31, 33])                  # Part-time hourly rates

# Combine all cost parameters into a single cost vector for the 15 decision variables
cost_vector = np.concatenate([full_time_rates, overtime_rates, part_time_rates])

def compute_total_labor_expenses(x):
    penalty_list = []  # List to store penalties for each solution
    for xi in x:
        xi = np.clip(xi, 0, None)  # Ensure all values are non-negative (no negative working hours)
        peak_hours = np.sum(xi[:5]) + np.sum(xi[5:10])  # Total hours for full-time and overtime
        non_peak_hours = np.sum(xi[10:15])  # Part-time hours

        penalty = 0
        if peak_hours < 4:
            penalty += (4 - peak_hours) * 1000  # Apply penalty if peak hours are less than 4
        if non_peak_hours < 2:
            penalty += (2 - non_peak_hours) * 1000  # Apply penalty if non-peak part-time hours are less than 2

        total_cost = np.sum(cost_vector * xi)  # Compute the total cost (labor cost without penalties)
        penalty_list.append(total_cost + penalty)  # Append the total cost with penalties
    return np.array(penalty_list)

# Define bounds for the optimization and PSO parameters
lower_bounds = np.zeros(15)  # Minimum working hours (0 hours for all types)
upper_bounds = np.ones(15) * 8  # Maximum working hours (8 hours per type)
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}  # PSO algorithm settings

# Perform the Particle Swarm Optimization (PSO)
optimizer = GlobalBestPSO(n_particles=30, dimensions=15, options=options, bounds=(lower_bounds, upper_bounds))
best_cost, best_position = optimizer.optimize(compute_total_labor_expenses, iters=100)

# Define categories and hourly rates
categories = ['Full-time (x_i)', 'Overtime (x_io)', 'Part-time (x_jp)']
rates = np.concatenate([full_time_rates, overtime_rates, part_time_rates])

# Create a DataFrame to display results
df = pd.DataFrame({
    'Category': [categories[i // 5] for i in range(15)],  # Assign category labels
    'Hourly Rate': rates,
    'Hours Worked': best_position,
    'Cost': rates * best_position  # Calculate total cost based on hours worked
})

# Calculate the total peak and non-peak hours worked
total_peak_hours = np.sum(best_position[:10])  # Sum of all peak hours (full-time + overtime)
total_non_peak_hours = np.sum(best_position[10:15])  # Sum of non-peak (part-time)

# Print the results
print(f"Total Labor Cost: {best_cost:.2f} SAR")
print(f"Peak Hours Total: {total_peak_hours:.2f} hours (Required ≥ 4)")
print(f"Non-Peak Part-Time Hours: {total_non_peak_hours:.2f} hours (Required ≥ 2)")

# Display the DataFrame with detailed results
df


# In[6]:




