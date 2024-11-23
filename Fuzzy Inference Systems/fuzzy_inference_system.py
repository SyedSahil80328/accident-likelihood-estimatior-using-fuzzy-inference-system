import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define ranges and categories for inputs and outputs
# a[0] is ranges, a[1] is categories, a[2] is increment factor
traffic_density_data = [[[0, 10], [11, 25], [26, 40], [41, 50]], ["Light", "Moderate", "Heavy", "Very Heavy"], 1]
driver_speed_data = [[[0, 50], [51, 80], [81, 100], [101, 150]], ["Slow", "Moderate", "Fast", "Very Fast"], 1]
rainfall_intensity_data = [[[0.1, 2.5], [2.6, 7.5], [7.6, 15], [15.1, 20]], ["Light Rain", "Moderate Rain", "Heavy Rain", "Very Heavy Rain"], 0.1]
time_of_day_data = [[[0, 299], [300, 719], [720, 1019], [1020, 1259], [1260, 1439]], ["Overnight", "Morning", "Afternoon", "Evening", "Night"], 1]
accident_likelihood_data = [[[0, 5], [5.1, 7], [7.1, 15], [15.1, 50], [50.1, 70], [70.1, 87], [87.1, 100]], ["Super Low", "Very Low", "Low", "Medium", "High", "Very High", "Super High"], 0.1]

# Define universes 
traffic_density_universe = np.arange(0, 51, traffic_density_data[-1])
driver_speed_universe = np.arange(0, 151, driver_speed_data[-1])
rainfall_intensity_universe = np.arange(0, 20.1, rainfall_intensity_data[-1])
time_of_day_universe = np.arange(0, 1440, time_of_day_data[-1])
accident_likelihood_universe = np.arange(0, 100.1, accident_likelihood_data[-1])

# Define fuzzy variables
traffic_density = ctrl.Antecedent(traffic_density_universe, 'traffic_density')
driver_speed = ctrl.Antecedent(driver_speed_universe, 'driver_speed')
rainfall_intensity = ctrl.Antecedent(rainfall_intensity_universe, 'rainfall_intensity')
time_of_day = ctrl.Antecedent(time_of_day_universe, 'time_of_day')
accident_likelihood = ctrl.Consequent(accident_likelihood_universe, 'accident_likelihood')

# Define membership functions for traffic density
traffic_density['light'] = fuzz.trapmf(traffic_density.universe, [0, 0, 5, 10])
traffic_density['moderate'] = fuzz.trapmf(traffic_density.universe, [10, 15, 20, 25])
traffic_density['heavy'] = fuzz.trapmf(traffic_density.universe, [25, 30, 35, 40])
traffic_density['very_heavy'] = fuzz.trapmf(traffic_density.universe, [40, 45, 50, 50])

# Define membership functions for driver speed
driver_speed['slow'] = fuzz.gbellmf(driver_speed.universe, 10, 3, 25)
driver_speed['moderate'] = fuzz.gbellmf(driver_speed.universe, 10, 3, 65)
driver_speed['fast'] = fuzz.gbellmf(driver_speed.universe, 10, 3, 90)
driver_speed['very_fast'] = fuzz.gbellmf(driver_speed.universe, 10, 3, 120)

# Define membership functions for rainfall intensity
rainfall_intensity['no'] = fuzz.gaussmf(rainfall_intensity.universe, 0, 0.05)
rainfall_intensity['light'] = fuzz.gaussmf(rainfall_intensity.universe, 1.3, 0.5)
rainfall_intensity['moderate'] = fuzz.gaussmf(rainfall_intensity.universe, 5, 1)
rainfall_intensity['heavy'] = fuzz.gaussmf(rainfall_intensity.universe, 10, 1.5)
rainfall_intensity['very_heavy'] = fuzz.gaussmf(rainfall_intensity.universe, 17, 1.5)

# Define membership functions for time of day
time_of_day['morning'] = fuzz.gaussmf(time_of_day.universe, 510, 100)
time_of_day['afternoon'] = fuzz.gaussmf(time_of_day.universe, 840, 100)
time_of_day['evening'] = fuzz.gaussmf(time_of_day.universe, 1140, 100)
time_of_day['night'] = fuzz.gaussmf(time_of_day.universe, 1320, 100)
time_of_day['overnight'] = fuzz.gaussmf(time_of_day.universe, 150, 100)

# Define membership functions for accident likelihood
accident_likelihood['super_low'] = fuzz.trapmf(accident_likelihood.universe, [0, 0, 2, 5])
accident_likelihood['very_low'] = fuzz.trapmf(accident_likelihood.universe, [5, 5, 6, 7])
accident_likelihood['low'] = fuzz.trapmf(accident_likelihood.universe, [7, 10, 13, 15])
accident_likelihood['normal'] = fuzz.trapmf(accident_likelihood.universe, [15, 25, 40, 50])
accident_likelihood['high'] = fuzz.trapmf(accident_likelihood.universe, [50, 55, 65, 70])
accident_likelihood['very_high'] = fuzz.trapmf(accident_likelihood.universe, [70, 75, 80, 87])
accident_likelihood['super_high'] = fuzz.trapmf(accident_likelihood.universe, [87, 92, 97, 100])

# Define fuzzy rules
rules = [
    ctrl.Rule(traffic_density['very_heavy'] & driver_speed['very_fast'] & rainfall_intensity['very_heavy'] & time_of_day['night'], accident_likelihood['super_high']),
    ctrl.Rule(traffic_density['heavy'] & driver_speed['fast'] & rainfall_intensity['heavy'] & time_of_day['evening'], accident_likelihood['very_high']),
    ctrl.Rule(traffic_density['moderate'] & driver_speed['fast'] & rainfall_intensity['moderate'] & time_of_day['afternoon'], accident_likelihood['high']),
    ctrl.Rule(traffic_density['light'] & driver_speed['moderate'] & rainfall_intensity['light'] & time_of_day['morning'], accident_likelihood['normal']),
    ctrl.Rule(traffic_density['moderate'] & driver_speed['slow'] & rainfall_intensity['no'] & time_of_day['morning'], accident_likelihood['low']),
    ctrl.Rule(traffic_density['light'] & driver_speed['slow'] & rainfall_intensity['no'] & time_of_day['afternoon'], accident_likelihood['very_low']),
    ctrl.Rule(traffic_density['very_heavy'] & driver_speed['moderate'] & rainfall_intensity['very_heavy'] & time_of_day['overnight'], accident_likelihood['high']),
    ctrl.Rule(traffic_density['heavy'] & driver_speed['fast'] & rainfall_intensity['light'] & time_of_day['evening'], accident_likelihood['normal']),
    ctrl.Rule(traffic_density['moderate'] & driver_speed['fast'] & rainfall_intensity['no'] & time_of_day['night'], accident_likelihood['very_high']),
    ctrl.Rule(traffic_density['light'] & driver_speed['moderate'] & rainfall_intensity['no'] & time_of_day['overnight'], accident_likelihood['super_low']),
]

# Create control system and simulation
accident_ctrl = ctrl.ControlSystem(rules)
accident_sim = ctrl.ControlSystemSimulation(accident_ctrl)

# Function to categorize input values
def categorize_input(value, universe_data, ignore_index = None):
    if ignore_index == 0 and value == 0:
        return "No Rain"

    for i in range(len(universe_data[0])):
        if universe_data[0][i][0] <= value <= universe_data[0][i][1]:
            return universe_data[1][i]
    
    print(f"Error: Given value is beyond the universe range (Allowed range: {universe_data[0][0][0]} to {universe_data[0][-1][-1]}.)")
    exit(0)

# User input handling section
traffic_density_value = int(input("Enter the Traffic Density (vehicles/100m): "))
driver_speed_value = int(input("Enter the Driver Speed (km/h): "))
rainfall_intensity_value = float(input("Enter the amount of rainfall (mm/h): "))
time_input = input("Enter time (hh:mm) (24hrs time format): ")
time_of_day_value = int(time_input.split(':')[0]) * 60 + int(time_input.split(':')[1])

# Categorize each input based on the defined fuzzy membership functions
traffic_category = categorize_input(traffic_density_value, traffic_density_data)
speed_category = categorize_input(driver_speed_value, driver_speed_data)
rain_category = categorize_input(rainfall_intensity_value, rainfall_intensity_data, 0)
time_category = categorize_input(time_of_day_value, time_of_day_data)

# Display categorized inputs
print()
print(f"Traffic Density: {traffic_category}")
print(f"Driver Speed: {speed_category}")
print(f"Rainfall Intensity: {rain_category}")
print(f"Time of Day: {time_category}")

# Input test values
accident_sim.input['traffic_density'] = traffic_density_value
accident_sim.input['driver_speed'] = driver_speed_value
accident_sim.input['rainfall_intensity'] = rainfall_intensity_value
accident_sim.input['time_of_day'] = time_of_day_value  # Time of day in minutes since midnight

# Perform fuzzy inference and defuzzification
accident_sim.compute()

# Retrieve output and determine category
likelihood_value = accident_sim.output['accident_likelihood']

# Find category based on likelihood percentage
category = categorize_input(likelihood_value, accident_likelihood_data)

print(f"There is {likelihood_value:.2f}% chance of occuring accident ({category}).")
