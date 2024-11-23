import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Set up the universe (ranges) for each input and output variable
x_traffic_density = np.arange(0, 50, 1)  # Traffic Density: 0-50 vehicles/100m
x_driver_speed = np.arange(0, 150, 1)    # Driver Speed: 0-150 km/h
x_rainfall_intensity = np.arange(0, 20, 0.1)  # Rainfall Intensity: 0-20 mm/hr
x_time_of_day = np.arange(0, 1440, 1)    # Time of Day: 0-1440 minutes (24 hours)
x_accident_likelihood = np.arange(0, 101, 1)  # Accident Likelihood: 0-100%

# Define fuzzy membership functions for Traffic Density
traffic_light = fuzz.trapmf(x_traffic_density, [0, 0, 5, 10])
traffic_moderate = fuzz.trapmf(x_traffic_density, [10, 15, 20, 25])
traffic_heavy = fuzz.trapmf(x_traffic_density, [25, 30, 35, 40])
traffic_very_heavy = fuzz.trapmf(x_traffic_density, [40, 45, 50, 50])

# Define fuzzy membership functions for Driver Speed
speed_slow = fuzz.gbellmf(x_driver_speed, 10, 3, 25)
speed_moderate = fuzz.gbellmf(x_driver_speed, 10, 3, 65)
speed_fast = fuzz.gbellmf(x_driver_speed, 10, 3, 90)
speed_very_fast = fuzz.gbellmf(x_driver_speed, 10, 3, 120)

# Define fuzzy membership functions for Rainfall Intensity
rain_no = fuzz.gaussmf(x_rainfall_intensity, 0, 0.05)
rain_light = fuzz.gaussmf(x_rainfall_intensity, 1.3, 0.5)
rain_moderate = fuzz.gaussmf(x_rainfall_intensity, 5, 1)
rain_heavy = fuzz.gaussmf(x_rainfall_intensity, 10, 1.5)
rain_very_heavy = fuzz.gaussmf(x_rainfall_intensity, 17, 1.5)

# Define fuzzy membership functions for Time of Day
time_morning = fuzz.gaussmf(x_time_of_day, 510, 100)
time_afternoon = fuzz.gaussmf(x_time_of_day, 840, 100)
time_evening = fuzz.gaussmf(x_time_of_day, 1140, 100)
time_night = fuzz.gaussmf(x_time_of_day, 1320, 100)
time_overnight = fuzz.gaussmf(x_time_of_day, 150, 100)

# Define fuzzy membership functions for Accident Likelihood
accident_super_low = fuzz.trapmf(x_accident_likelihood, [0, 0, 2, 5])
accident_very_low = fuzz.trapmf(x_accident_likelihood, [5, 5, 6, 7])
accident_low = fuzz.trapmf(x_accident_likelihood, [7, 10, 13, 15])
accident_normal = fuzz.trapmf(x_accident_likelihood, [15, 25, 40, 50])
accident_high = fuzz.trapmf(x_accident_likelihood, [50, 55, 65, 70])
accident_very_high = fuzz.trapmf(x_accident_likelihood, [70, 75, 80, 87])
accident_super_high = fuzz.trapmf(x_accident_likelihood, [87, 92, 97, 100])

# Plotting all membership functions

fig, axs = plt.subplots(5, 1, figsize=(10, 20))

# Traffic Density Membership Functions
axs[0].plot(x_traffic_density, traffic_light, 'b', label='Light')
axs[0].plot(x_traffic_density, traffic_moderate, 'g', label='Moderate')
axs[0].plot(x_traffic_density, traffic_heavy, 'r', label='Heavy')
axs[0].plot(x_traffic_density, traffic_very_heavy, 'm', label='Very Heavy')
axs[0].set_title('Traffic Density')
axs[0].legend()

# Driver Speed Membership Functions
axs[1].plot(x_driver_speed, speed_slow, 'b', label='Slow')
axs[1].plot(x_driver_speed, speed_moderate, 'g', label='Moderate')
axs[1].plot(x_driver_speed, speed_fast, 'r', label='Fast')
axs[1].plot(x_driver_speed, speed_very_fast, 'm', label='Very Fast')
axs[1].set_title('Driver Speed')
axs[1].legend()

# Rainfall Intensity Membership Functions
axs[2].plot(x_rainfall_intensity, rain_no, 'b', label='No Rain')
axs[2].plot(x_rainfall_intensity, rain_light, 'g', label='Light Rain')
axs[2].plot(x_rainfall_intensity, rain_moderate, 'r', label='Moderate Rain')
axs[2].plot(x_rainfall_intensity, rain_heavy, 'm', label='Heavy Rain')
axs[2].plot(x_rainfall_intensity, rain_very_heavy, 'c', label='Very Heavy Rain')
axs[2].set_title('Rainfall Intensity')
axs[2].legend()

# Time of Day Membership Functions
axs[3].plot(x_time_of_day, time_morning, 'b', label='Morning')
axs[3].plot(x_time_of_day, time_afternoon, 'g', label='Afternoon')
axs[3].plot(x_time_of_day, time_evening, 'r', label='Evening')
axs[3].plot(x_time_of_day, time_night, 'm', label='Night')
axs[3].plot(x_time_of_day, time_overnight, 'c', label='Overnight')
axs[3].set_title('Time of Day')
axs[3].legend()

# Accident Likelihood Membership Functions
axs[4].plot(x_accident_likelihood, accident_super_low, 'b', label='Super Low')
axs[4].plot(x_accident_likelihood, accident_very_low, 'g', label='Very Low')
axs[4].plot(x_accident_likelihood, accident_low, 'r', label='Low')
axs[4].plot(x_accident_likelihood, accident_normal, 'm', label='Normal')
axs[4].plot(x_accident_likelihood, accident_high, 'c', label='High')
axs[4].plot(x_accident_likelihood, accident_very_high, 'y', label='Very High')
axs[4].plot(x_accident_likelihood, accident_super_high, 'k', label='Super High')
axs[4].set_title('Accident Likelihood')
axs[4].legend()

# Display the plots
plt.tight_layout()
plt.show()
