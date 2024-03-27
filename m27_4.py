import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Kalman Filter functions
def predict(x, P, F, Q):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, H, R):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated, y, K, S

# Joint Probabilistic Data Association (JPDA)
def measurement_log_likelihood(z, x_pred, P_pred, H, R):
    innovation = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    log_det_S = np.log(np.linalg.det(S))
    log_likelihood = -0.5 * (len(z) * np.log(2 * np.pi) + log_det_S + np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation)))
    return log_likelihood

def association_probabilities(z, x_pred, P_pred, H, R, measurements):
    log_likelihoods = []
    for measurement in measurements:
        log_likelihood = measurement_log_likelihood(measurement, x_pred, P_pred, H, R)
        log_likelihoods.append(log_likelihood)
    max_log_likelihood = max(log_likelihoods)
    exp_log_likelihoods = np.exp(log_likelihoods - max_log_likelihood) # Subtract max for numerical stability
    association_probs = exp_log_likelihoods / np.sum(exp_log_likelihoods)
    return association_probs

# Constants
dt = 1.0  # Time step
F = np.array([[1, dt, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, dt, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, dt, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # State transition matrix
H = np.eye(9)  # Measurement matrix
Q = np.eye(9) * 0.01  # Process noise covariance
R = np.eye(9) * 0.1  # Measurement noise covariance

# Initial state for target 1
x_1 = np.array([0, 10, 0, 5, 0, 10, 0, 5, 0])  # [position_x, velocity_x, position_y, velocity_y, position_z, velocity_z]
P_1 = np.eye(9)  # Initial state covariance for target 1

# Measurements for target 1
measurements_1 = [
    np.array([94779.54, 217.0574, 2.7189, 21486.916, 0, 0, 0, 0, 0]),
    np.array([27197.81, 153.2595, 1.2913, 21487.193, 0, 0, 0, 0, 0]),
    np.array([85839.11, 226.6049, 5.0573, 21487.252, 0, 0, 0, 0, 0])
]

# True state for target 1
true_states_1 = [
    np.array([94780, 215, 3, 21487, 0, 0, 0, 0, 0]),
    np.array([27198, 155, 1, 21488, 0, 0, 0, 0, 0]),
    np.array([85840, 225, 6, 21488, 0, 0, 0, 0, 0, 0])
]

# Kalman Filter for target 1
x_trajectory_1 = []  # List to store x positions
y_trajectory_1 = []  # List to store y positions
z_trajectory_1 = []  # List to store z positions
x_estimated_1 = []  # List to store estimated x positions
y_estimated_1 = []  # List to store estimated y positions
z_estimated_1 = []  # List to store estimated z positions
for z_index, z in enumerate(measurements_1):
    # Predict step
    x_pred, P_pred = predict(x_1, P_1, F, Q)

    # Measurement update step
    x_updated, P_updated, innovation, K, S = update(x_pred, P_pred, z, H, R)

    # Store position for trajectory plotting
    x_trajectory_1.append(x_updated[0])  # x position
    y_trajectory_1.append(x_updated[2])  # y position
    z_trajectory_1.append(x_updated[4])  # z position
    x_estimated_1.append(x_pred[0])  # x position
    y_estimated_1.append(x_pred[2])  # y position
    z_estimated_1.append(x_pred[4])  # z position

    # Print results
    print("\nMeasurement:", z)
    print("Predicted State:", x_pred)
    print("Predicted Covariance:")
    print(P_pred)
    print("Innovation:", innovation)
    print("Kalman Gain:")
    print(K)
    print("Updated State:", x_updated)
    print("Updated Covariance:")
    print(P_updated)

    # Update state and covariance for next iteration
    x_1 = x_updated
    P_1 = P_updated

# Plot trajectory and states for target 1 in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_trajectory_1, y_trajectory_1, z_trajectory_1, marker='o', linestyle='-', color='blue', label='Trajectory (Target 1)')
ax.plot([state[0] for state in true_states_1], [state[2] for state in true_states_1], [state[4] for state in true_states_1], marker='x', linestyle='--', color='green', label='True State (Target 1)')
ax.plot(x_estimated_1, y_estimated_1, z_estimated_1, marker='s', linestyle=':', color='red', label='Estimated State (Target 1)')
ax.set_xlabel('Measurement Range')
ax.set_ylabel('Measurement Azimuth')
ax.set_zlabel('Measurement Elevation')
ax.set_title('Target 1 Trajectory and States')
ax.legend()
plt.show()
