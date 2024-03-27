import numpy as np

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
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # State transition matrix
H = np.eye(4)  # Measurement matrix
Q = np.eye(4) * 0.01  # Process noise covariance
R = np.eye(4) * 0.1  # Measurement noise covariance

# Initial state
x = np.array([0, 10, 0, 5])  # [position_x, velocity_x, position_y, velocity_y]
P = np.eye(4)  # Initial state covariance

# Measurements
measurements = [
    np.array([94779.54, 217.0574, 2.7189, 21486.916]),
    np.array([27197.81, 153.2595, 1.2913, 21487.193]),
    np.array([85839.11, 226.6049, 5.0573, 21487.252])
]

# Kalman Filter
for z in measurements:
    # Predict step
    x_pred, P_pred = predict(x, P, F, Q)

    # Measurement update step
    x_updated, P_updated, innovation, K, S = update(x_pred, P_pred, z, H, R)

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

# JPDA
for z in measurements:
    # Predict step
    x_pred, P_pred = predict(x, P, F, Q)

    # Measurement update step
    association_probs = association_probabilities(z, x_pred, P_pred, H, R, measurements)

    # Print association probabilities
    print("\nMeasurement:", z)
    for i, prob in enumerate(association_probs):
        print("Measurement", i+1, "Probability:", prob)
