import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

class QuadrotorModel:
    def __init__(self):
        # cf2x_P250 (the only two used by the current cost are mass and f_max).
        self.mass = 0.0318                                    # kg
        self.f_max = 0.12                                     # N per motor (thrust_max)
        self.gravity = np.array([0, 0, -9.81])
        # Set for correctness / future use (not used by the current cost function):
        self.arm_length = 0.046                               # m (cf2x approx)
        self.inertia = np.array([16.8e-6, 16.8e-6, 29.8e-6])  # kg m^2 (J diag)
        self.c_tau = 0.0069928948992470565                    # thrust2torque
        self.omega_max = np.array([15.0, 15.0, 3.0])          # rad/s (kept; tune if needed)

class PolygonGate:
    def __init__(self, id, vertices):
        self.id = id
        self.center = np.mean(vertices, axis=0)
        self.vertices = np.array(vertices).T
        self.v = self.vertices.shape[1]       
    
    def smooth_surjection(self, d):
        d_squared = np.square(d) 
        weights = d_squared / (np.sum(d_squared) + 1e-9)
        return self.vertices @ weights

class SplineTrajectory:
    def __init__(self, waypoints, time_segments):
        self.t_points = np.insert(np.cumsum(time_segments), 0, 0.0)
        self.waypoints = np.array(waypoints)
        
        k = min(3, len(self.waypoints) - 1)
        self.spline = make_interp_spline(self.t_points, self.waypoints, k=k)
        
        self.velocity_spline = self.spline.derivative(nu=1)
        self.acceleration_spline = self.spline.derivative(nu=2)

    def get_state_at(self, t: float):
        t = np.clip(t, self.t_points[0], self.t_points[-1])
        p = self.spline(t)
        v = self.velocity_spline(t)
        a = self.acceleration_spline(t)
        return p, v, a
        
    def get_motor_thrusts_at(self, t: float, drone_model: QuadrotorModel):
        _, _, acceleration = self.get_state_at(t)
        thrust_vector = drone_model.mass * (acceleration - drone_model.gravity)
        total_thrust = np.linalg.norm(thrust_vector)
        single_motor_thrust = total_thrust / 4.0 
        return np.array([single_motor_thrust] * 4)

class DroneRacingOptimizer:
    def __init__(self, drone_model, gates, start_pos, end_pos):
        self.drone = drone_model
        self.gates = gates
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.num_gates = len(gates)
        
    def transform_time(self, K):
        return np.exp(K)
    
    def cost_function(self, variables):
        split_idx = self.num_gates * 4 
        D_vars = variables[:split_idx].reshape((self.num_gates, 4))
        K_vars = variables[split_idx:]
        
        T_segments = self.transform_time(K_vars)
        total_time = np.sum(T_segments)
        
        waypoints = [self.start_pos]
        for i, gate in enumerate(self.gates):
            waypoints.append(gate.smooth_surjection(D_vars[i]))
        waypoints.append(self.end_pos)

        penalty = self.evaluate_trajectory_constraints(waypoints, T_segments)
        return total_time + penalty
    
    def evaluate_trajectory_constraints(self, waypoints, times):
        penalty = 0.0
        
        try:
            minco_traj = SplineTrajectory(waypoints, times)
        except ValueError:
            return 1000.0

        for t in minco_traj.t_points:
            u = minco_traj.get_motor_thrusts_at(t, self.drone)
            depassements = u - self.drone.f_max
            
            for depassement in depassements:
                if depassement > 0:
                    penalty += (depassement ** 2)
                
        return penalty * 10.0
    
    def solve(self):
        initial_d = np.ones(self.num_gates * 4) 
        initial_k = np.ones(self.num_gates + 1) * 0.5 
        initial_guess = np.concatenate([initial_d, initial_k])
        
        bounds = [(None, None)] * len(initial_d) + [(-1.0, 2.0)] * len(initial_k)
        
        options = {'maxfun': 15000, 'maxiter': 1000}
        
        return minimize(self.cost_function, initial_guess, method='L-BFGS-B', bounds=bounds, options=options)