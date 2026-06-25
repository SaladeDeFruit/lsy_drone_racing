import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

DRONE_RADIUS = 0.15

class QuadrotorModel:
    def __init__(self):
        self.mass = 0.85
        self.arm_length = 0.15
        self.drone_radius = self.arm_length + 0.08  # ← à ajouter
        self.inertia = np.array([1.0, 1.0, 1.7]) * 1e-3
        self.f_max = 6.88
        self.c_tau = 0.05
        self.omega_max = np.array([15.0, 15.0, 3.0])
        self.gravity = np.array([0, 0, -9.81])

class PolygonGate:
    def __init__(self, id, vertices, margin=0.15): 
        self.id = id
        self.center = np.mean(vertices, axis=0)
        
        # --- 1. Ajout de la marge de sécurité (Shrinking) ---
        safe_vertices = []
        for v in vertices:
            vector_to_center = self.center - v
            dist = np.linalg.norm(vector_to_center)
            shrink_ratio = max(0, dist - margin) / dist 
            safe_vertices.append(self.center - vector_to_center * shrink_ratio)
            
        self.vertices = np.array(safe_vertices).T
        self.v = self.vertices.shape[1]       
        
        # --- 2. Calcul des axes de la porte en 3D ---
        # On prend 3 points pour calculer le vecteur normal (perpendiculaire à la porte)
        v0, v1, v2 = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        self.normal = np.cross(v1 - v0, v2 - v1)
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-9)
        
        # L'axe horizontal est perpendiculaire à la normale ET à l'axe Z mondial (la gravité)
        z_world = np.array([0.0, 0.0, 1.0])
        self.horizontal_axis = np.cross(z_world, self.normal)
        
        # Normalisation (avec sécurité si la porte est posée à plat au sol, type "dive")
        norm_h = np.linalg.norm(self.horizontal_axis)
        if norm_h > 1e-6:
            self.horizontal_axis = self.horizontal_axis / norm_h
        else:
            self.horizontal_axis = np.array([1.0, 0.0, 0.0])
    
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
        gate_penalty = self._gate_frame_collision_penalty(waypoints, T_segments)

        return total_time + penalty + gate_penalty
    
    def _gate_frame_collision_penalty(self, waypoints, times):
        penalty = 0.0
        try:
            traj = SplineTrajectory(waypoints, times)
        except ValueError:
            return 500.0

        t_candidates = np.linspace(traj.t_points[0], traj.t_points[-1], 500)

        for gate_idx, gate in enumerate(self.gates):
            vertical_axis = np.cross(gate.normal, gate.horizontal_axis)

            h_coords = [np.dot(v - gate.center, gate.horizontal_axis) for v in gate.vertices.T]
            v_coords = [np.dot(v - gate.center, vertical_axis) for v in gate.vertices.T]
            half_w = max(abs(h) for h in h_coords)
            half_h = max(abs(v) for v in v_coords)

            inner_half_w = half_w - DRONE_RADIUS
            inner_half_h = half_h - DRONE_RADIUS

            # Trouver le croisement du plan de cette porte
            signed_dists = np.array([
                np.dot(traj.get_state_at(t)[0] - gate.center, gate.normal)
                for t in t_candidates
            ])
            sign_changes = np.where(np.diff(np.sign(signed_dists)))[0]

            if len(sign_changes) == 0:
                penalty += 200.0
                continue

            # Point de croisement exact
            idx = sign_changes[0]
            t_cross = (t_candidates[idx] + t_candidates[idx + 1]) / 2.0
            crossing_point, _, _ = traj.get_state_at(t_cross)

            diff = crossing_point - gate.center
            h_proj = np.dot(diff, gate.horizontal_axis)
            v_proj = np.dot(diff, vertical_axis)

            # Trop à droite/gauche ou trop haut/bas → touche le cadre
            h_overshoot = max(0.0, abs(h_proj) - inner_half_w)
            v_overshoot = max(0.0, abs(v_proj) - inner_half_h)
            penetration = max(h_overshoot, v_overshoot)
            if penetration > 0:
                penalty += penetration ** 2 * 200.0

        return penalty
        
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