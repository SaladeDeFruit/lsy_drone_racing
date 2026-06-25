import time
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

# ==========================================
# 1. MODÈLE PHYSIQUE DU DRONE
# ==========================================
class QuadrotorModel:
    def __init__(self):
        self.mass = 0.85
        self.arm_length = 0.15
        self.inertia = np.array([1.0, 1.0, 1.7]) * 1e-3
        self.f_max = 6.88
        self.c_tau = 0.05
        self.omega_max = np.array([15.0, 15.0, 3.0])
        self.gravity = np.array([0, 0, -9.81])

# ==========================================
# 2. GÉOMÉTRIE DES PORTES
# ==========================================
class PolygonGate:
    def __init__(self, id, vertices):
        self.id = id
        self.center = np.mean(vertices, axis=0)
        self.vertices = np.array(vertices).T
        self.v = self.vertices.shape[1]       
    
    def smooth_surjection(self, d):
        d_squared = np.square(d) 
        # L'epsilon (1e-9) empêche la division par zéro (le fameux NaN)
        weights = d_squared / (np.sum(d_squared) + 1e-9)
        return self.vertices @ weights 

# ==========================================
# 3. TRAJECTOIRE FLUIDE (SPLINE)
# ==========================================
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



# ==========================================
# 4. L'OPTIMISEUR L-BFGS
# ==========================================
class DroneRacingOptimizer:
    def __init__(self, drone_model, gates, start_pos, end_pos):
        self.drone = drone_model
        self.gates = gates
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.num_gates = len(gates)
        
    def transform_time(self, K):
        return np.exp(K)
    
    def _build_waypoints(self, D_vars, T_segments):
        """
        Reconstruit la liste complète des waypoints 3D à partir des variables optimisées.
        (Départ -> Portes -> Arrivée)
        """
        waypoints = [self.start_pos]
        
        for i, gate in enumerate(self.gates):
            waypoints.append(gate.smooth_surjection(D_vars[i]))
            
        waypoints.append(self.end_pos)
        
        return waypoints, T_segments
    
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

        # On vérifie la physique uniquement aux points de passage (accélération max)
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

# ==========================================
# TEST ET VISUALISATION (4 PORTES)
# ==========================================
if __name__ == "__main__":
    print("Initialisation du circuit en zigzag à 4 PORTES...")
    drone = QuadrotorModel()
    start_pos = [0, 0, 0]
    end_pos = [25, 0, 0]
    
    # Création de 4 portes rectangulaires (2m x 2m) en zigzag
    gate1 = PolygonGate(1, [[5, -1, -1], [5, 1, -1], [5, 1, 1], [5, -1, 1]])    # Centrée
    gate2 = PolygonGate(2, [[10, 4, -1], [10, 6, -1], [10, 6, 1], [10, 4, 1]])  # Décalée à gauche
    gate3 = PolygonGate(3, [[15, -1, -1], [15, 1, -1], [15, 1, 1], [15, -1, 1]])# Centrée
    gate4 = PolygonGate(4, [[20, 4, -1], [20, 6, -1], [20, 6, 1], [20, 4, 1]])  # Décalée à gauche
    
    gates = [gate1, gate2, gate3, gate4]
    
    optimizer = DroneRacingOptimizer(drone, gates, start_pos, end_pos)
    
    print("Recherche de la trajectoire optimale... (Calcul pur sans bruit)")
    
    start_calc_time = time.time()
    result = optimizer.solve()
    end_calc_time = time.time()
    
    print(f"--> Temps de calcul de l'algorithme : {end_calc_time - start_calc_time:.4f} secondes")
    
    if result.success or result.status == 0:
        print(f"--> Succès ! Temps de vol estimé pour le drone : {np.sum(np.exp(result.x[16:])):.2f} secondes")
        
        D_final = result.x[:16].reshape((4, 4))
        T_final = np.exp(result.x[16:])
        
        final_waypoints = [start_pos] + [gates[i].smooth_surjection(D_final[i]) for i in range(4)] + [end_pos]
        traj = SplineTrajectory(final_waypoints, T_final)
        
        # --- Affichage 3D ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        t_eval = np.linspace(0, np.sum(T_final), 200)
        path = np.array([traj.get_state_at(t)[0] for t in t_eval])
        ax.plot(path[:,0], path[:,1], path[:,2], label='Trajectoire Optimisée', color='red', linewidth=2)
        
        for i, g in enumerate(gates):
            v = np.vstack((g.vertices.T, g.vertices.T[0]))
            ax.plot(v[:,0], v[:,1], v[:,2], color='blue', linewidth=2, alpha=0.7)
            ax.text(g.center[0], g.center[1], g.center[2]+1.5, f"Gate {i+1}", color='blue')
            
        ax.scatter(*zip(*final_waypoints), color='green', s=60, label='Waypoints (Apex)')
        ax.set_title("Course de Drone Autonome (4 Portes)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.show()
    else:
        print("L'optimisation a échoué:", result.message)