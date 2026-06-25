import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# 1. LOGIQUE D'ÉVITEMENT (Capsules Finies)
# ==========================================

def check_segment_cylinder_collision_3d(p1, p2, cyl_center, cyl_axis, cyl_radius, cyl_length=None):
    p1, p2, c0 = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(cyl_center, dtype=float)
    u = np.array(cyl_axis, dtype=float)
    norm_u = np.linalg.norm(u)
    u = u / norm_u if norm_u > 0 else np.array([0.0, 0.0, 1.0])

    d = p2 - p1
    w0 = p1 - c0

    a, b, c = np.dot(d, d), np.dot(d, u), np.dot(u, u)
    d_val, e_val = np.dot(w0, d), np.dot(w0, u)
    denom = a * c - b * b

    t = 0.5 if denom < 1e-6 else (b * e_val - c * d_val) / denom
    t = np.clip(t, 0.0, 1.0)
    closest_p = p1 + t * d

    s = np.dot(closest_p - c0, u)
    if cyl_length is not None:
        s = np.clip(s, -cyl_length / 2.0, cyl_length / 2.0)
        
    closest_c = c0 + s * u
    dist = np.linalg.norm(closest_p - closest_c)

    if dist < cyl_radius:
        return True, closest_p, closest_c
    return False, None, None

def create_tangent_window_3d(p1, p2, closest_p, closest_c, cyl_axis, safe_radius, gate_opening=0.4):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    closest_p, closest_c = np.array(closest_p, dtype=float), np.array(closest_c, dtype=float)
    
    direction = p2 - p1
    norm_dir = np.linalg.norm(direction)
    normal = direction / norm_dir if norm_dir > 0 else np.array([1.0, 0.0, 0.0])

    vec_out = closest_p - closest_c
    norm_vec = np.linalg.norm(vec_out)

    if norm_vec < 1e-5:
        u = np.array(cyl_axis, dtype=float)
        u = u / np.linalg.norm(u)
        vec_out = np.cross(u, normal)
        if np.linalg.norm(vec_out) < 1e-5:
            vec_out = np.cross(np.array([1.0, 0.0, 0.0]), u)
            
    vec_out = vec_out / np.linalg.norm(vec_out)
    new_center = closest_c + vec_out * safe_radius

    Z_axis = np.array([0.0, 0.0, 1.0])
    if np.abs(normal[2]) > 0.99:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.cross(Z_axis, normal)
        v1 = v1 / np.linalg.norm(v1)

    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    half = gate_opening / 2.0
    return np.array([
        new_center + half * v1 + half * v2,
        new_center - half * v1 + half * v2,
        new_center - half * v1 - half * v2,
        new_center + half * v1 - half * v2,
    ])

def extract_obstacles_from_windows(windows_corners, frame_thickness=0.05):
    obstacles = []
    for corners in windows_corners:
        for i in range(4):
            pA, pB = np.array(corners[i], dtype=float), np.array(corners[(i + 1) % 4], dtype=float)
            vec = pB - pA
            length = np.linalg.norm(vec)
            if length < 1e-5: continue
            center = pA + (vec / 2.0)
            obstacles.append({
                'pos': center.tolist(),
                'axis': (vec / length).tolist(),
                'radius': frame_thickness,
                'length': length
            })
    return obstacles

# ==========================================
# 2. SCÉNARIO ET VISUALISATION
# ==========================================

if __name__ == "__main__":
    # --- Création du scénario ---
    # 1. On crée une grande porte centrale (2x2 mètres) au milieu du parcours
    p_center = np.array([5.0, 0.0, 0.0])
    half_gate = 1.0
    central_gate = np.array([
        p_center + [0, half_gate, half_gate],
        p_center + [0, -half_gate, half_gate],
        p_center + [0, -half_gate, -half_gate],
        p_center + [0, half_gate, -half_gate]
    ])

    # 2. On extrait les 4 montants de cette porte comme obstacles
    frame_thickness = 0.05
    gate_obstacles = extract_obstacles_from_windows([central_gate], frame_thickness)

    # 3. Le parcours du drone : Il veut aller de X=0 à X=10.
    # PIÈGE : Il vole à Y=0.8. Le bord de la porte est à Y=1.0. 
    # Avec un drone de rayon 0.2, l'espace n'est pas suffisant (0.8 + 0.2 = 1.0 -> Collision)
    drone_radius = 0.2
    start_pos = np.array([0.0, 0.85, 0.0]) 
    end_pos = np.array([10.0, 0.85, 0.0])

    # --- Application de l'algorithme ---
    # On simule la logique de la boucle principale pour ce segment unique
    safe_windows = []
    total_radius = frame_thickness + drone_radius + 0.05 # +0.05 de marge

    collision_found = False
    for obs in gate_obstacles:
        hit, cp, cc = check_segment_cylinder_collision_3d(
            start_pos, end_pos, obs['pos'], obs['axis'], total_radius, obs['length']
        )
        if hit:
            print("🚨 Risque d'accrochage du cadre détecté !")
            # Création de la porte d'esquive (taille standard 0.4m)
            avoid_win = create_tangent_window_3d(
                start_pos, end_pos, cp, cc, obs['axis'], total_radius, gate_opening=0.4
            )
            safe_windows.append(avoid_win)
            collision_found = True
            break # On s'arrête au premier montant touché

    if not collision_found:
        print("✅ Trajectoire claire.")

    # --- Tracé Matplotlib ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Tracer la porte centrale (Obstacle)
    poly_gate = Poly3DCollection([central_gate], alpha=0.2, facecolor='grey', edgecolor='black')
    ax.add_collection3d(poly_gate)

    # Tracer les montants de la porte (en rouge épais) pour montrer la zone de danger
    for obs in gate_obstacles:
        c = np.array(obs['pos'])
        a = np.array(obs['axis'])
        l = obs['length'] / 2.0
        ax.plot([c[0]-a[0]*l, c[0]+a[0]*l], 
                [c[1]-a[1]*l, c[1]+a[1]*l], 
                [c[2]-a[2]*l, c[2]+a[2]*l], color='red', linewidth=4)

    # Tracer la trajectoire initiale (danger)
    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 
            color='orange', linestyle=':', label="Trajectoire Initiale (Dangereuse)")

    # Tracer l'esquive si générée
    path_x, path_y, path_z = [start_pos[0]], [start_pos[1]], [start_pos[2]]
    for w in safe_windows:
        poly_avoid = Poly3DCollection([w], alpha=0.6, facecolor='cyan', edgecolor='blue')
        ax.add_collection3d(poly_avoid)
        wc = np.mean(w, axis=0)
        path_x.append(wc[0]); path_y.append(wc[1]); path_z.append(wc[2])
    path_x.append(end_pos[0]); path_y.append(end_pos[1]); path_z.append(end_pos[2])

    # Tracer la trajectoire corrigée
    ax.plot(path_x, path_y, path_z, color='blue', linewidth=2, label="Trajectoire Sécurisée")
    ax.scatter([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color='green', s=50, label="Départ/Arrivée")

    ax.set_xlim([-1, 11]); ax.set_ylim([-2, 2]); ax.set_zlim([-2, 2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Évitement d'un bord de fenêtre (Capsule Finie)")
    ax.legend()
    plt.show()