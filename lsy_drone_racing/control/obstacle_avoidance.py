import numpy as np

def get_window_center(corners):
    """Calcule le centre d'une fenêtre à partir de ses 4 coins."""
    return np.mean(corners, axis=0)

def check_segment_cylinder_collision_3d(p1, p2, cyl_center, cyl_axis, cyl_radius, cyl_length=None):
    """
    Vérifie la collision entre un segment 3D et un cylindre arbitraire (infini ou fini).
    Si cyl_length est fourni, le cylindre devient une capsule finie.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    c0 = np.array(cyl_center, dtype=float)
    
    u = np.array(cyl_axis, dtype=float)
    norm_u = np.linalg.norm(u)
    u = u / norm_u if norm_u > 0 else np.array([0.0, 0.0, 1.0])

    d = p2 - p1
    w0 = p1 - c0

    a, b, c = np.dot(d, d), np.dot(d, u), np.dot(u, u)
    d_val, e_val = np.dot(w0, d), np.dot(w0, u)
    denom = a * c - b * b

    if denom < 1e-6:
        t = 0.5 
    else:
        t = (b * e_val - c * d_val) / denom
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
    """
    Crée une fenêtre 3D orientée face au vol, repoussée hors d'un cylindre arbitraire.
    """
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    closest_p = np.array(closest_p, dtype=float)
    closest_c = np.array(closest_c, dtype=float)
    
    direction = p2 - p1
    norm_dir = np.linalg.norm(direction)
    if norm_dir == 0:
        normal = np.array([1.0, 0.0, 0.0])
    else:
        normal = direction / norm_dir

    vec_out = closest_p - closest_c
    norm_vec = np.linalg.norm(vec_out)

    if norm_vec < 1e-5:
        u = np.array(cyl_axis, dtype=float)
        u = u / np.linalg.norm(u)
        vec_out = np.cross(u, normal)
        
        if np.linalg.norm(vec_out) < 1e-5:
            vec_out = np.cross(np.array([1.0, 0.0, 0.0]), u)
            if np.linalg.norm(vec_out) < 1e-5:
                vec_out = np.cross(np.array([0.0, 1.0, 0.0]), u)
                
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
    corners = np.array([
        new_center + half * v1 + half * v2,
        new_center - half * v1 + half * v2,
        new_center - half * v1 - half * v2,
        new_center + half * v1 - half * v2,
    ])
    
    return corners

def extract_obstacles_from_windows(windows_corners, frame_thickness=0.05):
    """
    Transforme les 4 bords de chaque fenêtre en une liste d'obstacles.
    NOUVEAUTÉ : Ajoute un 'gate_idx' pour savoir à quelle porte appartient le cadre.
    """
    obstacles = []
    
    for gate_idx, corners in enumerate(windows_corners):
        for i in range(4):
            pA = np.array(corners[i], dtype=float)
            pB = np.array(corners[(i + 1) % 4], dtype=float)
            
            vec = pB - pA
            length = np.linalg.norm(vec)
            
            if length < 1e-5:
                continue
                
            center = pA + (vec / 2.0)
            axis = vec / length
            
            obstacles.append({
                'pos': center.tolist(),
                'axis': axis.tolist(),
                'radius': frame_thickness,
                'length': length,
                'gate_idx': gate_idx
            })
            
    return obstacles

def insert_avoidance_windows(windows_corners, obstacles, drone_radius, gate_opening=0.4):
    """
    Insère des fenêtres d'évitement sur le parcours.
    """
    if len(windows_corners) < 2:
        return windows_corners

    safe_windows = [windows_corners[0]]
    
    for i in range(1, len(windows_corners)):
        p1 = get_window_center(safe_windows[-1])
        p2 = get_window_center(windows_corners[i])
        
        for obs in obstacles:
            obs_pos = obs['pos']
            obs_axis = obs.get('axis', [0.0, 0.0, 1.0]) 
            obs_length = obs.get('length', None)
            
            # CORRECTION : On n'ignore PLUS les cadres de la porte cible !
            # Si le drone aborde sa propre porte avec un angle trop fermé, 
            # il faut qu'il génère une esquive pour se réaligner.
            
            # On limite le rayon de sécurité pour qu'il ne "bouche" pas mathématiquement 
            # le centre de la porte. (Le centre est à gate_opening/2 du cadre).
            max_allowed_radius = (gate_opening / 2.0) - 0.02
            total_radius = min(obs['radius'] + drone_radius + 0.05, max_allowed_radius)
            
            hit, closest_p, closest_c = check_segment_cylinder_collision_3d(
                p1, p2, obs_pos, obs_axis, total_radius, obs_length
            )
            
            if hit:
                tangent_window = create_tangent_window_3d(
                    p1, p2, closest_p, closest_c, obs_axis, total_radius, gate_opening
                )
                safe_windows.append(tangent_window)
                break 
                
        safe_windows.append(windows_corners[i])
        
    return safe_windows