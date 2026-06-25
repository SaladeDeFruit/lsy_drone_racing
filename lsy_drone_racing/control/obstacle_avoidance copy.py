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

    # 1. Calculer le paramètre t sur le segment de vol
    if denom < 1e-6:
        t = 0.5 
    else:
        t = (b * e_val - c * d_val) / denom
    t = np.clip(t, 0.0, 1.0)

    closest_p = p1 + t * d

    # 2. Calculer s sur l'axe de l'obstacle
    s = np.dot(closest_p - c0, u)
    
    # NOUVEAU : Si c'est un segment fini, on bloque s aux extrémités du bord
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

    # Le vecteur d'éjection : de l'axe du cylindre vers le point de collision
    vec_out = closest_p - closest_c
    norm_vec = np.linalg.norm(vec_out)

    if norm_vec < 1e-5:
        # CAS CRITIQUE : Trajectoire tapant pile le centre du cylindre.
        # On force une éjection orthogonale à la fois à l'axe du cylindre ET à la trajectoire
        u = np.array(cyl_axis, dtype=float)
        u = u / np.linalg.norm(u)
        vec_out = np.cross(u, normal)
        
        # Si le drone vole parfaitement dans l'axe du tube, on prend un axe arbitraire orthogonal
        if np.linalg.norm(vec_out) < 1e-5:
            vec_out = np.cross(np.array([1.0, 0.0, 0.0]), u)
            if np.linalg.norm(vec_out) < 1e-5:
                vec_out = np.cross(np.array([0.0, 1.0, 0.0]), u)
                
    vec_out = vec_out / np.linalg.norm(vec_out)

    # Le nouveau centre est repoussé le long de ce vecteur jusqu'à la limite de sécurité
    new_center = closest_c + vec_out * safe_radius

    # Calcul des axes locaux de la fenêtre
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
    Transforme les 4 bords de chaque fenêtre en une liste d'obstacles (cylindres finis).
    
    Args:
        windows_corners: Liste des fenêtres (chaque fenêtre = 4 coins 3D).
        frame_thickness: L'épaisseur physique du cadre de la porte en mètres.
    """
    obstacles = []
    
    for corners in windows_corners:
        for i in range(4):
            # Prendre le coin actuel et le suivant (modulo 4 pour boucler)
            pA = np.array(corners[i], dtype=float)
            pB = np.array(corners[(i + 1) % 4], dtype=float)
            
            # Calcul des propriétés du cylindre fini
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
                'length': length # C'est cet argument qui rendra le cylindre fini !
            })
            
    return obstacles

def insert_avoidance_windows(windows_corners, obstacles, drone_radius, gate_opening=0.4):
    """
    Parcourt une liste de fenêtres (coins) et insère des fenêtres d'évitement
    si la ligne entre le centre de la fenêtre n-1 et n croise un obstacle.
    
    obstacles: liste de dict [{'pos': [x,y,z], 'radius': r, 'axis': [u,v,w] (optionnel)}, ...]
    """
    if len(windows_corners) < 2:
        return windows_corners

    safe_windows = [windows_corners[0]]
    
    for i in range(1, len(windows_corners)):
        p1 = get_window_center(safe_windows[-1])
        p2 = get_window_center(windows_corners[i])
        
        # Pour simplifier, on vérifie l'obstacle le plus gênant et on passe à la suite.
        # Dans un environnement très dense, on pourrait trier les obstacles par 't' croissant.
        for obs in obstacles:
            obs_pos = obs['pos']
            
            # Si l'axe n'est pas spécifié, on assume un cylindre vertical classique
            obs_axis = obs.get('axis', [0.0, 0.0, 1.0]) 
            obs_length = obs.get('length', None) # <-- NOUVEAU
            total_radius = obs['radius'] + drone_radius + 0.1 
            
            hit, closest_p, closest_c = check_segment_cylinder_collision_3d(
                p1, p2, obs_pos, obs_axis, total_radius, obs_length # <-- NOUVEAU
            )
            
            if hit:
                # Créer une fenêtre d'évitement 3D complète (N'OUBLIE PAS obs_axis !)
                tangent_window = create_tangent_window_3d(
                    p1, p2, closest_p, closest_c, obs_axis, total_radius, gate_opening
                )
                safe_windows.append(tangent_window)
                break # On passe à la porte suivante
                
        # Après avoir potentiellement ajouté une fenêtre d'esquive, on ajoute la fenêtre cible
        safe_windows.append(windows_corners[i])
        
    return safe_windows