import numpy as np

def _get_potentially_optimal_rects(diams_splittable, fvals_splittable, f_min, eps, idx_splittable):
    """
    Identifica i rettangoli potenzialmente ottimali (POH) tramite lower convex hull 
    e filtro di miglioramento relativo epsilon.
    """
    
    # --- FASE 1: ORDINAMENTO E PRUNING INIZIALE ---
    # Ordinamento per diametro (primario) e valore funzione (secondario).
    order = np.lexsort((fvals_splittable, diams_splittable))
    d_sorted = diams_splittable[order]
    f_sorted = fvals_splittable[order]

    # Identificazione dell'ultimo occorrimento del minimo globale corrente.
    # Rettangoli con diametro minore e valore maggiore/uguale non possono essere POH.
    last_min_index = len(f_sorted) - 1 - np.argmin(f_sorted[::-1])

    # Riduzione del set di ricerca ai soli candidati con d >= d(f_min).
    d_sorted = d_sorted[last_min_index:]
    f_sorted = f_sorted[last_min_index:]

    # --- FASE 2: RAGGRUPPAMENTO PER DIAMETRO ---
    # Per ogni gruppo di rettangoli con lo stesso diametro, si conserva solo il valore minimo.
    diff_d = np.diff(d_sorted)
    is_best_in_group = np.concatenate(([True], diff_d > 1e-12))
    
    cand_d = d_sorted[is_best_in_group]
    cand_f = f_sorted[is_best_in_group]
    
    # --- FASE 3: COSTRUZIONE DEL LOWER CONVEX HULL ---
    # Algoritmo incrementale (Monotone Chain) per individuare i punti che minimizzano 
    # la funzione per un qualche valore della costante di Lipschitz K.
    hull = [0] # Il primo punto (minimo corrente) è sempre parte dell'inviluppo.

    for i in range(len(cand_d) - 1):
        while len(hull) > 1:
            p2 = i + 1
            p1 = hull[-1]
            p0 = hull[-2]
            
            # Calcolo pendenze tra i punti per verificare la convessità.
            slope2 = (cand_f[p2] - cand_f[p1]) / (cand_d[p2] - cand_d[p1])
            
            # Se la pendenza è non-positiva, il punto non può essere ottimale.
            if slope2 <= 0:
                hull.pop()
                continue

            slope1 = (cand_f[p1] - cand_f[p0]) / (cand_d[p1] - cand_d[p0])
            
            # Se slope2 <= slope1, p1 è ridondante (non è un vertice dell'inviluppo inferiore).
            if slope2 <= slope1:
                hull.pop()
                continue
            break

        hull.append(i + 1)

    # --- FASE 4: FILTRO EPSILON (Miglioramento Relativo) ---
    # Si scartano i rettangoli che non garantiscono un miglioramento significativo.
    epsilon_term = eps * max(abs(f_min), 1.0)

    # Ricerca del primo punto dell'inviluppo (oltre al minimo globale) che soddisfa il filtro.
    second_poh_index = 1
    while True:
        if second_poh_index >= len(hull) - 1:
            break
        
        # K_max calcolato come pendenza verso il punto successivo nel hull.
        K_max = (cand_f[hull[second_poh_index+1]] - cand_f[hull[second_poh_index]]) / \
                (cand_d[hull[second_poh_index+1]] - cand_d[hull[second_poh_index]])
        
        if cand_f[hull[second_poh_index]] - cand_d[hull[second_poh_index]] * K_max <= f_min - epsilon_term:
            break
        second_poh_index += 1

    # Costruzione lista finale POH: minimo globale (hull[0]) + punti che passano il filtro.
    poh_list = hull[:1] + hull[second_poh_index:]   

    # --- FASE 5: MAPPATURA AGLI INDICI ORIGINALI ---
    poh_indices = np.array(poh_list)
    
    # Mapping inverso: cand_d -> d_sorted -> diams_splittable -> indici globali.
    local_indices = last_min_index + np.where(is_best_in_group)[0][poh_indices]
    
    return idx_splittable[order[local_indices]]


def validate_bounds(bounds):
    """
    Valida la struttura e il contenuto dei limiti del dominio di ricerca.
    """
    if not isinstance(bounds, (list, tuple, np.ndarray)):
        raise TypeError("I bounds devono essere una sequenza (lista, tupla o array).")

    if len(bounds) == 0:
        raise ValueError("L'elenco dei bounds non può essere vuoto.")

    for i, b in enumerate(bounds):
        if not isinstance(b, (list, tuple, np.ndarray)) or len(b) != 2:
            raise ValueError(f"L'elemento {i} deve essere una coppia (lower, upper).")

        lower, upper = b

        # Verifica che i limiti siano numerici e finiti (evita NaN/Inf).
        if not isinstance(lower, (int, float, np.floating, np.integer)) or \
           not isinstance(upper, (int, float, np.floating, np.integer)):
            raise TypeError(f"I limiti dell'elemento {i} devono essere numeri reali.")

        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError(f"I limiti dell'elemento {i} devono essere valori finiti.")

        # Condizione di ammissibilità del dominio.
        if lower >= upper:
            raise ValueError(f"Errore all'elemento {i}: lower ({lower}) deve essere minore di upper ({upper}).")