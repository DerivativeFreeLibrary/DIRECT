import numpy as np
from .utils import _get_potentially_optimal_rects
from .utils import validate_bounds

def direct_optimize_batch(func, bounds, max_iter=50, max_evals=20000, eps=1e-4):
    """
    Implementazione batch-parallelizzata dell'algoritmo DIRECT per l'ottimizzazione globale.
    """
    
    # CREAZIONE STRUTTURE DATI EFFICIENTI PER I BOUNDS
    # Creazione preventiva delle strutture per la trasformazione da coordinate [0,1] a reali.
    # Il salvataggio di diff_bounds e lower_bounds evita ricalcoli inutili nelle iterazioni.
    validate_bounds(bounds)

    bounds = np.array(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    diff_bounds = upper_bounds - lower_bounds
    n_dims = len(bounds)

    SAFETY_PADDING = 100
    CAPACITY = max_evals + SAFETY_PADDING

    # Structure of Arrays (SoA) in float64.
    # Il float64 permette di scendere fino a 30-35 divisioni in profondità;
    # il float32 limiterebbe l'algoritmo a circa 10-15 divisioni, portando a soluzioni approssimate.
    C_centers = np.zeros((CAPACITY, n_dims), dtype=np.float64)
    C_lengths = np.zeros((CAPACITY, n_dims), dtype=np.float64)
    C_fvals   = np.zeros(CAPACITY, dtype=np.float64)
    C_diams   = np.zeros(CAPACITY, dtype=np.float64)

    # CREAZIONE DEL PRIMO RETTANGOLO
    # Il primo rettangolo è l'ipercubo unitario [0,1]^n con lato 1.0 e centro 0.5.
    C_centers[0] = 0.5
    C_lengths[0] = 1.0

    # 1. PRE-CHECK: Verifica che func sia invocabile.
    if not callable(func):
        raise TypeError(f"Il parametro 'func' deve essere invocabile. Ricevuto: {type(func).__name__}")

    real_x = lower_bounds + C_centers[0] * diff_bounds
    
    # 2. TENTATIVO DI VALUTAZIONE VETTORIALIZZATA E FALLBACK
    try:
        # Input 2D (1, n_dims) per testare la vettorializzazione.
        raw_result = func(real_x[np.newaxis, :])
    except Exception as e_batch:
        try:
            # Fallback scalare (n_dims,).
            raw_result = func(real_x)
        except Exception as e_scalar:
            raise ValueError(f"Errore nella valutazione della funzione obiettivo: {e_batch} / {e_scalar}")

    # 3. CONTROLLO TIPO E VALIDAZIONE MATEMATICA
    try:
        out_array = np.asarray(raw_result, dtype=np.float64).ravel()
        if out_array.size != 1:
            raise ValueError(f"Shape errata: attesi 1 valore, ricevuti {out_array.size}.")
        f_val = float(out_array[0])
    except Exception as e_cast:
        raise TypeError(f"Errore nella conversione del risultato in float: {e_cast}")

    if not np.isfinite(f_val):
        raise ValueError(f"La funzione ha restituito un valore non valido (NaN o Inf): {f_val}")

    C_fvals[0] = f_val
    # Per efficienza viene memorizzato il quadrato della semidiagonale (d^2).
    C_diams[0] = np.sum((C_lengths[0] / 2.0)**2)

    n_rects = 1
    f_global_min = f_val
    x_best = real_x

    # MAIN LOOP
    for t in range(max_iter):
        
        if n_rects >= max_evals:
            break

        # --- SELEZIONE POH (Potentially Optimal Rectangles) ---
        # Si filtrano i rettangoli con quadrato della semidiagonale d^2 > 1e-16 (ovvero d > 10^-8).
        # Sotto questa soglia il rumore numerico domina il segnale e un'ulteriore suddivisione
        # causerebbe figli numericamente identici al padre (collasso geometrico).
        # Un POH deve giacere sul lower convex hull del grafico (diametri, valori_funzione).
        # Il filtro epsilon regola l'esplorazione, ma l'attuale minimo globale (I*) viene
        # sempre suddiviso per garantire un continuo raffinamento locale.

        splittable_mask = C_diams[:n_rects] > 1e-16
        idx_splittable = np.where(splittable_mask)[0]
    
        if len(idx_splittable) == 0:
            break
        
        rects_to_divide = _get_potentially_optimal_rects(
            C_diams[idx_splittable], 
            C_fvals[idx_splittable], 
            f_global_min, 
            eps,
            idx_splittable
        )

        if len(rects_to_divide) == 0:
            break
        
        # Identificazione dei lati di lunghezza massima per ogni padre selezionato.
        parent_lengths = C_lengths[rects_to_divide]
        max_lens = np.max(parent_lengths, axis=1, keepdims=True)
        
        # 1e-16 è un margine per le approssimazioni della rappresentazione in virgola mobile.
        split_mask = parent_lengths >= (max_lens - 1e-16)
        rows, cols = np.where(split_mask)
        
        n_evals_needed = len(rows) * 2
        if n_rects + n_evals_needed > max_evals:
            break
        
        # Generazione vettoriale dei nuovi centri tramite trisezione lungo le dimensioni identificate.
        new_centers_left = C_centers[rects_to_divide][rows]
        new_centers_right = new_centers_left.copy()
        
        active_deltas = (max_lens[rows] / 3.0).flatten()
        new_centers_left[np.arange(len(rows)), cols] -= active_deltas
        new_centers_right[np.arange(len(rows)), cols] += active_deltas

        batch_unit = np.vstack((new_centers_left, new_centers_right))
        batch_real = lower_bounds + batch_unit * diff_bounds

        # VALUTAZIONE DELLA FUNZIONE NEI NUOVI CENTRI
        expected_size = batch_real.shape[0]
        try:
            raw_results = func(batch_real)
        except Exception:
            try:
                raw_results = [func(x) for x in batch_real]
            except Exception as e:
                raise ValueError(f"Errore valutazione batch: {e}")

        all_values = np.asarray(raw_results, dtype=np.float64).ravel()
        if all_values.size != expected_size or not np.all(np.isfinite(all_values)):
            raise ValueError("Output funzione inconsistente o non finito in modalità batch.")

        # AGGIORNAMENTO BEST GLOBAL
        min_idx_batch = np.argmin(all_values)
        if all_values[min_idx_batch] < f_global_min:
            f_global_min = all_values[min_idx_batch]
            x_best = batch_real[min_idx_batch]

        # SUDDIVISIONE DEI RETTANGOLI E GENERAZIONE DEI FIGLI
        n_splits = len(rows)
        vals_left, vals_right = all_values[:n_splits], all_values[n_splits:]
        unique_parents_rel, split_counts = np.unique(rows, return_counts=True)

        cursor = 0
        write_cursor = n_rects

        for i, count in zip(unique_parents_rel, split_counts):
            p_idx = rects_to_divide[i]
            start, end = cursor, cursor + count
            cursor = end

            dims, v_l, v_r = cols[start:end], vals_left[start:end], vals_right[start:end]
            p_l, p_r = new_centers_left[start:end], new_centers_right[start:end]

            # Si ordinano le dimensioni per dare priorità ai figli più promettenti (Jones et al.).
            sort_order = np.argsort(np.minimum(v_l, v_r))
            current_max_len = max_lens[i, 0]

            for k in sort_order:
                dim_k = dims[k]
                child_len = C_lengths[p_idx].copy()
                child_len[dim_k] = current_max_len / 3.0

                # Scrittura figli direttamente nella SoA globale.
                for center, val in [(p_l[k], v_l[k]), (p_r[k], v_r[k])]:
                    C_centers[write_cursor] = center
                    C_fvals[write_cursor]   = val
                    C_lengths[write_cursor] = child_len
                    write_cursor += 1

                # Il padre viene aggiornato diventando il rettangolo centrale.
                C_lengths[p_idx, dim_k] = current_max_len / 3.0

            # Ricalcolo diametro padre dopo le suddivisioni.
            C_diams[p_idx] = np.sum((C_lengths[p_idx] / 2.0) ** 2)

        # Calcolo vettoriale dei diametri per tutti i nuovi figli.
        C_diams[n_rects:write_cursor] = np.sum((C_lengths[n_rects:write_cursor] / 2.0) ** 2, axis=1)
        n_rects = write_cursor

    return x_best, f_global_min, n_rects