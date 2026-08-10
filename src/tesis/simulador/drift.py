"""Inyectores de deterioro controlado — Tesis PR-196.

Cada inyector toma la **población de referencia** (`pool`, ver `poblacion.py`) y
devuelve un lote semanal de tamaño ``n`` con el deterioro aplicado según ``intensidad``
∈ [0, 1] (0 = lote sano, 1 = deterioro máximo). Todos comparten la firma::

    muestrear(pool, n, rng, intensidad) -> pd.DataFrame

Se implementan tres mecanismos, elegidos por tener **firmas de señales distintas**:

- ``covariate_shift``    — remuestreo sesgado a clientes riesgosos. P(X) cambia,
                           X→y se preserva. PSI de features ↑, AUC ~estable, bad-rate ↑.
- ``concept_drift``      — invierte etiquetas en una región coherente. X→y se rompe.
                           AUC ↓, PSI de features ~estable, prior ~constante.
- ``mezcla_poblacional`` — remuestreo por clase a un bad-rate objetivo. Prior P(y)
                           cambia, discriminación ~estable.

El "sano" (``intensidad=0``, o ``muestra_sana``) es muestreo uniforme con reemplazo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fuentes de mora del buró usadas para sesgar el covariate shift hacia alto riesgo.
VARS_RIESGO = [
    "MAX_DVEN_SCE_6M", "PROM_VEN_SCE_6M", "maySalVen24M269",
    "NENT_VEN_SCE_24M", "NOPE_VENC_OP_3M", "NOPE_XVEN_OP_3M",
]


def muestra_sana(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Lote de referencia: muestreo uniforme con reemplazo."""
    idx = rng.integers(0, len(pool), size=n)
    return pool.iloc[idx].reset_index(drop=True)


def covariate_shift(pool: pd.DataFrame, n: int, rng: np.random.Generator,
                    intensidad: float, vars_riesgo: list[str] | None = None) -> pd.DataFrame:
    """Remuestreo sesgado hacia clientes de alto riesgo (preserva cada par X, y).

    El peso de cada registro combina el rango percentil de sus variables de mora;
    ``intensidad`` tuerce el muestreo desde uniforme hacia el extremo riesgoso.
    """
    vars_riesgo = vars_riesgo or VARS_RIESGO
    cols = [c for c in vars_riesgo if c in pool.columns]
    riesgo = pool[cols].rank(pct=True).mean(axis=1).fillna(0.5).to_numpy()  # 0..1
    w = (1.0 - intensidad) + intensidad * (2.0 * riesgo)   # sube peso a alto riesgo
    w = w / w.sum()
    idx = rng.choice(len(pool), size=n, replace=True, p=w)
    return pool.iloc[idx].reset_index(drop=True)


def mezcla_poblacional(pool: pd.DataFrame, n: int, rng: np.random.Generator,
                       intensidad: float, delta: float = 0.35) -> pd.DataFrame:
    """Remuestreo por clase hacia un bad-rate objetivo (cambia el prior P(y)).

    El bad-rate objetivo = base + intensidad·delta (topado en 0.97). Cada registro
    conserva su (X, y): solo cambia la proporción bueno/malo del lote.
    """
    base = float(pool["VarDep"].mean())
    objetivo = min(0.97, base + intensidad * delta)
    n_malos = int(round(n * objetivo))
    n_buenos = n - n_malos
    malos = pool[pool["VarDep"] == 1]
    buenos = pool[pool["VarDep"] == 0]
    im = rng.integers(0, len(malos), size=n_malos)
    ib = rng.integers(0, len(buenos), size=n_buenos)
    batch = pd.concat([malos.iloc[im], buenos.iloc[ib]], ignore_index=True)
    return batch.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


CONCEPT_VARS = ["MAX_DVEN_SCE_6M", "numOpsVencidas101", "NOPE_VENC_OP_3M"]


def concept_drift(pool: pd.DataFrame, n: int, rng: np.random.Generator,
                  intensidad: float, vars_concepto: list[str] | None = None,
                  frac_max: float = 0.9) -> pd.DataFrame:
    """Concept drift **aprendible y dirigido** (no ruido): invierte la relación X→y.

    Simula un cambio de régimen coherente definido por FEATURES: un segmento que antes
    era 'seguro' (bajo riesgo según sus variables de mora) **ahora incumple**, y uno
    'riesgoso' **ahora paga**. El flip se decide por las variables (no al azar ni por el
    score del modelo), de modo que:

    - un modelo **reentrenado** puede aprender la relación nueva y **recuperarse**;
    - el modelo viejo se equivoca (aplica la regla vieja) → AUC ↓;
    - las **variables de entrada NO cambian** → invisible al PSI;
    - el flip es **balanceado** → el bad-rate se mantiene ~constante.

    ``riesgo`` combina el rango percentil de las variables de mora; el segmento seguro
    (riesgo bajo) y el riesgoso (riesgo alto) intercambian su comportamiento.
    """
    batch = muestra_sana(pool, n, rng)
    cols = [c for c in (vars_concepto or CONCEPT_VARS) if c in batch.columns]
    if intensidad <= 0 or not cols:
        return batch

    riesgo = batch[cols].rank(pct=True).mean(axis=1).to_numpy()   # 0 = seguro, 1 = riesgoso
    y = batch["VarDep"].to_numpy()
    seguro = riesgo <= np.quantile(riesgo, 0.45)
    riesgoso = riesgo >= np.quantile(riesgo, 0.55)

    seguro_a_malo = np.where(seguro & (y == 0))[0]    # antes buenos → ahora incumplen
    riesgoso_a_bueno = np.where(riesgoso & (y == 1))[0]  # antes malos → ahora pagan
    k = int(round(min(len(seguro_a_malo), len(riesgoso_a_bueno)) * intensidad * frac_max))
    if k > 0:
        col = batch.columns.get_loc("VarDep")
        batch.iloc[rng.choice(seguro_a_malo, k, replace=False), col] = 1
        batch.iloc[rng.choice(riesgoso_a_bueno, k, replace=False), col] = 0
    return batch


# Registro nombre -> función, para configurar el plan de deterioro por nombre.
INYECTORES = {
    "sano": lambda pool, n, rng, intensidad: muestra_sana(pool, n, rng),
    "covariate_shift": covariate_shift,
    "concept_drift": concept_drift,
    "mezcla_poblacional": mezcla_poblacional,
}
