"""Población de referencia para el simulador semanal — Tesis PR-196.

El simulador NO genera datos desde cero: muestrea la **población real congelada**
(`data/buro/InfoModelamiento.pkl`) para armar los lotes semanales de "producción". Sobre esos
lotes se inyecta el deterioro controlado (ver `drift.py`) y el modelo víctima los puntúa.

Se usan solo los registros **etiquetables** (`VarDep` en {0=bueno, 1=malo}), porque el
oráculo de desempeño real de cada semana (que llega con retardo) necesita una etiqueta
buena/malo bien definida.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RUTA_REFERENCIA = "data/buro/InfoModelamiento.pkl"
COHORTE_OOT = pd.Timestamp("2022-09-30")


def cargar_poblacion(ruta: str | Path = RUTA_REFERENCIA,
                     solo_etiquetables: bool = True,
                     cohorte: str | pd.Timestamp | None = COHORTE_OOT) -> pd.DataFrame:
    """Carga la población de referencia (pool del simulador).

    cohorte: si se indica, restringe a esa `FECHA_CORTE`. Por defecto usa la cohorte
        TEST_OOT (2022-09), que el modelo **no** entrenó → sin fuga de datos y como
        punto de partida de "el futuro" (la simulación arranca tras esa fecha).
        Pasar ``None`` usa toda la MDT.
    solo_etiquetables: restringe a VarDep {0,1} (población de scoring), la que tiene
        etiqueta buena/malo para calcular el oráculo de desempeño.
    """
    info = pd.read_pickle(ruta)
    if cohorte is not None:
        info = info[info["FECHA_CORTE"] == pd.Timestamp(cohorte)]
    if solo_etiquetables:
        info = info[info["VarDep"].isin([0, 1])].copy()
    return info.reset_index(drop=True)
