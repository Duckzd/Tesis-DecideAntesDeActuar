"""Orquestador del flujo por períodos — Tesis PR-196.

Genera un **stream de lotes** (por defecto **mensuales**, con fecha de corte) a partir
de la población de referencia, aplicando un **plan de deterioro** con ground truth
conocido y simulando el **retardo de etiquetas**: la `VarDep` real del período ``t``
solo se revela en ``t + retardo``.

Cada período representa **un corte**: un lote nuevo de solicitantes muestreado del pool
(ver `poblacion.py`), con el drift del plan aplicado (ver `drift.py`). La frecuencia es
configurable (``'ME'`` mensual, ``'W'`` semanal, …). El retardo por defecto es **12
meses**, coherente con la ventana de desempeño M2–M13 con que se define la `VarDep`.

El lote (`LotePeriodo.features`) incluye la `VarDep` verdadera para calcular el oráculo
de desempeño y para la revelación tardía, pero **el agente no debe leerla directamente**:
solo ve señales indirectas (Fase 2) y las etiquetas ya reveladas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import drift


@dataclass
class EspecPeriodo:
    """Deterioro planificado para un período (ground truth)."""
    tipo: str = "sano"          # clave en drift.INYECTORES
    intensidad: float = 0.0     # 0 = sano, 1 = máximo


@dataclass
class ConfigSimulacion:
    n_periodos: int = 24                  # nº de cortes (meses por defecto)
    tam_lote: int = 1500                  # solicitantes por corte
    retardo_periodos: int = 12            # cortes hasta que llega la VarDep real
    fecha_inicio: str = "2023-01-31"      # fecha del primer corte
    frecuencia: str = "ME"                # 'ME' mensual, 'W' semanal, 'QE' trimestral…
    semilla: int = 13579


@dataclass
class LotePeriodo:
    periodo: int                           # índice del corte (0, 1, 2, …)
    fecha: pd.Timestamp                     # fecha de corte del período
    features: pd.DataFrame                  # lote (incluye VarDep verdadera, oculta al agente)
    espec: EspecPeriodo                     # ground truth del deterioro de este período
    etiquetas_reveladas: pd.Series | None   # VarDep del período (t - retardo), o None
    periodo_revelado: int | None            # a qué período pertenecen esas etiquetas
    fecha_revelada: pd.Timestamp | None     # fecha de ese período revelado


# --------------------------------------------------------------------------- #
# Construcción de planes de deterioro
# --------------------------------------------------------------------------- #
def plan_sano(n_periodos: int) -> list[EspecPeriodo]:
    """Plan de referencia: todos los períodos sanos."""
    return [EspecPeriodo() for _ in range(n_periodos)]


def agregar_episodio(plan: list[EspecPeriodo], tipo: str, inicio: int, fin: int,
                     intensidad: float, forma: str = "rampa") -> list[EspecPeriodo]:
    """Inserta un episodio de deterioro en los períodos [inicio, fin).

    forma='escalon' aplica ``intensidad`` constante; forma='rampa' sube linealmente
    de 0 a ``intensidad`` a lo largo del episodio. Modifica y devuelve ``plan``.
    """
    dur = max(1, fin - inicio)
    for i, s in enumerate(range(inicio, fin)):
        if not (0 <= s < len(plan)):
            continue
        inten = intensidad if forma == "escalon" else intensidad * (i + 1) / dur
        plan[s] = EspecPeriodo(tipo=tipo, intensidad=round(float(inten), 4))
    return plan


# --------------------------------------------------------------------------- #
# Simulador
# --------------------------------------------------------------------------- #
class Simulador:
    """Produce el stream de lotes por período con deterioro y retardo de etiquetas."""

    def __init__(self, pool: pd.DataFrame, config: ConfigSimulacion,
                 plan: list[EspecPeriodo] | None = None):
        self.pool = pool
        self.config = config
        self.plan = plan or plan_sano(config.n_periodos)
        if len(self.plan) != config.n_periodos:
            raise ValueError(f"plan de {len(self.plan)} períodos != n_periodos={config.n_periodos}")
        self.fechas = pd.date_range(config.fecha_inicio, periods=config.n_periodos,
                                    freq=config.frecuencia)
        self._rng = np.random.default_rng(config.semilla)
        self._labels_hist: list[pd.Series] = []   # solo VarDep por período (no lotes completos)

    def _generar_lote(self, espec: EspecPeriodo) -> pd.DataFrame:
        fn = drift.INYECTORES[espec.tipo]
        return fn(self.pool, self.config.tam_lote, self._rng, espec.intensidad)

    def stream(self):
        """Itera período a período devolviendo un `LotePeriodo`.

        Reinicia el generador aleatorio en cada llamada → el stream es reproducible
        (idempotente) para una misma semilla. Solo retiene las `VarDep` pasadas (no los
        lotes completos) para la revelación tardía de etiquetas.
        """
        self._rng = np.random.default_rng(self.config.semilla)
        self._labels_hist = []
        retardo = self.config.retardo_periodos
        for t in range(self.config.n_periodos):
            espec = self.plan[t]
            batch = self._generar_lote(espec)

            t_rev = t - retardo
            if t_rev >= 0:
                etiquetas = self._labels_hist[t_rev]
                periodo_rev, fecha_rev = t_rev, self.fechas[t_rev]
            else:
                etiquetas, periodo_rev, fecha_rev = None, None, None

            self._labels_hist.append(batch["VarDep"].reset_index(drop=True))
            lote = LotePeriodo(periodo=t, fecha=self.fechas[t], features=batch, espec=espec,
                               etiquetas_reveladas=etiquetas, periodo_revelado=periodo_rev,
                               fecha_revelada=fecha_rev)
            yield lote

    def ground_truth(self) -> pd.DataFrame:
        """Tabla período × (fecha, tipo, intensidad) con el deterioro real planificado."""
        return pd.DataFrame([
            {"periodo": t, "fecha": self.fechas[t], "tipo": e.tipo, "intensidad": e.intensidad}
            for t, e in enumerate(self.plan)
        ])
