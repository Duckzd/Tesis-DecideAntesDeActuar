"""Función de costo asimétrico — Tesis PR-196 (Fase 5, inicio).

La métrica que convierte "qué decide cada política" en "quién gana". En vocabulario MDP
es la **recompensa negada**: el agente quiere minimizar el costo total de la trayectoria.

    costo = Σ(daño del mes) + Σ(costo de la acción)

- **Daño del mes** (decisión 1B): proporcional a la caída del AUC REAL del mes (oráculo),
  en "meses-de-daño equivalentes" — un mes totalmente roto (AUC≈0.5) = 1.0, sano = 0:

      daño = clip( (AUC_base − AUC_real) / (AUC_base − 0.5),  0,  cap )

  Se mide con el AUC del mes EN CURSO (lo que el negocio paga hoy), no el tardío. El
  agente decide con señales parciales; el costo es objetivo. Esa brecha es la tesis.

- **Costo de acción**: escalonado por esfuerzo (misma unidad que el daño). Son perillas
  de calibración; el ratio daño/acción define el punto de equilibrio entre tardar y
  sobre-actuar.

Este módulo es **contabilidad pura**: puntúa cualquier trayectoria (AUC del mes + acción).
Cómo una acción *repara* el modelo (y por tanto reduce el daño futuro) es el "efecto de la
acción" — vive en el evaluador / la Fase 4 (re-fit real), no aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..politicas import Accion


@dataclass
class Costos:
    """Parámetros de la función de costo (calibrables)."""
    auc_base: float = 0.885                 # AUC del modelo sano (OOT honesto)
    piso_auc: float = 0.5                   # AUC de un modelo aleatorio → daño = 1.0
    cap: float = 1.5                        # tope del daño mensual (peor que aleatorio)
    costo_accion: dict = field(default_factory=lambda: {
        Accion.ESPERAR.value: 0.0,
        Accion.FINE_TUNE.value: 0.5,        # reajuste leve (capa final)
        Accion.REENTRENAR.value: 2.0,       # imputador + escalador + modelo
        Accion.RECONSTRUIR.value: 5.0,      # binning + selección + todo
    })


def dano_mes(auc_real: float | None, c: Costos = Costos()) -> float:
    """Daño de un mes (0 = sano) según su AUC real. `None` → 0 (sin oráculo, no se cuenta)."""
    if auc_real is None:
        return 0.0
    sev = (c.auc_base - auc_real) / (c.auc_base - c.piso_auc)
    return round(float(min(max(sev, 0.0), c.cap)), 4)


def costo_de_accion(accion: Accion, c: Costos = Costos()) -> float:
    """Costo de una acción, en la misma unidad que el daño."""
    clave = accion.value if isinstance(accion, Accion) else str(accion)
    return float(c.costo_accion.get(clave, 0.0))


def tabla_costos(trayectoria: pd.DataFrame, c: Costos = Costos()) -> pd.DataFrame:
    """Anota una trayectoria mes a mes con daño, costo de acción y costo acumulado.

    `trayectoria` debe tener al menos las columnas ``auc_real`` y ``accion`` (str o Accion).
    Devuelve una copia con columnas añadidas: ``dano``, ``costo_accion``, ``costo_acum``.
    """
    df = trayectoria.copy()
    df["dano"] = df["auc_real"].apply(lambda a: dano_mes(a, c))
    df["costo_accion"] = df["accion"].apply(lambda a: costo_de_accion(a, c))
    df["costo_acum"] = (df["dano"] + df["costo_accion"]).cumsum().round(4)
    return df


def puntuar(trayectoria: pd.DataFrame, c: Costos = Costos()) -> dict:
    """Resume una trayectoria en el veredicto de costo.

    Devuelve: meses_danados, dano_total, n_acciones, costo_acciones, costo_total.
    """
    df = tabla_costos(trayectoria, c)
    n_acc = int((df["costo_accion"] > 0).sum())
    return {
        "meses_danados": int((df["dano"] > 0).sum()),
        "dano_total": round(float(df["dano"].sum()), 2),
        "n_acciones": n_acc,
        "costo_acciones": round(float(df["costo_accion"].sum()), 2),
        "costo_total": round(float((df["dano"] + df["costo_accion"]).sum()), 2),
    }


def comparar_costos(trayectorias: dict[str, pd.DataFrame], c: Costos = Costos()) -> pd.DataFrame:
    """Puntúa varias políticas (nombre → trayectoria) y las ordena por costo total."""
    filas = {nombre: puntuar(tr, c) for nombre, tr in trayectorias.items()}
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "politica"
    return tabla.sort_values("costo_total")
