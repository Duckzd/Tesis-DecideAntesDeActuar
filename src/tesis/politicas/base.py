"""Contrato común de las políticas de decisión — Tesis PR-196 (Fase 3).

Una **política** es el cerebro del agente: mapea el `ReporteSenales` del mes (+ contexto)
a una **acción** correctiva. Todas comparten un único método, `decidir()`, para que sean
intercambiables — y se registran en `REGISTRO` (patrón plugin: otra persona puede añadir
su propia política y compararla en el mismo harness).

Vocabulario MDP: estado = reporte + historia, acción = `Accion`, y la recompensa (costo)
se calcula aparte en la Fase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, runtime_checkable


class Accion(str, Enum):
    """Espacio de acciones correctivas (los 'brazos' de la decisión)."""
    ESPERAR = "esperar"
    FINE_TUNE = "fine_tune"
    REENTRENAR = "reentrenar"
    RECONSTRUIR = "reconstruir"


@dataclass
class Decision:
    """Lo que devuelve toda política: la acción + su justificación."""
    accion: Accion
    razon: str = ""
    score_riesgo: float | None = None   # riesgo estimado (para políticas que lo calculan)


@dataclass
class Contexto:
    """Lo observable que rodea al reporte del mes (nunca el oráculo).

    historia: reportes de meses previos (para tendencias/persistencia).
    herramientas: funciones que una política agéntica (LLM) puede invocar para
        investigar antes de decidir; las heurísticas las ignoran.
    """
    historia: list = field(default_factory=list)
    herramientas: dict[str, Callable] = field(default_factory=dict)


@runtime_checkable
class Politica(Protocol):
    """Contrato: toda política implementa `decidir(reporte, contexto) -> Decision`."""
    nombre: str

    def decidir(self, reporte, contexto: Contexto) -> Decision: ...


# --------------------------------------------------------------------------- #
# Registro de políticas (plugins) — para que el proyecto sea consumible
# --------------------------------------------------------------------------- #
REGISTRO: dict[str, type] = {}


def registrar(nombre: str):
    """Decorador que inscribe una política en el `REGISTRO` bajo `nombre`."""
    def deco(cls):
        cls.nombre = nombre
        REGISTRO[nombre] = cls
        return cls
    return deco


def crear(nombre: str, **kwargs) -> Politica:
    """Instancia una política registrada por nombre (para configs/CLI)."""
    if nombre not in REGISTRO:
        raise KeyError(f"política '{nombre}' no registrada. Disponibles: {list(REGISTRO)}")
    return REGISTRO[nombre](**kwargs)
