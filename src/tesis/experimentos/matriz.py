"""Experimentos a escala: matriz de escenarios × semillas — Tesis PR-196 (Fase 5).

Corre las políticas sobre varios escenarios y varias semillas, y agrega el costo en
media ± desviación. Así el resultado deja de ser un caso suelto y se vuelve **robusto**.

Escenarios (persistentes, desde el mes 2):
- sano        : control, sin deterioro → todos deberían esperar (costo ~0).
- covariate   : la distribución cambia pero el modelo sigue ordenando bien → la trampa de
                la FALSA ALARMA (reglas gasta de más; multi/llm esperan).
- concept     : el modelo se rompe con PSI casi intacto → la trampa del PUNTO CIEGO
                (reglas nunca actúa; multi/llm actúan al madurar el AUC).
- mezcla      : cambia la composición (prior) pero el AUC se mantiene → sin daño real.

Usa el LLM SIMULADO (gratis) por defecto para la fábrica de políticas; el LLM real se
reserva para una corrida final mínima con caché de decisiones.
"""

from __future__ import annotations

import time

import pandas as pd

from ..senales.monitor import Referencia
from ..simulador.simulador import ConfigSimulacion, plan_sano, agregar_episodio
from .banco import evaluar_con_efecto
from .costo import Costos

ESCENARIOS = ["sano", "covariate", "concept", "mezcla"]
_TIPO = {"covariate": "covariate_shift", "concept": "concept_drift", "mezcla": "mezcla_poblacional"}


def construir_plan(nombre: str, N: int, inicio: int = 2, intensidad: float = 1.0):
    """Plan de deterioro persistente (desde `inicio` hasta el final) del tipo pedido."""
    plan = plan_sano(N)
    if nombre == "sano":
        return plan
    return agregar_episodio(plan, _TIPO[nombre], inicio=inicio, fin=N,
                            intensidad=intensidad, forma="escalon")


def correr_matriz(fabrica, escenarios, semillas, pool, modelo, referencia: Referencia,
                  N: int = 42, costos: Costos = Costos(), ventana: int = 10, lag: int = 1,
                  verbose: bool = True) -> pd.DataFrame:
    """Corre `fabrica()` (dict nombre→política) sobre cada escenario × semilla.

    Devuelve un DataFrame largo: una fila por (escenario, semilla, política) con el costo.
    """
    filas = []
    for esc in escenarios:
        plan = construir_plan(esc, N)
        for sem in semillas:
            cfg = ConfigSimulacion(n_periodos=N, tam_lote=2000, semilla=sem)
            t = time.time()
            for nombre, pol in fabrica().items():
                df = evaluar_con_efecto(pol, pool, cfg, plan, modelo, referencia,
                                        costos=costos, ventana=ventana, lag=lag)
                filas.append({
                    "escenario": esc, "semilla": sem, "politica": nombre,
                    "costo_total": round(float((df["dano"] + df["costo_accion"]).sum()), 2),
                    "dano_total": round(float(df["dano"].sum()), 2),
                    "n_acciones": int((df["costo_accion"] > 0).sum()),
                })
            if verbose:
                print(f"  {esc:10s} semilla {sem}: {time.time()-t:.0f}s")
    return pd.DataFrame(filas)


def resumen_matriz(df: pd.DataFrame, metrica: str = "costo_total") -> pd.DataFrame:
    """Agrega la matriz: media ± desviación de `metrica` por (escenario, política)."""
    g = df.groupby(["escenario", "politica"])[metrica].agg(["mean", "std", "count"])
    g["resumen"] = g.apply(lambda r: f"{r['mean']:.1f} ± {0.0 if pd.isna(r['std']) else r['std']:.1f}", axis=1)
    tabla = g["resumen"].unstack("politica")
    # ordenar escenarios de forma legible
    orden = [e for e in ESCENARIOS if e in tabla.index]
    return tabla.reindex(orden)
