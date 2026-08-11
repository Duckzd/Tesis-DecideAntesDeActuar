"""Banco de pruebas: corre políticas sobre el mismo stream y compara — Fase 5 (inicio).

Un solo harness para todas las políticas. Aprovecha los *common random numbers* del
simulador (misma semilla → mismo stream) para que la comparación sea **justa**: cada
política ve exactamente la misma secuencia de lotes y señales.

Nota: por ahora las intervenciones se **registran** (para que funcione el cooldown) pero el
modelo NO se re-entrena todavía — eso es la Fase 4 (acciones con efecto real). Aquí las
políticas se evalúan como *observadoras* sobre un stream fijo: sirve para ver **qué
decide** cada una y cuándo, que es justo lo que distingue una política de otra.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from ..acciones import aplicar_accion, dataset_refit
from ..politicas import Accion, Contexto
from ..senales.monitor import MonitorSenales, Referencia
from ..simulador.simulador import Simulador
from .costo import Costos, costo_de_accion, dano_mes


def evaluar(politica, pool, config, plan, modelo, referencia: Referencia) -> pd.DataFrame:
    """Corre UNA política sobre el stream y devuelve una fila por período.

    Columnas: periodo, tipo_real (del plan), psi_score, auc_tardio (revelado), accion, razon.
    """
    sim = Simulador(pool, config, plan)
    gt = sim.ground_truth().set_index("periodo")
    mon = MonitorSenales(referencia)
    filas = []
    for lote in sim.stream():
        rep = mon.observar(lote, modelo)
        ctx = Contexto(historia=mon.reportes[:-1])       # historia = reportes previos
        dec = politica.decidir(rep, ctx)
        if dec.accion != Accion.ESPERAR:
            mon.registrar_intervencion(rep.periodo)       # habilita el cooldown
        filas.append({
            "periodo": rep.periodo,
            "tipo_real": gt.loc[rep.periodo, "tipo"],
            "psi_score": round(rep.psi_score, 3),
            "auc_tardio": (round(rep.auc_revelado, 3) if rep.auc_revelado is not None else None),
            "accion": dec.accion.value,
            "razon": dec.razon,
        })
    return pd.DataFrame(filas)


def comparar(politicas: dict, pool, config, plan, modelo,
             referencia: Referencia) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre varias políticas sobre el MISMO stream (CRN) y las pone lado a lado.

    Devuelve (tabla, resumen):
      - tabla:  periodo · tipo_real · psi_score · auc_tardio · <accion por política>
      - resumen: conteo de acciones por política.
    """
    base = None
    resumenes = {}
    for nombre, pol in politicas.items():
        df = evaluar(pol, pool, config, plan, modelo, referencia)
        if base is None:
            base = df[["periodo", "tipo_real", "psi_score", "auc_tardio"]].copy()
        base[nombre] = df["accion"].values
        resumenes[nombre] = df["accion"].value_counts()

    resumen = pd.DataFrame(resumenes).fillna(0).astype(int)
    resumen.index.name = "accion"
    return base, resumen


def _auc_oraculo(modelo, lote) -> float | None:
    """AUC REAL del mes: la verdad que el negocio paga, con el modelo EN VIGOR ese mes.

    Usa la VarDep verdadera del lote (disponible al evaluador, oculta al agente).
    """
    y = lote.features["VarDep"].to_numpy()
    if len(set(y[~pd.isna(y)])) != 2:
        return None
    return float(roc_auc_score(y, modelo.score_malo(lote.features)))


def evaluar_con_efecto(politica, pool, config, plan, modelo0, referencia0: Referencia,
                       costos: Costos = Costos(), ventana: int = 10,
                       lag: int = 1) -> pd.DataFrame:
    """Corre una política CON efecto real (Fase 4) y puntúa el costo mes a mes.

    Cuando la política actúa, el modelo se re-ajusta (`aplicar_accion`) con los meses ya
    etiquetados (`dataset_refit`) y, tras `lag` meses, reemplaza al modelo víctima para el
    scoring, las señales del monitor y la referencia PSI. El daño se mide con el AUC real
    del modelo EN VIGOR cada mes (oráculo).
    """
    sim = Simulador(pool, config, plan)
    gt = sim.ground_truth().set_index("periodo")
    modelo = modelo0
    mon = MonitorSenales(referencia0)
    pendiente = None            # (mes_efectivo, modelo_nuevo, referencia_nueva, origen)
    filas = []

    for lote in sim.stream():
        t = lote.periodo
        # ¿entra en vigor una reparación programada?
        if pendiente is not None and t >= pendiente[0]:
            modelo, mon.ref = pendiente[1], pendiente[2]
            pendiente = None

        auc_real = _auc_oraculo(modelo, lote)          # verdad con el modelo en vigor
        dano = dano_mes(auc_real, costos)

        rep = mon.observar(lote, modelo)                # el agente ve señales del modelo actual
        dec = politica.decidir(rep, Contexto(historia=mon.reportes[:-1]))
        c_acc = costo_de_accion(dec.accion, costos)

        origen = ""
        if dec.accion != Accion.ESPERAR:
            mon.registrar_intervencion(t)
            datos = dataset_refit(pool, config, plan, t, ventana)
            if datos is not None:
                modelo_nuevo = aplicar_accion(dec.accion, modelo, datos)
                ref_nueva = Referencia.construir(datos, modelo_nuevo)
                pendiente = (t + lag, modelo_nuevo, ref_nueva, dec.accion.value)
                origen = f"re-fit({dec.accion.value}, n={len(datos)})"
            else:
                origen = "sin datos maduros aún"

        filas.append({
            "periodo": t, "tipo_real": gt.loc[t, "tipo"],
            "auc_real": (round(auc_real, 3) if auc_real is not None else None),
            "accion": dec.accion.value, "dano": round(dano, 3), "costo_accion": c_acc,
            "efecto": origen,
        })

    df = pd.DataFrame(filas)
    df["costo_acum"] = (df["dano"] + df["costo_accion"]).cumsum().round(3)
    return df


def comparar_con_efecto(politicas: dict, pool, config, plan, modelo0, referencia0: Referencia,
                        costos: Costos = Costos(), ventana: int = 10,
                        lag: int = 1) -> tuple[dict, pd.DataFrame]:
    """Corre varias políticas CON efecto real (mismo stream) y arma el veredicto de costo.

    Devuelve (trayectorias, resumen): las tablas mes a mes por política + la comparación
    de costos totales (daño + acciones), ordenada de menor a mayor.
    """
    trayectorias, resumenes = {}, {}
    for nombre, pol in politicas.items():
        df = evaluar_con_efecto(pol, pool, config, plan, modelo0, referencia0,
                                costos, ventana, lag)
        trayectorias[nombre] = df
        resumenes[nombre] = {
            "meses_danados": int((df["dano"] > 0).sum()),
            "dano_total": round(float(df["dano"].sum()), 2),
            "n_acciones": int((df["costo_accion"] > 0).sum()),
            "costo_acciones": round(float(df["costo_accion"].sum()), 2),
            "costo_total": round(float((df["dano"] + df["costo_accion"]).sum()), 2),
        }
    resumen = pd.DataFrame(resumenes).T.sort_values("costo_total")
    resumen.index.name = "politica"
    return trayectorias, resumen
