"""Acciones correctivas con efecto REAL — Tesis PR-196 (Fase 4).

Cuando una política decide algo ≠ esperar, el "mundo" (el evaluador) re-ajusta el modelo
víctima **de verdad** y lo reemplaza para los meses siguientes. Cada nivel de acción toca
una parte distinta de la cadena del scorecard:

- **fine_tune**  : re-ajusta SOLO la capa final (logit), reusando imputador+escalador+binning
                   congelados. Recalibración leve de coeficientes.
- **reentrenar** : re-ajusta imputador+escalador+logit sobre datos nuevos, manteniendo el
                   binning y la selección viejos.
- **reconstruir**: rehace TODO (binning + pipeline) con los datos nuevos.

Cuánto recupera cada acción NO se asume: sale de re-entrenar de verdad y medir el AUC.

El **retardo de etiquetas** condiciona el re-fit: al mes ``t`` solo hay etiquetas maduras
hasta ``t - retardo``. Por eso `dataset_refit` arma el conjunto con la ventana de meses ya
etiquetados (del mismo tamaño que el train original, por consistencia metodológica). Como
el stream es idempotente, se **regeneran** esos lotes sin retener nada pesado.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..modelo_base import (ModeloBase, RANDOM_STATE, construir_features,
                           entrenar_modelo_base, make_pipeline)
from ..politicas import Accion
from ..simulador.simulador import Simulador


def _matriz_entrenamiento(datos: pd.DataFrame, reglas: dict, seleccion: list[str]):
    """Construye (X, y) con VarDep 0/1 usando unas reglas de binning dadas."""
    feat = construir_features(datos, reglas)
    mask = feat["VarDep"].isin([0, 1])
    X = feat.loc[mask, seleccion].astype(float)
    y = feat.loc[mask, "VarDep"].astype(int)
    return X, y


def aplicar_accion(accion: Accion, modelo: ModeloBase, datos: pd.DataFrame) -> ModeloBase:
    """Re-ajusta el modelo según el nivel de la acción y devuelve el modelo NUEVO.

    `datos` = lotes recientes con etiquetas maduras (ver `dataset_refit`). `esperar`
    devuelve el mismo modelo sin cambios.
    """
    C = float(modelo.metadata.get("C", 1.0))
    seleccion = modelo.seleccion

    if accion == Accion.RECONSTRUIR:
        # Rehace binning + selección(cols) + pipeline con los datos nuevos.
        return entrenar_modelo_base(datos, seleccion, C,
                                    metadata_extra={"origen": "reconstruir"})

    if accion == Accion.REENTRENAR:
        # Mantiene el binning viejo; re-ajusta imputador+escalador+logit.
        X, y = _matriz_entrenamiento(datos, modelo.reglas, seleccion)
        pipe = make_pipeline(C).fit(X, y)
        return ModeloBase(reglas=modelo.reglas, seleccion=seleccion, pipeline=pipe,
                          metadata={**modelo.metadata, "origen": "reentrenar"})

    if accion == Accion.FINE_TUNE:
        # Reusa binning + imputador + escalador congelados; re-ajusta SOLO el logit.
        X, y = _matriz_entrenamiento(datos, modelo.reglas, seleccion)
        pre = modelo.pipeline[:-1]                       # imputer+scaler ya ajustados
        logit = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)
        logit.fit(pre.transform(X), y)
        pipe = Pipeline([("imputer", modelo.pipeline.named_steps["imputer"]),
                         ("scaler", modelo.pipeline.named_steps["scaler"]),
                         ("logit", logit)])
        return ModeloBase(reglas=modelo.reglas, seleccion=seleccion, pipeline=pipe,
                          metadata={**modelo.metadata, "origen": "fine_tune"})

    return modelo   # esperar → sin cambios


def dataset_refit(pool: pd.DataFrame, config, plan, t_actual: int,
                  ventana: int) -> pd.DataFrame | None:
    """Arma el conjunto de re-entrenamiento: los últimos `ventana` meses YA etiquetados.

    Al mes ``t_actual`` solo hay etiquetas maduras hasta ``t_actual - retardo``. Devuelve
    None si aún no hay ningún mes maduro. Regenera los lotes vía el stream idempotente.
    """
    retardo = config.retardo_periodos
    t_ultimo = t_actual - retardo
    if t_ultimo < 0:
        return None
    ini = max(0, t_ultimo - ventana + 1)
    objetivo = set(range(ini, t_ultimo + 1))
    trozos = []
    for lote in Simulador(pool, config, plan).stream():
        if lote.periodo in objetivo:
            trozos.append(lote.features)
        if lote.periodo >= t_ultimo:
            break
    return pd.concat(trozos, ignore_index=True) if trozos else None
