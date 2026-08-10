"""Capa de señales indirectas — Tesis PR-196 (Fase 2).

Lo único que el agente PUEDE observar mes a mes, **sin las etiquetas reales**. Convierte
cada lote mensual en un `ReporteSenales` estructurado — la entrada de las políticas
(Fase 3) y del router LLM.

Señales de distribución (PSI del score y de features) y de modelo (confianza) están
disponibles **el mismo mes**. Las señales de desempeño (bad-rate/AUC/KS) solo aparecen
cuando las etiquetas maduran (retardo de 12 meses): son tardías y parciales.

`MonitorSenales` es **con estado**: guarda la referencia (distribución "normal"), el
historial de scores (para emparejarlos con las etiquetas que llegan tarde) y la última
intervención. El agente nunca ve la `VarDep` del mes actual — solo este reporte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Panel de variables crudas a monitorear (interpretables como drivers de riesgo).
VARS_MONITOR = [
    "MAX_DVEN_SCE_6M", "PROM_VEN_SCE_6M", "DEUDA_TOTAL_SCE_12M", "INGRESOS",
    "edad", "SALDO_PROMEDIO_AHORRO", "numOpsVencidas101", "NOPE_VENC_OP_3M",
]


def psi(base: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre una distribución base y una actual."""
    base = np.asarray(base, dtype=float)
    base = base[~np.isnan(base)]
    actual = np.asarray(actual, dtype=float)
    actual = actual[~np.isnan(actual)]
    if len(base) == 0 or len(actual) == 0:
        return float("nan")
    cortes = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:                       # variable casi constante
        return 0.0
    def prop(x):
        c = pd.Series(pd.cut(x, cortes, include_lowest=True)).value_counts().sort_index()
        return (c / c.sum()).clip(lower=1e-6).to_numpy()
    e, a = prop(base), prop(actual)
    return float(((a - e) * np.log(a / e)).sum())


def ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estadístico KS entre las distribuciones de score de buenos y malos."""
    y_true = np.asarray(y_true)
    orden = np.argsort(y_score)
    b = np.cumsum(y_true[orden] == 0) / max((y_true == 0).sum(), 1)
    m = np.cumsum(y_true[orden] == 1) / max((y_true == 1).sum(), 1)
    return float(np.max(np.abs(b - m)))


@dataclass
class Referencia:
    """Distribución esperada del modelo (el 'comportamiento normal') para medir drift.

    Estándar de monitoreo de scorecards: la referencia es la muestra de **desarrollo
    (DEV/train)**, no la población de ejecución. Así el PSI mide la desviación respecto
    a lo que el modelo aprendió a esperar, y un mes sano ya refleja la deriva real
    train↔ejecución (no un cero artificial).
    """
    score: np.ndarray
    features: dict[str, np.ndarray]

    @staticmethod
    def construir(df_ref: pd.DataFrame, modelo, vars_monitor: list[str] = VARS_MONITOR,
                  n: int = 5000, semilla: int = 13579) -> "Referencia":
        """Construye la referencia desde ``df_ref`` (debe ser la muestra DEV/train)."""
        rng = np.random.default_rng(semilla)
        idx = rng.integers(0, len(df_ref), size=min(n, len(df_ref)))
        ref = df_ref.iloc[idx]
        return Referencia(
            score=modelo.score_malo(ref),
            features={v: ref[v].to_numpy(dtype=float) for v in vars_monitor if v in ref.columns},
        )


@dataclass
class ReporteSenales:
    """Señales observables de un período (lo que ve el agente)."""
    periodo: int
    fecha: pd.Timestamp
    n: int                              # volumen
    score_medio: float
    score_std: float
    psi_score: float                    # vs referencia (train) — deriva acumulada
    psi_score_vs_previo: float | None    # vs mes anterior — cambio reciente
    psi_feat_prom: float
    psi_feat_max: float
    psi_por_feature: dict[str, float]
    confianza_media: float              # 2*|score-0.5| promedio (1 = muy seguro, 0 = a la mitad)
    frac_incierto: float                # fracción de scores en [0.45, 0.55]
    tasa_nulos: float                   # nulos promedio en las variables monitoreadas
    meses_desde_intervencion: int | None
    # Señales tardías (None hasta que maduran las etiquetas de hace `retardo` meses)
    etiquetas_disponibles: bool = False
    periodo_revelado: int | None = None
    bad_rate_revelado: float | None = None
    auc_revelado: float | None = None
    ks_revelado: float | None = None


class MonitorSenales:
    """Computa señales mes a mes a partir del stream del simulador."""

    def __init__(self, referencia: Referencia, vars_monitor: list[str] = VARS_MONITOR):
        self.ref = referencia
        self.vars_monitor = [v for v in vars_monitor if v in referencia.features]
        self._scores_hist: dict[int, np.ndarray] = {}   # periodo -> scores que el modelo produjo
        self.reportes: list[ReporteSenales] = []
        self._ultima_intervencion: int | None = None

    def registrar_intervencion(self, periodo: int) -> None:
        """La política avisa que intervino en este período (reinicia el contador)."""
        self._ultima_intervencion = periodo

    def observar(self, lote, modelo) -> ReporteSenales:
        """Calcula el `ReporteSenales` del período del lote (sin mirar su VarDep)."""
        df = lote.features
        score = modelo.score_malo(df)
        previo = self._scores_hist.get(lote.periodo - 1)   # antes de guardar el actual
        self._scores_hist[lote.periodo] = score

        psi_feat = {v: psi(self.ref.features[v], df[v].to_numpy(dtype=float))
                    for v in self.vars_monitor}
        vals = [x for x in psi_feat.values() if not np.isnan(x)]
        nulos = float(np.mean([df[v].isna().mean() for v in self.vars_monitor])) if self.vars_monitor else 0.0
        meses = (lote.periodo - self._ultima_intervencion
                 if self._ultima_intervencion is not None else None)

        rep = ReporteSenales(
            periodo=lote.periodo, fecha=lote.fecha, n=len(df),
            score_medio=float(score.mean()), score_std=float(score.std()),
            psi_score=psi(self.ref.score, score),
            psi_score_vs_previo=(psi(previo, score) if previo is not None else None),
            psi_feat_prom=float(np.mean(vals)) if vals else float("nan"),
            psi_feat_max=float(np.max(vals)) if vals else float("nan"),
            psi_por_feature={k: round(v, 4) for k, v in psi_feat.items()},
            confianza_media=float(np.mean(2 * np.abs(score - 0.5))),
            frac_incierto=float(np.mean((score >= 0.45) & (score <= 0.55))),
            tasa_nulos=nulos,
            meses_desde_intervencion=meses,
        )

        # Señales tardías: si llegaron las etiquetas de la cohorte t-retardo, medir desempeño
        if lote.etiquetas_reveladas is not None and lote.periodo_revelado in self._scores_hist:
            y = lote.etiquetas_reveladas.to_numpy()
            s = self._scores_hist[lote.periodo_revelado]
            rep.etiquetas_disponibles = True
            rep.periodo_revelado = lote.periodo_revelado
            rep.bad_rate_revelado = float(y.mean())
            if len(np.unique(y)) == 2:
                rep.auc_revelado = float(roc_auc_score(y, s))
                rep.ks_revelado = ks(y, s)

        self.reportes.append(rep)
        return rep

    def historia(self) -> pd.DataFrame:
        """Todas las señales acumuladas como DataFrame (una fila por período)."""
        return pd.DataFrame([{k: v for k, v in vars(r).items() if k != "psi_por_feature"}
                             for r in self.reportes])
