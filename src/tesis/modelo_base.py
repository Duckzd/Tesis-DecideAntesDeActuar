"""Modelo base (modelo víctima) — Tesis PR-196.

Empaqueta la **cadena completa** del scorecard de credit scoring que el agente
vigilará: binning (árboles) + ingeniería de variables (dummies, capeo) + pipeline
sklearn (imputación → escalado → logística L2). Todo se ajusta **una sola vez sobre
DEV** y se congela en ``models/modelo_base.pkl``.

Diseño clave: **ninguna función depende de variables globales**. La misma cadena que
entrena el modelo se reaplica, sin cambios y **congelada**, a cada lote semanal del
simulador. Congelar el preprocesamiento es requisito del experimento: si se re-ajustara
por lote, absorbería el drift y ocultaría el deterioro.

Portado verbatim desde ``notebooks/fase0_modelo_base/03_Modelamiento.ipynb``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 13579

# Variables a discretizar:  nombre -> columna(s) fuente.
# Las bivariadas dejan que el árbol modele la interacción entre las dos columnas.
VARIABLES: dict[str, list[str]] = {
    # univariadas
    "MAX_DVEN_SCE_6M":                 ["MAX_DVEN_SCE_6M"],
    "PROM_VEN_SCE_6M":                 ["PROM_VEN_SCE_6M"],
    "maySalVen24M269":                 ["maySalVen24M269"],
    "NENT_VEN_SCE_24M":                ["NENT_VEN_SCE_24M"],
    "maySalVenD3M227":                 ["maySalVenD3M227"],
    "califHisTitularY361":             ["califHisTitularY361"],
    "NOPE_XVEN_OP_3M":                 ["NOPE_XVEN_OP_3M"],
    "NOPE_APERT_SCE_OP_3M":            ["NOPE_APERT_SCE_OP_3M"],
    "PROM_NDI_SCE_36M":                ["PROM_NDI_SCE_36M"],
    "NOPE_VENC_OP_3M":                 ["NOPE_VENC_OP_3M"],
    # bivariadas
    "MAX_DVEN_SCE_6M_y_SALDO_PROMEDIO_AHORRO": ["MAX_DVEN_SCE_6M", "SALDO_PROMEDIO_AHORRO"],
    "NOPE_VENC_OP_12_y_INGRESOS":              ["NOPE_VENC_OP_12M", "INGRESOS"],
    "edad_y_INGRESOS":                         ["edad", "INGRESOS"],
    "NOPE_NDI_OP_3MySalTotOpD383":             ["NOPE_NDI_OP_3M", "SalTotOpD383"],
}


# --------------------------------------------------------------------------- #
# Binning con árboles (ajuste en DEV / aplicación en cualquier muestra)
# --------------------------------------------------------------------------- #
def fit_arbol(train: pd.DataFrame, cols: list[str], max_leaf_nodes: int = 5,
              min_frac: float = 0.05, random_state: int = RANDOM_STATE) -> dict:
    """Ajusta un árbol de binning con ``train`` restringido a VarDep 0/1.

    Devuelve la regla: árbol + tasa de malos del bin "missing" (registros con algún
    nulo en ``cols``) + tasa global, para poder aplicarla luego a cualquier lote.
    """
    base = train[train["VarDep"].isin([0, 1])]
    completos = base[cols].notna().all(axis=1)
    d = base.loc[completos]
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_frac,
        random_state=random_state,
    )
    tree.fit(d[cols], d["VarDep"])
    faltantes = base.loc[~completos, "VarDep"]
    p_na = faltantes.mean() if len(faltantes) else base["VarDep"].mean()
    return {"cols": cols, "tree": tree, "p_na": float(p_na),
            "tasa_global": float(base["VarDep"].mean())}


def aplicar_arbol(df: pd.DataFrame, regla: dict) -> tuple[pd.Series, pd.Series]:
    """Aplica una regla ya ajustada. Devuelve (bin, prbm); bin = -1 para nulos."""
    cols = regla["cols"]
    completos = df[cols].notna().all(axis=1)
    bin_ = pd.Series(-1, index=df.index, dtype="int64")
    prbm = pd.Series(regla["p_na"], index=df.index, dtype="float64")
    if completos.any():
        X = df.loc[completos, cols]
        bin_.loc[completos] = regla["tree"].apply(X)
        prbm.loc[completos] = regla["tree"].predict_proba(X)[:, 1]
    prbm = prbm.fillna(regla["tasa_global"])
    return bin_, prbm


def cortes_de(regla: dict) -> list[float]:
    """Umbrales internos aprendidos por el árbol de una regla."""
    t = regla["tree"].tree_
    es_hoja = t.children_left == t.children_right
    return sorted(np.round(t.threshold[~es_hoja], 4).tolist())


def fit_reglas(train: pd.DataFrame, variables: dict[str, list[str]] = VARIABLES) -> dict[str, dict]:
    """Ajusta todas las reglas de binning sobre ``train`` (DEV)."""
    return {nombre: fit_arbol(train, cols) for nombre, cols in variables.items()}


# --------------------------------------------------------------------------- #
# Ingeniería de variables (determinista + binning) — pura, sin globales
# --------------------------------------------------------------------------- #
def construir_features(df: pd.DataFrame, reglas: dict[str, dict]) -> pd.DataFrame:
    """Construye TODAS las variables del modelo sobre ``df`` usando ``reglas``.

    Binning (con las reglas ajustadas en DEV) + dummies deterministas + capeo de
    atípicos con umbrales fijos. No re-ajusta nada: apto para lotes de producción.
    """
    df = df.copy()

    # --- Binning con árboles (reglas ajustadas en DEV) ---
    for nombre, regla in reglas.items():
        b, prbm = aplicar_arbol(df, regla)
        df[f"bin_{nombre}"] = b
        df[f"prbm_{nombre}"] = prbm
        df[f"prbb_{nombre}"] = 1 - prbm

    # --- Variables dummy (deterministicas, fila a fila) ---
    df["d_numOpsVencidas3M102_cast"] = np.where(df["numOpsVencidas3M102"] > 0, 1, 0)
    df["d_numOpsVencidas3M102_prem"] = np.where(df["numOpsVencidas3M102"] == 0, 1, 0)
    df["d_numOpsVencidas101_cast"] = np.where(df["numOpsVencidas101"] > 0, 1, 0)
    df["d_numOpsVencidas101_prem"] = np.where(df["numOpsVencidas101"] == 0, 1, 0)
    df["d_numOpeCarteraCastigadaTitular376_cast"] = np.where(
        df["numOpeCarteraCastigadaTitular376"] > 0, 1, 0)
    df["d_numOpeCarteraCastigadaTitular376_prem"] = np.where(
        df["numOpeCarteraCastigadaTitular376"] == 0, 1, 0)
    df["d_peorNivelRiesgoValorOpBanCooComD415_prem"] = np.where(
        df["peorNivelRiesgoValorOpBanCooComD415"] <= 1, 1, 0)
    df["d_peorNivelRiesgoValorOpBanCooComD415_cast"] = np.where(
        df["peorNivelRiesgoValorOpBanCooComD415"] > 1, 1, 0)
    df["d_NOPE_NDI_OP_3M_cast"] = np.where(df["NOPE_NDI_OP_3M"] > 0, 1, 0)

    # --- Capeo de atipicos (umbrales fijos, derivados en DEV) ---
    df["califHisTitularV360_c"] = np.where(df["califHisTitularV360"] > 23, 23, df["califHisTitularV360"])
    df["MAX_DVEN_SF_OP_12M_c"] = np.where(df["MAX_DVEN_SF_OP_12M"] > 270, 270, df["MAX_DVEN_SF_OP_12M"])
    df["numOpsVig092_c"] = np.where(df["numOpsVig092"] > 59, 59, df["numOpsVig092"])
    df["MAX_DVEN_SCE_36M_c"] = np.where(df["MAX_DVEN_SCE_36M"] > 1583.54, 1583.54, df["MAX_DVEN_SCE_36M"])
    df["MAX_DVEN_SCE_6M_c"] = np.where(df["MAX_DVEN_SCE_6M"] > 360, 360, df["MAX_DVEN_SCE_6M"])
    df["SALDO_PROMEDIO_AHORRO_c"] = np.where(
        df["SALDO_PROMEDIO_AHORRO"] > 10500.9854, 10500.9854, df["SALDO_PROMEDIO_AHORRO"])
    df["NOPE_APERT_SCE_OP_24M_c"] = np.where(
        df["NOPE_APERT_SCE_OP_24M"] > 25, 25, df["NOPE_APERT_SCE_OP_24M"])
    df["NOPE_NDI_OP_24M_c"] = np.where(df["NOPE_NDI_OP_24M"] > 2, 2, df["NOPE_NDI_OP_24M"])
    df["SalTotOpD383_c"] = np.where(df["SalTotOpD383"] > 141419.8156, 2, df["SalTotOpD383"])
    df["NOPE_APERT_SF_OP_6M_c"] = np.where(df["NOPE_APERT_SF_OP_6M"] > 7, 7, df["NOPE_APERT_SF_OP_6M"])
    df["NOPE_APERT_SCE_OP_12M_c"] = np.where(
        df["NOPE_APERT_SCE_OP_12M"] > 15, 15, df["NOPE_APERT_SCE_OP_12M"])
    df["salOpVen014_c"] = np.where(df["salOpVen014"] > 8723.8756, 8723.8756, df["salOpVen014"])
    df["DEUDA_TOTAL_SCE_12M_c"] = np.where(
        df["DEUDA_TOTAL_SCE_12M"] > 1745712.081, 1745712.081, df["DEUDA_TOTAL_SCE_12M"])
    df["NOPE_REFIN_OP_12M_c"] = np.where(df["NOPE_REFIN_OP_12M"] >= 2, 2, df["NOPE_REFIN_OP_12M"])
    df["numOpsVencidas101_c"] = np.where(df["numOpsVencidas101"] >= 2, 2, df["numOpsVencidas101"])
    df["numOpsVencidas3M102_c"] = np.where(df["numOpsVencidas3M102"] >= 3, 3, df["numOpsVencidas3M102"])
    df["maySalVen24M269_c"] = np.where(
        df["maySalVen24M269"] >= 1881.4544, 1881.4544, df["maySalVen24M269"])

    return df.copy()   # defragmentar tras las inserciones


def make_pipeline(C: float, random_state: int = RANDOM_STATE) -> Pipeline:
    """Pipeline del scorecard: imputación (mediana) → escalado → logística L2."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=1000, random_state=random_state)),
    ])


# --------------------------------------------------------------------------- #
# Contenedor del modelo base congelado
# --------------------------------------------------------------------------- #
@dataclass
class ModeloBase:
    """Modelo víctima congelado: reglas de binning + selección + pipeline ajustado."""

    reglas: dict[str, dict]
    seleccion: list[str]
    pipeline: Pipeline
    metadata: dict = field(default_factory=dict)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la cadena de preparación (binning + dummies + capeo)."""
        return construir_features(df, self.reglas)

    def _matriz(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)[self.seleccion].astype(float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(bueno), P(malo) por fila, desde datos crudos."""
        return self.pipeline.predict_proba(self._matriz(df))

    def score_malo(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad de incumplimiento P(malo) por fila, desde datos crudos."""
        return self.predict_proba(df)[:, 1]

    def guardar(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def cargar(path: str | Path) -> "ModeloBase":
        with open(path, "rb") as f:
            return pickle.load(f)


def entrenar_modelo_base(dev: pd.DataFrame, seleccion: list[str], C: float,
                         variables: dict[str, list[str]] = VARIABLES,
                         random_state: int = RANDOM_STATE,
                         metadata_extra: dict | None = None) -> ModeloBase:
    """Ajusta el modelo base COMPLETO sobre DEV y lo devuelve empaquetado.

    1. Ajusta las reglas de binning en DEV.
    2. Construye las features y entrena el pipeline sobre VarDep 0/1.
    3. Empaqueta todo (reglas + selección + pipeline + metadata) en ``ModeloBase``.
    """
    import sklearn

    reglas = fit_reglas(dev, variables)
    feat = construir_features(dev, reglas)
    mask = feat["VarDep"].isin([0, 1])
    X = feat.loc[mask, seleccion].astype(float)
    y = feat.loc[mask, "VarDep"].astype(int)

    pipeline = make_pipeline(C, random_state)
    pipeline.fit(X, y)

    metadata = {
        "modelo": "scorecard credit scoring (logística L2)",
        "rol": "modelo víctima — Tesis PR-196",
        "n_features": len(seleccion),
        "C": float(C),
        "random_state": random_state,
        "n_dev": int(mask.sum()),
        "tasa_malos_dev": float(y.mean()),
        "congelado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versiones": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    return ModeloBase(reglas=reglas, seleccion=list(seleccion),
                      pipeline=pipeline, metadata=metadata)
