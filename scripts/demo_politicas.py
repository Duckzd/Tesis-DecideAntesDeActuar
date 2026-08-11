"""Demo: cómo trabajan las políticas de decisión — Tesis PR-196.

Corre desde la raíz del proyecto:
    .venv/bin/python scripts/demo_politicas.py            # comparación (LLM simulado, gratis)
    .venv/bin/python scripts/demo_politicas.py --real     # + traza REAL del LLM (usa tu API key)

Parte 1: compara reglas vs multi-señal vs LLM sobre DOS escenarios (mismo stream, CRN):
  - covariate shift  -> la trampa de la FALSA ALARMA (el modelo está bien pero el PSI sube).
  - concept drift    -> la trampa del PUNTO CIEGO (el modelo se rompe pero el PSI no se mueve).
Parte 2 (--real): corre el LLM agéntico REAL en 3 meses ilustrativos y muestra su traza
  (qué herramientas usó, su razonamiento y la decisión). Barato: ~3 decisiones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from tesis.modelo_base import ModeloBase
from tesis.simulador.poblacion import cargar_poblacion, COHORTE_OOT
from tesis.simulador.simulador import ConfigSimulacion, plan_sano, agregar_episodio, Simulador
from tesis.senales.monitor import MonitorSenales, Referencia
from tesis.politicas import Contexto, crear, ClienteSimulado, PoliticaLLM
from tesis.experimentos.banco import comparar

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def cargar_todo():
    """Modelo víctima, pool (OOT) para simular, y referencia PSI (train/DEV)."""
    modelo = ModeloBase.cargar("models/modelo_base.pkl")
    pool = cargar_poblacion(cohorte=COHORTE_OOT)                 # futuro simulado
    dev = cargar_poblacion(cohorte=None)                        # todas las cohortes...
    dev = dev[dev["FECHA_CORTE"] < COHORTE_OOT]                 # ...menos OOT = train/DEV
    referencia = Referencia.construir(dev, modelo)
    return modelo, pool, referencia


def politicas_demo():
    """Las tres políticas. El LLM en modo SIMULADO (sin API, para la comparación gratis)."""
    return {
        "reglas": crear("reglas"),
        "multisenal": crear("multisenal"),
        "llm(sim)": PoliticaLLM(cliente=ClienteSimulado()),
    }


def escenario_covariate():
    cfg = ConfigSimulacion(n_periodos=18, tam_lote=2000, semilla=13579)
    plan = agregar_episodio(plan_sano(18), "covariate_shift", inicio=4, fin=11,
                            intensidad=1.0, forma="rampa")
    return cfg, plan


def escenario_concept():
    cfg = ConfigSimulacion(n_periodos=24, tam_lote=2000, semilla=13579)
    plan = agregar_episodio(plan_sano(24), "concept_drift", inicio=2, fin=11,
                            intensidad=1.0, forma="escalon")
    return cfg, plan


def parte1_comparacion(modelo, pool, referencia):
    print("\n" + "=" * 78)
    print(" PARTE 1 — COMPARACIÓN DE POLÍTICAS (mismo stream, common random numbers)")
    print("=" * 78)

    for titulo, (cfg, plan) in [
        ("ESCENARIO A · covariate shift  (trampa: FALSA ALARMA)", escenario_covariate()),
        ("ESCENARIO B · concept drift    (trampa: PUNTO CIEGO)", escenario_concept()),
    ]:
        tabla, resumen = comparar(politicas_demo(), pool, cfg, plan, modelo, referencia)
        print(f"\n--- {titulo} ---")
        print(tabla.to_string(index=False))
        print("\n  Conteo de acciones por política:")
        print(resumen.to_string())


def _recolectar_reportes(modelo, pool, referencia, cfg, plan):
    """Corre el monitor y devuelve [(reporte, contexto)] mes a mes (sin decidir)."""
    mon = MonitorSenales(referencia)
    pares = []
    for lote in Simulador(pool, cfg, plan).stream():
        rep = mon.observar(lote, modelo)
        pares.append((rep, Contexto(historia=mon.reportes[:-1])))
    return pares


def parte2_llm_real(modelo, pool, referencia):
    print("\n" + "=" * 78)
    print(" PARTE 2 — TRAZA DEL LLM AGÉNTICO REAL (usa tu API key)")
    print("=" * 78)

    pol = PoliticaLLM(modelo="claude-opus-5")     # cliente real (lee .env)

    # 3 meses ilustrativos del escenario concept, que cuentan la historia completa:
    #   - sano temprano                      -> debe esperar
    #   - concept OCURRIENDO pero invisible  -> aún no puede saberlo (como todos)
    #   - daño ya REVELADO por el AUC tardío -> debe actuar
    cfg, plan = escenario_concept()
    pares = _recolectar_reportes(modelo, pool, referencia, cfg, plan)
    invisible = next((i for i, (r, _) in enumerate(pares)
                      if plan[r.periodo].tipo == "concept_drift" and r.auc_revelado is None), 8)
    danado = next((i for i, (r, _) in enumerate(pares)
                   if r.auc_revelado is not None and r.auc_revelado < 0.5), 14)
    elegidos = [1, invisible, danado]

    for i in elegidos:
        rep, ctx = pares[i]
        auc = f"{rep.auc_revelado:.3f}" if rep.auc_revelado is not None else "None (aún no madura)"
        print(f"\n>>> Mes {rep.periodo} | tipo real={plan[rep.periodo].tipo} | "
              f"psi={rep.psi_score:.3f} | auc_tardío={auc}")
        dec = pol.decidir(rep, ctx)
        tz = pol.trazas[-1]
        print(f"    herramientas usadas : {[t['herramienta'] for t in tz['herramientas_usadas']]}")
        for r in tz["razonamiento"]:
            print(f"    razonamiento        : {r}")
        print(f"    DECISIÓN            : {dec.accion.value}  —  {dec.razon}")


def main():
    real = "--real" in sys.argv
    modelo, pool, referencia = cargar_todo()
    parte1_comparacion(modelo, pool, referencia)
    if real:
        parte2_llm_real(modelo, pool, referencia)
    else:
        print("\n(Para ver una traza REAL del LLM agéntico: "
              ".venv/bin/python scripts/demo_politicas.py --real)")


if __name__ == "__main__":
    main()
