# Tesis PR-196 — Políticas de selección y delegación para intervenir modelos degradados

Comparación de **políticas de decisión bajo incertidumbre** que eligen qué acción
correctiva aplicar cuando un modelo de ML se degrada, observando solo **señales
indirectas** (drift, distribución de scores, confianza, latencia, volumen, historial)
y **sin las etiquetas reales**, que llegan con retardo.

El caso de aplicación son modelos de **credit scoring** (los "modelos víctima"),
**congelados**. Sobre ellos se simula un flujo **mensual** con deterioro controlado e
inyectado, del que conocemos el *ground truth* (cuándo y cómo empieza el daño) para medir
si cada política decide bien. Dos modelos: uno de buró (principal) y uno de LendingClub
(generalización, con 139 meses reales).

## Estructura

```
data/                       Datos (NO versionados, ver data/README.md)
  buro/                       modelo principal (buró de crédito)
  lendingclub/                segundo modelo víctima (generalización)
models/                     Artefactos congelados (modelo_base.pkl, modelo_lendingclub.pkl)
notebooks/
  modelo_buro/                creación del modelo principal (01 → 02 → 03)
  modelo_lendingclub/         creación del segundo modelo (borrador + original)
  lab/                        cuadernos de exploración (simulador, señales)
src/tesis/
  modelo_base.py              modelo víctima congelado (preprocess + pipeline)
  simulador/                  lotes mensuales + inyección de drift
  senales/                    señales indirectas (PSI, distribución de score, confianza…)
  politicas/                  reglas · score multi-señal · router LLM · bandit
  acciones/                   acciones correctivas (fine-tune / reentrenar / reconstruir)
  experimentos/               harness, logging y métricas
docs/                       presentación y documentos
experiments/                resultados y logs de corridas
frontend/                   visualización
```

## Entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # versiones pineadas (reproducible)
```

- `.venv/` es local y **no** se versiona (no es portable).
- `requirements.in` = lista curada a mano; `requirements.txt` = versiones exactas
  (`pip freeze`) → **este es el artefacto de reproducibilidad**.

## Fases

0. **Modelo base** — congelar modelo + preprocesamiento + dataset de referencia.
1. **Simulador mensual** — lotes + 3 tipos de deterioro (covariate / concept / mezcla) con ground truth y retardo de etiquetas.
2. **Señales** — capa de señales indirectas.
3. **Políticas** — reglas (baseline), score de riesgo multi-señal, router LLM, bandit (opcional).
4. **Acciones** — acciones correctivas y sus efectos.
5. **Experimentos** — logging, métricas, comparación justa de políticas.
6. **Frontend** — visualización.
