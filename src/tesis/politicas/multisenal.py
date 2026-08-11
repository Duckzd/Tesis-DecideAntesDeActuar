"""Política multi-señal: score de riesgo + reglas de desambiguación — Fase 3.

Mejora sobre el baseline: no reacciona en automático al drift de distribución. Combina
dos evidencias y las contrasta:

- **r_dist** — drift de distribución (PSI del score). *Alerta temprana, pero ambigua*:
  se mueve tanto con covariate (benigno) como con un problema real.
- **r_perf** — caída del **desempeño tardío** (AUC que ya maduró). *La señal verdadera,
  pero llega con retardo.*

Reglas de desambiguación (el "pero" que la regla no puede):
1. Si el desempeño tardío **confirma daño** → actúa (reentrenar/reconstruir por severidad).
2. Si la distribución se movió **pero** el desempeño tardío confirma que el modelo está
   **bien** → es covariate → **no actúa** (evita la falsa alarma del baseline).
3. Zona ambigua (drift sin confirmación aún) → actúa solo si es fuerte y persistente.

Resultado esperado: menos falsas alarmas que el baseline en covariate, y **detecta el
concept drift** (tarde, cuando maduran sus etiquetas) en vez de estar ciego para siempre.
"""

from __future__ import annotations

from .base import Accion, Contexto, Decision, registrar


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@registrar("multisenal")
class PoliticaMultiSenal:
    """Score de riesgo continuo (0–1) con reglas de desambiguación."""

    def __init__(self, auc_baseline: float = 0.885, margen_auc: float = 0.05,
                 umbral_dist: float = 0.15, umbral_actuar: float = 0.6,
                 umbral_reconstruir: float = 0.85, persistencia: int = 2, cooldown: int = 3,
                 retardo: int = 12):
        self.auc_baseline = auc_baseline
        self.margen_auc = margen_auc
        self.umbral_dist = umbral_dist
        self.umbral_actuar = umbral_actuar
        self.umbral_reconstruir = umbral_reconstruir
        self.persistencia = max(1, persistencia)
        self.cooldown = cooldown
        self.retardo = retardo   # meses hasta que madura data puntuada por el modelo NUEVO

    def _persiste_dist(self, contexto: Contexto, psi_actual: float) -> bool:
        if self.persistencia == 1:
            return psi_actual >= self.umbral_dist
        previos = [r.psi_score for r in contexto.historia[-(self.persistencia - 1):]]
        return psi_actual >= self.umbral_dist and len(previos) == self.persistencia - 1 \
            and all(p >= self.umbral_dist for p in previos)

    def decidir(self, reporte, contexto: Contexto) -> Decision:
        if reporte.meses_desde_intervencion is not None \
                and reporte.meses_desde_intervencion < self.cooldown:
            return Decision(Accion.ESPERAR, f"cooldown ({reporte.meses_desde_intervencion} < {self.cooldown})")

        r_dist = _clip01(reporte.psi_score / self.umbral_dist)

        # Confirmación consciente del retardo: el AUC tardío del mes t mide la cohorte
        # t-retardo. Si intervine DESPUÉS de que esa cohorte fue puntuada, ese AUC refleja
        # el modelo VIEJO (pre-arreglo) → es evidencia obsoleta, no confirma daño actual.
        # Solo cuenta si nunca intervine o si ya pasó el retardo desde la última acción.
        meses = reporte.meses_desde_intervencion
        confirmado = meses is None or meses >= self.retardo
        tiene_perf = reporte.etiquetas_disponibles and reporte.auc_revelado is not None and confirmado
        r_perf = _clip01((self.auc_baseline - reporte.auc_revelado) / self.margen_auc) if tiene_perf else None

        # 1. Daño confirmado por el desempeño tardío → actuar por severidad
        if r_perf is not None and r_perf >= 0.5:
            riesgo = 0.5 + 0.5 * r_perf
            accion = Accion.RECONSTRUIR if riesgo >= self.umbral_reconstruir else Accion.REENTRENAR
            return Decision(accion, f"daño confirmado: AUC tardío {reporte.auc_revelado:.3f} "
                            f"< baseline {self.auc_baseline}", score_riesgo=round(riesgo, 3))

        # 2. Distribución movida PERO desempeño tardío OK → covariate, no actuar
        if tiene_perf and r_perf <= 0.2 and r_dist >= 0.6:
            return Decision(Accion.ESPERAR, f"PSI alto ({reporte.psi_score:.3f}) pero AUC tardío "
                            f"OK ({reporte.auc_revelado:.3f}) → covariate", score_riesgo=round(0.15 * r_dist, 3))

        # 3. Zona ambigua → actuar solo si es fuerte y persistente
        riesgo = 0.6 * r_dist + (0.4 * r_perf if r_perf is not None else 0.0)
        if riesgo >= self.umbral_actuar and self._persiste_dist(contexto, reporte.psi_score):
            return Decision(Accion.REENTRENAR, f"riesgo {riesgo:.2f} (drift persistente, sin "
                            f"confirmación aún)", score_riesgo=round(riesgo, 3))
        return Decision(Accion.ESPERAR, f"riesgo {riesgo:.2f} bajo el umbral",
                        score_riesgo=round(riesgo, 3))
