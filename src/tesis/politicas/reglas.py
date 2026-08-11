"""Política baseline: reglas / umbral — Tesis PR-196 (Fase 3).

El punto de referencia "ingenuo" del monitoreo clásico: *si una señal de distribución
cruza un umbral, interviene*. Es deliberadamente simple, pero **justo**: incluye los
elementos realistas de una regla de alerta de producción —

- **escalado**: umbral medio → `reentrenar`, umbral alto → `reconstruir`;
- **persistencia (histéresis)**: solo actúa si la señal supera el umbral `persistencia`
  meses seguidos → no reacciona a un pico de ruido puntual;
- **cooldown**: tras intervenir, espera `cooldown` meses antes de volver a actuar.

La tesis mostrará que, aun bien afinado, este baseline cae en dos trampas: **falsa
alarma** con covariate shift (dispara aunque el modelo esté bien) y **punto ciego** con
concept drift (nunca dispara porque la señal de distribución no se mueve).
"""

from __future__ import annotations

from .base import Accion, Contexto, Decision, registrar


@registrar("reglas")
class PoliticaReglas:
    """Umbral sobre una señal, con escalado + persistencia + cooldown."""

    def __init__(self, senal: str = "psi_score", umbral_reentrenar: float = 0.10,
                 umbral_reconstruir: float = 0.25, persistencia: int = 1,
                 cooldown: int = 3):
        self.senal = senal
        self.umbral_reentrenar = umbral_reentrenar
        self.umbral_reconstruir = umbral_reconstruir
        self.persistencia = max(1, persistencia)
        self.cooldown = cooldown

    def _persiste(self, contexto: Contexto, valor_actual: float, umbral: float) -> bool:
        """¿La señal superó el umbral en los últimos `persistencia` meses (incl. el actual)?"""
        if self.persistencia == 1:
            return valor_actual >= umbral
        previos = [getattr(r, self.senal) for r in contexto.historia[-(self.persistencia - 1):]]
        return valor_actual >= umbral and all(v >= umbral for v in previos) \
            and len(previos) == self.persistencia - 1

    def decidir(self, reporte, contexto: Contexto) -> Decision:
        # Cooldown: si intervino hace poco, esperar
        m = reporte.meses_desde_intervencion
        if m is not None and m < self.cooldown:
            return Decision(Accion.ESPERAR, f"cooldown ({m} < {self.cooldown} meses)")

        valor = getattr(reporte, self.senal)

        if self._persiste(contexto, valor, self.umbral_reconstruir):
            return Decision(Accion.RECONSTRUIR,
                            f"{self.senal}={valor:.3f} ≥ {self.umbral_reconstruir} (alto)",
                            score_riesgo=valor)
        if self._persiste(contexto, valor, self.umbral_reentrenar):
            return Decision(Accion.REENTRENAR,
                            f"{self.senal}={valor:.3f} ≥ {self.umbral_reentrenar} (medio)",
                            score_riesgo=valor)
        return Decision(Accion.ESPERAR,
                        f"{self.senal}={valor:.3f} < {self.umbral_reentrenar} (estable)",
                        score_riesgo=valor)
