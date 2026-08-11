"""Política agéntica con un LLM — el brazo estrella de la tesis (Fase 3).

"Elegir bien antes de actuar", literal: en vez de aplicar un umbral fijo, el LLM
**investiga** con herramientas antes de decidir. Recibe el `ReporteSenales` del mes y
puede llamar funciones para mirar más de cerca —

- `psi_por_variable()`   — qué variables se movieron (drift focalizado vs difuso);
- `tendencia(senal, n)`  — la evolución de una señal en los últimos meses;
- `historial_intervenciones()` — cuándo intervino y cuántas veces;
- `desempeno_tardio()`   — el AUC/bad-rate que ya maduró (la verdad, pero con retardo).

…y solo entonces emite su decisión llamando a `registrar_decision(accion, razon)`.

Es agéntico de verdad (bucle de *tool use*/function-calling), no un router de un tiro:
el LLM decide **qué** mirar. Eso es el aporte de IA de la tesis, y se documenta abierto
(modelo, prompts, herramientas, y las trazas de cada decisión quedan en `self.trazas`).

Reproducibilidad: los modelos actuales de Anthropic ya no aceptan `temperature`, así que
el determinismo no está garantizado. La tesis lo maneja midiendo la **variabilidad de la
decisión sobre N corridas** con la misma semilla (Fase 5), no forzando temperatura 0.

Se prueba de punta a punta **sin API key** con `ClienteSimulado`; para las corridas reales
se conecta `ClienteAnthropic` (ver `_cargar_api_key` para dónde pegar la key).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .base import Accion, Contexto, Decision, registrar

# --------------------------------------------------------------------------- #
# API key: dónde pegarla
# --------------------------------------------------------------------------- #
# La key se lee de la variable de entorno ANTHROPIC_API_KEY. La forma más cómoda es
# crear un archivo `.env` en la raíz del proyecto (ya está en .gitignore → NO se sube a
# GitHub) con una línea:
#
#     ANTHROPIC_API_KEY=sk-ant-...tu-key-aqui...
#
# Hay una plantilla lista en `.env.example`: cópiala a `.env` y pega tu key ahí.
# `_cargar_api_key()` lee ese `.env` automáticamente antes de crear el cliente real.


def _raiz_proyecto() -> Path:
    """Sube desde este archivo hasta la raíz del proyecto (donde vive `.env`)."""
    aqui = Path(__file__).resolve()
    for padre in aqui.parents:
        if (padre / ".gitignore").exists() or (padre / "requirements.in").exists():
            return padre
    return aqui.parents[3]  # src/tesis/politicas/llm.py → raíz


def _cargar_api_key() -> str | None:
    """Devuelve la API key (de `.env` o del entorno); None si no está configurada."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = _raiz_proyecto() / ".env"
    if env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, _, valor = linea.partition("=")
            if clave.strip() == "ANTHROPIC_API_KEY":
                valor = valor.strip().strip('"').strip("'")
                if valor and not valor.startswith("pega-tu-key"):
                    os.environ["ANTHROPIC_API_KEY"] = valor
                    return valor
    return None


# --------------------------------------------------------------------------- #
# Herramientas expuestas al LLM (esquema JSON) + su ejecución
# --------------------------------------------------------------------------- #
_ACCIONES = [a.value for a in Accion]
_SENALES = ["psi_score", "psi_score_vs_previo", "psi_feat_max", "score_medio",
            "confianza_media", "auc_revelado", "bad_rate_revelado"]

HERRAMIENTAS_INVESTIGACION = [
    {
        "name": "psi_por_variable",
        "description": "Devuelve el PSI de cada variable monitoreada este mes, ordenado de "
                       "mayor a menor. Útil para distinguir drift FOCALIZADO (una o dos "
                       "variables) de uno DIFUSO (muchas a la vez).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "tendencia",
        "description": "Serie de los últimos meses de una señal, para ver si un valor es un "
                       "pico puntual o una tendencia sostenida.",
        "input_schema": {
            "type": "object",
            "properties": {
                "senal": {"type": "string", "enum": _SENALES,
                          "description": "Qué señal seguir en el tiempo."},
                "meses": {"type": "integer", "description": "Cuántos meses hacia atrás (1-24)."},
            },
            "required": ["senal"],
            "additionalProperties": False,
        },
    },
    {
        "name": "historial_intervenciones",
        "description": "Cuántos meses pasaron desde la última intervención y cuántas veces se "
                       "intervino antes. Sirve para no reintervenir en frío (cooldown) ni "
                       "quedar oscilando.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "desempeno_tardio",
        "description": "El desempeño REAL del modelo (AUC, bad-rate, KS) de la cohorte cuyas "
                       "etiquetas ya maduraron. Es la verdad sobre si el modelo sirve, pero "
                       "llega con retardo (12 meses). None si aún no hay etiquetas.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

HERRAMIENTA_DECIDIR = {
    "name": "registrar_decision",
    "description": "Emite la decisión final. Llama esta herramienta UNA sola vez, cuando ya "
                   "investigaste lo necesario.",
    "input_schema": {
        "type": "object",
        "properties": {
            "accion": {"type": "string", "enum": _ACCIONES,
                       "description": "esperar | fine_tune | reentrenar | reconstruir."},
            "razon": {"type": "string",
                      "description": "Justificación breve basada en la evidencia observada."},
        },
        "required": ["accion", "razon"],
        "additionalProperties": False,
    },
}

TODAS_LAS_HERRAMIENTAS = HERRAMIENTAS_INVESTIGACION + [HERRAMIENTA_DECIDIR]


def _contar_intervenciones(contexto: Contexto) -> int:
    """Cuántas veces se intervino (meses_desde_intervencion == 0) en la historia."""
    return sum(1 for r in contexto.historia
               if getattr(r, "meses_desde_intervencion", None) == 0)


def ejecutar_herramienta(nombre: str, args: dict, reporte, contexto: Contexto) -> dict:
    """Ejecuta una herramienta de investigación sobre el reporte + la historia."""
    if nombre == "psi_por_variable":
        orden = sorted(reporte.psi_por_feature.items(), key=lambda kv: kv[1], reverse=True)
        return {"psi_por_variable": {k: v for k, v in orden},
                "psi_feat_max": round(reporte.psi_feat_max, 4),
                "psi_feat_prom": round(reporte.psi_feat_prom, 4)}

    if nombre == "tendencia":
        senal = args.get("senal", "psi_score")
        meses = int(args.get("meses", 6))
        serie = [getattr(r, senal, None) for r in contexto.historia] + [getattr(reporte, senal, None)]
        serie = serie[-max(1, meses):]
        return {"senal": senal,
                "valores": [round(v, 4) if isinstance(v, float) else v for v in serie]}

    if nombre == "historial_intervenciones":
        return {"meses_desde_intervencion": reporte.meses_desde_intervencion,
                "intervenciones_previas": _contar_intervenciones(contexto)}

    if nombre == "desempeno_tardio":
        if not reporte.etiquetas_disponibles or reporte.auc_revelado is None:
            return {"etiquetas_disponibles": False,
                    "nota": "aún no maduran etiquetas para medir desempeño real"}
        return {"etiquetas_disponibles": True,
                "periodo_revelado": reporte.periodo_revelado,
                "auc_revelado": round(reporte.auc_revelado, 4),
                "bad_rate_revelado": round(reporte.bad_rate_revelado, 4),
                "ks_revelado": round(reporte.ks_revelado, 4) if reporte.ks_revelado else None}

    return {"error": f"herramienta desconocida: {nombre}"}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM = """\
Eres el agente de decisión de un sistema que monitorea un modelo de credit scoring \
CONGELADO en producción (el "modelo víctima"). Cada mes recibes SEÑALES INDIRECTAS del \
modelo —nunca las etiquetas reales del mes en curso, que llegan con 12 meses de retardo— \
y decides qué acción correctiva tomar.

Acciones posibles (de menor a mayor costo/impacto):
- esperar      : no hacer nada este mes.
- fine_tune    : reajuste leve (solo la capa final del modelo).
- reentrenar   : reentrenar imputador + escalador + modelo con datos recientes.
- reconstruir  : rehacer el modelo completo (binning + selección + todo).

La función de costo es ASIMÉTRICA: cada mes que el modelo opera degradado acumula \
penalización, pero cada acción también cuesta (y reconstruir cuesta más que esperar). \
Actuar tarde y sobre-actuar son ambos malos.

Lo difícil es que las señales son ambiguas:
- El PSI (drift de distribución) sube tanto por 'covariate shift' BENIGNO (la población \
  cambió pero el modelo sigue ordenando bien el riesgo) como por un problema real. Un PSI \
  alto NO implica que haya que actuar.
- El 'concept drift' puede romper el modelo dejando el PSI casi intacto: es INVISIBLE a \
  los detectores de distribución. Solo el desempeño tardío (AUC) lo delata, y llega tarde.

Por eso NO decidas por un umbral simple. INVESTIGA con las herramientas antes de decidir:
mira qué variables se movieron, si es un pico o una tendencia, si ya hay desempeño real \
que confirme o descarte daño, y hace cuánto interviniste. Cuando tengas evidencia \
suficiente, llama a `registrar_decision` UNA vez con la acción y una razón breve.

Prefiere `esperar` si el drift parece covariate benigno (PSI alto pero AUC tardío sano) o \
si acabas de intervenir. Escala a `reentrenar`/`reconstruir` solo cuando la evidencia \
apunte a daño real y por su severidad.

CUIDADO con la confirmación retardada (crítico): cuando intervienes, el modelo se re-ajusta, \
pero el AUC tardío que ves el mes t mide la cohorte de hace 12 meses (t-12). Si interviniste \
DESPUÉS de que esa cohorte fue puntuada, ese AUC refleja el modelo VIEJO de antes de tu \
arreglo — es evidencia OBSOLETA, no prueba que sigas roto. Usa `historial_intervenciones`: si \
`meses_desde_intervencion` es menor que ~12, quedas CIEGO a si tu arreglo funcionó y el AUC \
tardío bajo que ves es probablemente pre-arreglo → NO vuelvas a actuar, ESPERA la confirmación \
(pagar otra acción sin saber si ya sanaste es puro desperdicio). Solo re-actúa si el AUC tardío \
que ya es POST-arreglo (intervención hace ≥12 meses) sigue mostrando daño."""


def _resumen_reporte(reporte) -> str:
    """Texto compacto del reporte del mes + un bloque JSON legible por máquina."""
    d = {
        "periodo": reporte.periodo,
        "fecha": str(reporte.fecha.date()) if hasattr(reporte.fecha, "date") else str(reporte.fecha),
        "n": reporte.n,
        "score_medio": round(reporte.score_medio, 4),
        "psi_score": round(reporte.psi_score, 4),
        "psi_score_vs_previo": (round(reporte.psi_score_vs_previo, 4)
                                if reporte.psi_score_vs_previo is not None else None),
        "psi_feat_max": round(reporte.psi_feat_max, 4),
        "confianza_media": round(reporte.confianza_media, 4),
        "meses_desde_intervencion": reporte.meses_desde_intervencion,
        "etiquetas_disponibles": reporte.etiquetas_disponibles,
        "auc_revelado": (round(reporte.auc_revelado, 4)
                         if reporte.auc_revelado is not None else None),
    }
    return ("Señales del mes (resumen). Investiga con las herramientas antes de decidir.\n"
            f"RESUMEN_JSON: {json.dumps(d, ensure_ascii=False)}")


# --------------------------------------------------------------------------- #
# Clientes: real (Anthropic) y simulado (para probar el flujo sin key)
# --------------------------------------------------------------------------- #
@dataclass
class RespuestaLLM:
    """Respuesta normalizada: por qué paró y los bloques de contenido."""
    stop_reason: str
    content: list           # bloques estilo Anthropic (con .type, .name, .input, .id)


class ClienteLLM(Protocol):
    """Contrato mínimo de un cliente de LLM para la política."""
    def responder(self, system: str, messages: list, tools: list): ...


class ClienteAnthropic:
    """Cliente real: llama a la API de Anthropic con *tool use* (function calling).

    Requiere el paquete `anthropic` y la variable ANTHROPIC_API_KEY (ver `_cargar_api_key`).
    """

    def __init__(self, modelo: str = "claude-opus-5", max_tokens: int = 4096,
                 pensar: bool = True):
        self.modelo = modelo
        self.max_tokens = max_tokens
        # `pensar` = razonamiento extendido (thinking). Cuesta más tokens pero da mejores
        # decisiones y las trazas de razonamiento. Para TESTEAR barato: pensar=False + un
        # modelo chico (claude-haiku-4-5). Haiku no soporta thinking adaptativo → se apaga.
        self.pensar = pensar and "haiku" not in modelo
        self._client = None

    def _asegurar_cliente(self):
        if self._client is not None:
            return
        try:
            import anthropic
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Falta el paquete 'anthropic'. Instálalo con:  pip install anthropic\n"
                "(ya está en requirements.in)."
            ) from e
        if _cargar_api_key() is None:
            raise RuntimeError(
                "No encuentro tu API key. Copia `.env.example` a `.env` y pega tu key en la "
                "línea ANTHROPIC_API_KEY=...  (el .env no se sube a GitHub)."
            )
        self._client = anthropic.Anthropic()

    def responder(self, system: str, messages: list, tools: list):
        self._asegurar_cliente()
        kwargs = dict(model=self.modelo, max_tokens=self.max_tokens,
                      system=system, messages=messages, tools=tools)
        if self.pensar:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return self._client.messages.create(**kwargs)


@dataclass
class _Bloque:
    """Bloque de contenido mínimo para imitar la respuesta de la API en modo simulado."""
    type: str
    name: str | None = None
    input: dict | None = None
    id: str | None = None
    thinking: str | None = None


class ClienteSimulado:
    """LLM de mentira: imita el bucle agéntico (investiga → decide) con reglas.

    Sirve para probar TODO el flujo (herramientas, ida y vuelta de tool_use/tool_result,
    parseo de la decisión) sin gastar API ni tener key. Lee solo lo que un LLM real vería
    (el RESUMEN_JSON del mensaje del usuario), así que ejercita la tubería de verdad.
    """

    def __init__(self):
        self._contador = 0

    def responder(self, system: str, messages: list, tools: list):
        self._contador += 1
        # ¿Ya investigó? (hay algún tool_result en el historial)
        ya_investigo = any(
            isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
            for m in messages
        )
        if not ya_investigo:
            # Turno 1: investigar el desempeño tardío (la señal que desambigua).
            return RespuestaLLM("tool_use", [
                _Bloque("thinking", thinking="Reviso si ya hay desempeño real que confirme daño."),
                _Bloque("tool_use", name="desempeno_tardio", input={}, id="sim_1"),
            ])

        # Turno 2: decidir con una heurística que imita a la multi-señal.
        d = self._leer_resumen(messages)
        accion, razon = self._regla(d)
        return RespuestaLLM("tool_use", [
            _Bloque("thinking", thinking="Con la evidencia decido la acción."),
            _Bloque("tool_use", name="registrar_decision",
                    input={"accion": accion, "razon": razon}, id="sim_2"),
        ])

    @staticmethod
    def _leer_resumen(messages: list) -> dict:
        for m in messages:
            cont = m.get("content")
            if isinstance(cont, str) and "RESUMEN_JSON:" in cont:
                return json.loads(cont.split("RESUMEN_JSON:", 1)[1].strip())
        return {}

    @staticmethod
    def _regla(d: dict, retardo: int = 12) -> tuple[str, str]:
        auc = d.get("auc_revelado")
        psi = d.get("psi_score") or 0.0
        meses = d.get("meses_desde_intervencion")
        # Confirmación consciente del retardo: el AUC tardío solo cuenta si nunca intervine
        # o si ya pasó el retardo (data post-arreglo). Si no, quedo ciego → espero.
        confirmado = meses is None or meses >= retardo
        if auc is not None and confirmado and auc < 0.835:      # daño confirmado (post-arreglo)
            accion = "reconstruir" if auc < 0.70 else "reentrenar"
            return accion, f"[simulado] AUC tardío {auc} confirma daño (confirmación válida)"
        if auc is not None and not confirmado and auc < 0.835:  # ciego: intervine hace poco
            return "esperar", f"[simulado] intervine hace {meses}m (<{retardo}) → espero confirmación"
        if auc is not None and confirmado and auc >= 0.86 and psi >= 0.15:  # PSI alto pero AUC sano
            return "esperar", f"[simulado] PSI {psi} alto pero AUC {auc} sano → covariate"
        if psi >= 0.20:
            return "reentrenar", f"[simulado] PSI {psi} fuerte sin confirmación aún"
        return "esperar", f"[simulado] señales bajo umbral (PSI {psi})"


# --------------------------------------------------------------------------- #
# La política
# --------------------------------------------------------------------------- #
@registrar("llm")
class PoliticaLLM:
    """Política agéntica: un LLM investiga con herramientas y luego decide.

    Parameters
    ----------
    cliente : ClienteLLM | None
        El motor de razonamiento. Si es None, se usa `ClienteAnthropic(modelo)` (modo real,
        necesita API key). Para pruebas se inyecta `ClienteSimulado()`.
    modelo : str
        Id del modelo Anthropic (solo se usa si `cliente` es None). Configurable por costo:
        'claude-opus-5' (más capaz), 'claude-sonnet-5' o 'claude-haiku-4-5' (más baratos).
    max_iteraciones : int
        Tope de vueltas del bucle agéntico (investigar ↔ recibir resultados) por decisión.
    """

    def __init__(self, cliente: ClienteLLM | None = None, modelo: str = "claude-opus-5",
                 max_iteraciones: int = 4, pensar: bool = True):
        self.cliente = cliente if cliente is not None else ClienteAnthropic(modelo, pensar=pensar)
        self.modelo = modelo
        self.max_iteraciones = max(1, max_iteraciones)
        self.trazas: list[dict] = []   # log de cada decisión (para documentar en la tesis)

    def decidir(self, reporte, contexto: Contexto) -> Decision:
        messages = [{"role": "user", "content": _resumen_reporte(reporte)}]
        traza: dict = {"periodo": reporte.periodo, "herramientas_usadas": [],
                       "razonamiento": [], "iteraciones": 0}

        for _ in range(self.max_iteraciones):
            traza["iteraciones"] += 1
            resp = self.cliente.responder(SYSTEM, messages, TODAS_LAS_HERRAMIENTAS)

            # Capturar razonamiento (resúmenes de thinking) para las trazas.
            for b in resp.content:
                if getattr(b, "type", None) == "thinking" and getattr(b, "thinking", None):
                    traza["razonamiento"].append(b.thinking)

            bloques_tool = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not bloques_tool:
                # El LLM respondió sin llamar herramientas → no hay decisión estructurada.
                break

            messages.append({"role": "assistant", "content": resp.content})

            # ¿Emitió su decisión final?
            decision_bloque = next((b for b in bloques_tool if b.name == "registrar_decision"), None)
            if decision_bloque is not None:
                dec = self._parsear_decision(decision_bloque.input)
                traza["decision"] = {"accion": dec.accion.value, "razon": dec.razon}
                self.trazas.append(traza)
                return dec

            # Si no, ejecutar las herramientas de investigación y devolver resultados.
            resultados = []
            for b in bloques_tool:
                salida = ejecutar_herramienta(b.name, b.input or {}, reporte, contexto)
                traza["herramientas_usadas"].append({"herramienta": b.name, "args": b.input})
                resultados.append({"type": "tool_result", "tool_use_id": b.id,
                                   "content": json.dumps(salida, ensure_ascii=False)})
            messages.append({"role": "user", "content": resultados})

        # Fallback: el LLM no emitió una decisión estructurada.
        dec = Decision(Accion.ESPERAR, "el LLM no emitió una decisión en el límite de iteraciones")
        traza["decision"] = {"accion": dec.accion.value, "razon": dec.razon}
        self.trazas.append(traza)
        return dec

    @staticmethod
    def _parsear_decision(entrada: dict) -> Decision:
        cruda = (entrada or {}).get("accion", "")
        razon = (entrada or {}).get("razon", "")
        try:
            accion = Accion(cruda)
        except ValueError:
            return Decision(Accion.ESPERAR, f"acción no reconocida del LLM: '{cruda}' ({razon})")
        return Decision(accion, razon)
