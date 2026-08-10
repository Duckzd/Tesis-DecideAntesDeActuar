# Datos

Los datos **no se versionan** en git (son pesados) — ver `.gitignore`. Para reproducir,
colocar los archivos en estas carpetas.

## `buro/` — Modelo principal (credit scoring de buró)

Datos crudos del buró de crédito y artefactos derivados. Archivos clave:

- `InfoModelamiento.pkl` — base de modelamiento (features + `VarDep`), salida del
  notebook `notebooks/modelo_buro/02_VariableDependiente.ipynb`.
- `info.pkl` — base consolidada (salida del notebook 01).
- `reglas_binning.pkl`, `features_seleccionadas.pkl` — artefactos del notebook 03.
- Fuentes crudas: `DataInicial_*.txt`, `INDICADORES.RData`, `EstructuraVariables/`.

## `lendingclub/` — Segundo modelo víctima (generalización)

- `accepted_2007_to_2018Q4.csv` — dataset LendingClub (préstamos 2007–2018).
  Descargable de Kaggle: *wordsforthewise / lending-club*.

## Regenerar artefactos

1. Colocar los datos crudos en las carpetas de arriba.
2. Correr los notebooks de `notebooks/modelo_buro/` (01 → 02 → 03) para el modelo principal.
3. Correr `notebooks/modelo_lendingclub/modelo2_lendingclub_borrador.ipynb` para el segundo.
