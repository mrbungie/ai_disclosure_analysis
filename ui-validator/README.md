# Validador humano de frames y prefiltro

Cierra el ítem #1 de `docs/problemas_academicos.md`: hoy ninguna etiqueta del
extractor de frames se comparó con un humano. Todo local, sin LLM, sin red.

```bash
uv run --frozen --no-sync python ui-validator/build_sample.py --frames 300 --prefilter 300 --activities 180
cd ui-validator && python -m http.server 8765        # abrir http://localhost:8765
```

- **Frames**: el párrafo numerado por oración tal como lo vio el juez, con la
  evidencia resaltada, y cada frame con sus etiquetas. Por frame: ¿existe?,
  ¿promocional?, temporal correcto, ¿specificity correcta?, ¿evidencia correcta?
  Si el juez no extrajo ninguno, ¿debería?
- **Actividades**: la actividad divulgada escrita como una frase ("la empresa
  despliega *copilot* para *desarrollo de software*", con destinatario, etapa,
  proveedor y tipo de evidencia) sobre el párrafo con la evidencia resaltada.
  Tres botones: **1 · Sí, está bien** (pasa solo al siguiente), **2 · Casi,
  algo está mal** (aparecen chips para tocar qué campo: acción, objeto,
  función, destinatario, etapa, proveedor, evidencia), **3 · No, el texto no
  dice eso**. Sostiene `docs/analytics/09_actividades_ia.md` y lo que `02`,
  `03`, `05`, `06` y `08` toman de ahí. Muestra estratificada por formulario y
  por fuerza de evidencia.
- **Prefiltro**: el párrafo y la decisión del prefiltro v2. ¿Menciona IA?, ¿es
  divulgación sustantiva sobre la propia empresa? Muestra estratificada por
  formulario y por probabilidad (positivo / zona gris / negativo con término).

Las anotaciones viven en `localStorage` del navegador (clave `validador.v1`),
se acumulan entre sesiones y se sacan con **Exportar** (JSON al portapapeles).
Guardar el export en `ui-validator/annotations/<quien>.json`; **Importar**
fusiona exports de otra máquina o persona (gana el más reciente por ítem).
**Resumen** muestra acuerdos crudos en pantalla; el número honesto sale de

```bash
uv run --frozen --no-sync python ui-validator/summarize.py
```

que calcula κ humano–juez por dimensión, precisión por campo de las
actividades, y precisión/recall del prefiltro
reponderando cada estrato a su tamaño real en el corpus.

Atajos: `←` `→` navegar, `1` sí, `2` no (sobre el primer frame sin responder);
en actividades `1` bien, `2` algo mal, `3` no existe.

`data.json` es determinístico (`--seed`); se regenera y no se versiona.
`annotations/` sí se versiona: es la única data que cuesta tiempo humano.
