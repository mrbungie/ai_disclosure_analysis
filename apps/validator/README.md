# Validador Atómico de Divulgación de IA (`apps/validator`)

Herramienta de validación humana atómica para auditar las extracciones de modelos de IA del proyecto (prefiltro, frames semánticos y actividades operacionales).

Cada verificación está **aplanada en 1 hecho a la vez**: una única pregunta directa y objetiva contrastada contra el texto del párrafo (con las oraciones de evidencia resaltadas), respondida con 3 opciones simples:
- **[Y] Sí / Yes**
- **[N] No**
- **[U] Incierto / Uncertain**

---

## 1. Puesta en Marcha

```bash
# Servir la interfaz web:
make validator
# o bien:
cd apps/validator && python -m http.server 8765

# Abrir en el navegador:
http://localhost:8765
```

Para regenerar o aplanar la muestra de verificación:
```bash
# Aplanar a partir de un sample previo (rápido):
uv run --frozen --no-sync python apps/validator/build_sample.py --from-existing apps/validator/data.json

# O muestrear desde cero recorriendo bronze y silver:
uv run --frozen --no-sync python apps/validator/build_sample.py --frames 300 --prefilter 300 --activities 120
```

---

## 2. Ámbitos de Validación

En la barra superior podés elegir el **ámbito**:

1. **Prefilter** (Mención de IA y Relevancia Corporativa):
   - Mención explícita de IA / Machine Learning / GenAI.
   - Divulgación sustantiva sobre la propia empresa vs incidental / boilerplate.
2. **Frames** (Frames Semánticos del Juez LLM):
   - Existencia real del frame (no alucinación).
   - Dimensión temporal (`realized`, `planned`, `expected`, `hypothetical`).
   - Retórica promocional (hype corporativo sin sustancia vs factual).
   - Especificidades identificadas (producto/sistema, proceso de negocio, proveedor/socio, métrica cuantificada, fecha/plazo).
   - Validez de la evidencia textual atribuida.
   - Párrafos negativos: si se omitió algún frame que debió extraerse.
3. **Activities** (Actividades Operacionales de IA):
   - Hecho principal afirmado (empresa despliega/desarrolla herramienta X).
   - Origen de la tecnología de IA (**propia** sólo si la empresa la construye/es su producto; **de terceros** si nombra proveedor/herramienta externa; **no dice** si no se especifica).
   - Acción y objeto tecnológico.
   - Destinatario final y etapa de madurez / adopción.
   - Entidades nombradas y roles (marca propia, proveedor externo, socio, etc.).
   - Evidencia textual directa.
   - Completitud del párrafo (si faltó alguna actividad que el texto sí describe).

---

## 3. Atajos de Teclado (Ultra-rápido)

- `Y` ó `1`: Marcar **Sí / Yes** (y avanza automáticamente al siguiente ítem).
- `N` ó `2`: Marcar **No** (y avanza automáticamente al siguiente ítem).
- `U` ó `3`: Marcar **Incierto / Uncertain** (y avanza automáticamente).
- `←` ó `A`: Ítem anterior.
- `→` ó `D`: Ítem siguiente.
- `P`: Ir al primer ítem pendiente sin responder.

---

## 4. Persistencia y Exportación

- Las respuestas se guardan continuamente en el `localStorage` del navegador.
- **Exportar**: Genera un archivo JSON con todas las validaciones realizadas. Guardar en `apps/validator/annotations/<nombre>.json`.
- **Importar**: Permite cargar o fusionar anotaciones desde otra máquina o sesión.
- **Resumen**:
  ```bash
  uv run --frozen --no-sync python apps/validator/summarize.py
  ```
  Calcula métricas agregadas: matrices de confusión, precisión/recall (crudos y reponderados por estrato para el prefiltro), tasas de acuerdo, y métricas por categoría.
