# TODO — pase final antes de congelar `thesis.qmd`

Checklist para agentes/colaboradores que retomen esta tesis. Cada ítem referencia una
sección de `thesis.qmd` (numeración del documento renderizado, no de headings de Quarto —
usar `grep`/lectura de contexto para ubicar el bloque exacto antes de editar). Después de
cada tanda de cambios: `./render.py` desde `thesis_document/` y verificar que compile antes
de seguir. Commitear seguido (ver `CLAUDE.md`: "COMMIT TO B2 AND GIT OFTEN").

**Orden de ejecución: Bloque A → Bloque B → Bloque C.**
A es lo que el jurado lee contra su propia rúbrica; B es lo que hace que la tesis se vea
segura (mueve robustez/detalle técnico al apéndice); C es limpieza menor. Si el tiempo
aprieta, A completo + los primeros 6 de B ya cambian la lectura del documento.

---

## Bloque A — criterios de nota (guía MIB)

- [x] **1. §1.5 (o nuevo §1.6) — Metodología + expected outcome por RQ.** Agregar ½
      página: metodología en 4-5 líneas (S&P 500 fijo 2021-01-01, 59.8k documentos, gate
      ML + extractor LLM, seis constructos D/P/A/G/S/W, tres diseños empíricos: archetypes,
      cross-venue, mercado) + expected outcome por RQ (una línea cada uno). Criterio
      explícito de nota de la guía MIB. **Nuevo — Prioridad: Máxima**
- [x] **2. Executive Summary — reescribir para lector ejecutivo.** Eliminar β, p,
      "pooled per-1,000-word rates without firm-year fixed effects", "survives
      multiplicity correction". Quedarse con: problema, tres archetypes, gap
      call→filing, mercado no premia evidencia, framework de tres pasos. Los dos
      caveats metodológicos a una cláusula cada uno. **Modificado (el pase anterior solo
      comprimió, no simplificó el registro) — Prioridad: Máxima**
- [x] **3. Capítulo 8 completo — objetivo 2 páginas.** §8.4 → 3-4 frases con la
      definición de AI washing de esta tesis, sin recap de las tres teorías. §8.2 −25%,
      o fusionar con §8.3 (Synthesis). **Modificado (agrega meta de páginas) — Prioridad:
      Máxima** — Hecho: §8.4 reducida a 3 frases (definición operacional, sin
      re-explicar las teorías de §1.4). §8.2 y §8.3 fusionadas en "What the evidence
      establishes and its synthesis" (eran redundantes: hallazgos + interpretación de
      los mismos). Capítulo 8 completo ~1130 palabras (~2 páginas). §8.1, §8.5, §8.6 sin
      tocar (ítems 4 y 23).
- [x] **4. §8.1 — responder RQ1-RQ4 explícitamente.** Una línea cada una, además del
      veredicto H-operational/H-market. Cierra el criterio "coherent development as set
      out in the introduction". **Nuevo — Prioridad: Alta** — Hecho: agregado un párrafo
      con una línea por RQ (RQ1-RQ4) después del veredicto H-operational/H-market
      existente (sin borrarlo), citando el hallazgo real de cada pregunta.
- [x] **5. §1.1 — reordenar apertura.** Partir del decision problem (investors/boards/
      regulators) y recién después la difusión post-ChatGPT. **Original — Prioridad: Alta**
      — Hecho: la sección abre ahora con los tres problemas prácticos (investors/analysts,
      managers/boards, regulators), y el párrafo de adopción post-ChatGPT pasa a segundo
      lugar, conectado como el motivo de urgencia actual del problema. Sin contenido nuevo,
      solo reordenamiento y ajuste mínimo de conectores.
- [x] **6. §1.3 + roadmap §1.5 — reordenar contribuciones.** framework → evidencia →
      arquitectura de medición. El roadmap debe decir explícitamente que los capítulos
      2-6 construyen el framework aplicado en el capítulo 7 — si no, la intro promete un
      orden que el cuerpo no entrega. **Modificado — Prioridad: Alta**
      — Hecho: las tres contribuciones de §1.3 quedaron en orden framework de
      credibilidad práctica → evidencia empírica → arquitectura de medición
      multidimensional. El párrafo final de "Methodology overview and expected outcomes"
      (roadmap de cierre del cap. 1) ahora dice explícitamente que los capítulos 2-6
      construyen el framework de medición aplicado en el capítulo 7, y que el capítulo 8
      concluye.
- [x] **7. §1.4.2 — recortar definición teórica.** Agregar 1 línea de implicancia
      managerial por cada teoría citada. **Original — Prioridad: Alta** — Hecho: las
      cuatro teorías (signalling/cheap talk, legitimidad institucional, decoupling,
      cumplimiento regulatorio) mantienen la cita mínima necesaria y cada una cierra con
      una línea de implicancia managerial concreta (board/investor, peer-group,
      directores, ponderación call vs. 10-K), consistente con el framework de due
      diligence del capítulo 7. Largo total de la subsección se redujo levemente
      (307 → 300 palabras).
- [x] **8. §1.4.3 — reescribir literature gaps.** Como cuatro preguntas que un
      decision-maker no puede responder hoy (no como vacíos académicos abstractos).
      **Original — Prioridad: Media**
- [x] **9. §1.5 (RQs) — reemplazar las cuatro colas "Derived from X theory…" por una
      tabla de 4 filas** (RQ | Teoría | Expected outcome | Capítulo). No borrar el mapa
      RQ→teoría (criterio "adequate use of theories in framing the topic"); la tabla
      resuelve también el ítem 1 (expected outcomes) y el ítem 4 (§8.1 respondiendo cada
      RQ) al mismo tiempo. **Corregido (no eliminar, convertir a tabla) — Prioridad:
      Alta**
- [x] **10. Todo el texto + AI Statement — "we/our" → "this thesis" / primera persona
      singular.** La guía MIB insiste en que el trabajo es individual. **Nuevo —
      Prioridad: Media**
- [x] **11. Cover letter — declarar uso de AI también ahí.** La guía lo exige en ambos
      lugares (cover letter y AI Statement). **Nuevo — Prioridad: Alta (cumplimiento)** —
      Hecho: se insertó un párrafo breve en `cover.docx`, después de la declaración de
      trabajo individual, consistente con el AI Statement de `thesis.qmd`.
- [x] **12. Figuras/tablas — medir páginas de figuras+tablas del cuerpo (42
      elementos).** Límite: 20 páginas. Si está cerca del límite, mover primero Table 3
      (prevalence) y Table 4 (distribution) al apéndice. **Nuevo — Prioridad: Alta
      (verificar antes de tocar el resto)** — Hecho: medido sobre el PDF compilado
      (`GermanOviedo_FinalThesis_20260914_101629.pdf`, 71 páginas). El cuerpo (páginas
      4–56, antes de "Appendices" en p.57) contiene 43 elementos numerados (19 tablas +
      24 figuras), no 42 — número casi idéntico al estimado del checklist. Midiendo el
      espacio vertical real de cada figura/tabla vía `pdftotext -bbox` (altura de imagen
      exacta para figuras; extensión de fila hasta el siguiente heading/nota para
      tablas) el total ocupado en el cuerpo es de aproximadamente **15.8 páginas
      -equivalentes**, con margen razonable (~4 páginas) por debajo del límite de 20.
      No se movió nada: Table 3 (`tbl-funnel-summary`, "Compact summary of the
      inference funnel") y Table 4 (`tbl-frame-schema`, "Semantic frame schema...
      observed corpus distributions") permanecen en el cuerpo, en `thesis.qmd` líneas
      579 y 638 respectivamente.
- [ ] **12b. Figure 25 ("Incremental value of NLP layers over mention counts") —
      mover de Appendix E al cuerpo**, en §6.6 o como apertura de §8.1, un párrafo de
      cuatro frases. El cuerpo nunca la cita hoy. Es la respuesta más directa a la RQ
      principal (distinguir volume/posture/substance revela diferencias económicas
      reales) y la única prueba empírica de que contar keywords no basta: posture +
      activities suben R² en valuation +4.2 pp donde mention counts no aportan nada, y W
      no aporta nada (H-market matizada en una sola figura). **Nuevo — Prioridad:
      Máxima**

## Bloque B — poda mayor (robustez al apéndice)

- [x] **13. §3.3.4 — cuerpo solo con el resultado.** k=3 estable bajo ambos bootstraps
      + una frase de validación temporal OOS. PCA, k=2, fixed-N, K-means, ARI/Jaccard →
      apéndice. **Original — Prioridad: Máxima**
- [x] **14. §6.3 (robustness) — casi todo a Appendix E.** Conservar solo el punchline
      del placebo (asociación cross-sectional, no call-induced). **Original — Prioridad:
      Máxima**
- [x] **15. §6.2 — sacar "martingale processes" y la justificación técnica.** Ecuación
      ANCOVA pasa a opcional/appendix. Detalle → Appendix D/E. **Original — Prioridad:
      Máxima**
- [x] **16. §7.2.1 — comprimir solo la defensa de W, Simonian queda intacto.**
      La triple aclaración de que W no define cuadrante / no es fraud classifier / solo
      rankea se comprime a una frase: "D y S determinan el cuadrante; W prioriza revisión
      dentro de él y no es un fraud label." La referencia a Simonian (CFA Institute) que
      ancla el framework a un estándar de industria NO se corta — es lo que muestra
      "adequate use of models derived from the literature" para la rúbrica. **Corregido
      (pase anterior iba a cortar Simonian también; no corresponde) — Prioridad: Máxima**
- [x] **17. §2.4 — recortar a la mitad.** False negatives y borderline passages →
      Appendix B. **Hecho (150→~74 palabras de prosa; detalle movido a "Error taxonomy
      and classification robustness" en Appendix B) — Prioridad: Muy alta**
- [x] **18. §3.3.2.1 (refit 10-K-only) — una frase en el cuerpo.** Correlaciones y
      z-scores → apéndice. **Hecho (cuerpo a una frase; correlaciones y z-scores del
      refit a nueva tabla @tbl-e-10k-refit en Appendix E, no existían antes en ninguna
      tabla) — Prioridad: Muy alta**
- [x] **19. §7.1 — Table 15 + casos en prosa.** Resuelto con opción **(b)**: los cuatro
      casos (Microsoft, JPM vs. Goldman Sachs, Apple, Welltower) se mantienen en prosa,
      cada uno a 2-3 líneas (hallazgo → magnitud/mecanismo → interpretación). El estado
      previo ya tenía los cuatro casos en prosa pero repetía literalmente los valores de
      D/S/W de Table 15 dentro del texto; la revisión quita esa duplicación (los números
      quedan solo en la tabla, la prosa pasa a interpretación/mecanismo) sin sacrificar
      ningún caso — consistente con la preferencia explícita del autor por mantener los
      cuatro worked examples. **Prioridad: Muy alta**
- [x] **20. §2.8 — dos frases.** Resuelto: el cuerpo quedó en dos frases. (1) Blind
      human audits on the production judge's output validate the pipeline (91.2%
      precision on GBDT prefiltering; 89.1%–95.8% field-level accuracy). (2) Cross-model
      agreement (Gemini vs. Qwen, $\kappa = 0.81$–$0.87$) is a secondary consistency
      check, since neither model is the production judge. Todos los números ya existían
      en `@tbl-evaluation-datasets` (Appendix B), así que no hizo falta crear entradas
      nuevas. **Prioridad: Muy alta**
- [ ] **21. §5.4 (Welltower) — comprimir los tres caveats causales.** A: "descriptive
      case evidence, not causal identification." **Original — Prioridad: Muy alta**
- [ ] **22. Apertura Cap. 6 — a la mitad.** "Analyses are exploratory, common
      specifications plus FDR correction." Cortar Tukey; conservar la cita a
      Gelman-Loken (es la señal de por qué se corrige por multiplicidad). **Corregido
      (Gelman-Loken se queda, solo Tukey se corta) — Prioridad: Muy alta**
- [ ] **23. §8.6 (Limitations) — cuatro límites.** Disclosed ≠ implemented; no causal;
      S&P 500; 2026 YTD. Gemini/Qwen queda documentado en §2.8, no repetir acá.
      **Original — Prioridad: Muy alta**
- [ ] **24. §3.3.1 — reducir ~40% la prosa** entre Table 6 / Fig 6 / Table 7; caveat de
      "No AI" a una frase. **Original — Prioridad: Muy alta**

## Bloque C — poda menor y precisión

- [ ] **25. §2.1 — detalle 499/504/M&A a nota o apéndice.** Figure 1 a 2-3
      observaciones en prosa. **Original — Prioridad: Alta**
- [ ] **26. §2.2 (KDD/CRISP-DM) — no borrar, mover a Appendix A, sección "pipeline".**
      Framework que el comité (management) reconoce; ahí no estorba y suma. **Corregido
      (mover, no cortar) — Prioridad: Alta**
- [ ] **27. §2.6 (aggregation) — mantener la mediana en el cuerpo.** Reliability rho y
      empirical Bayes → Appendix B/E. **Original — Prioridad: Alta**
- [ ] **28. §2.7 — coverage % y fallback → Appendix D.** **Original — Prioridad: Alta**
- [ ] **29. §3.3 (AA setup) — una frase.** "AA represents each firm as a convex
      mixture of extreme disclosure profiles." Ecuación pasa a opcional. **Original —
      Prioridad: Alta**
- [ ] **30. §3.3.3 — metodología a una frase** (expanding windows, no look-ahead);
      prosa a 3 mensajes. **Original — Prioridad: Alta**
- [ ] **31. §3.4 — −30% recap empírico.** k=4 fallido a una frase. **Original —
      Prioridad: Alta**
- [ ] **32. §4.8 — un párrafo de síntesis + transición, nada más.** **Original —
      Prioridad: Alta**
- [ ] **33. §5.1 — no repetir la fórmula de W.** Prosa de Figure 16 −30-40%; split-half
      → apéndice. **Original — Prioridad: Alta**
- [ ] **34. §5.2 — FE a 3 líneas.** "filtering under statutory liability" →
      "consistent with filtering under greater statutory exposure" (lenguaje de
      asociación, no causal). **Original — Prioridad: Alta**
- [ ] **35. §5.3 — el caveat del denominador solo en un lugar** (nota o prosa, no
      ambos). **Original — Prioridad: Alta**
- [ ] **36. §6.3 — leave-one-year-out → Appendix E.** Párrafo de atenuación a una
      frase. **Original — Prioridad: Alta**
- [ ] **37. §6.4 — mantener el null result + multiplicity correction.** Antes de sacar
      cada p-value listado, verificar que esté en Figure 23 o Table 33; los que no estén
      ahí van a una tabla compacta en Appendix E (no se borran sin más). **Corregido
      (verificar antes de cortar) — Prioridad: Alta**
- [ ] **38. §6.5.1 — horizonte a una frase.** Sector-exclusion y posture-switchers →
      Appendix E. La comparación con Yang (2026) NO se toca: es el único resultado de
      mercado robusto y es lo que lo sitúa en la literatura en vez de dejarlo como
      asociación suelta. **Corregido (Yang 2026 se queda intacta) — Prioridad: Alta**
- [ ] **39. Recortes menores varios** — §2.3, §2.5, §3.1, §3.2, §4.3, §4.4, §4.6, §4.7,
      apertura §5, §5.5, §8.2: threshold/features al apéndice, distribuciones al
      apéndice, "never input to clustering" decirlo una sola vez, seasonal factor a
      nota al pie, etc. Ver detalle caso a caso al tocar cada sección. **§6.6 (Blades et
      al.) queda fuera de esta lista — NO se toca, ver nota abajo. Corregido — Prioridad:
      Media**

      **Nota sobre §6.6 (Blades et al.): se queda íntegro, no se recorta.** No es "otra
      cita más": es el único lugar del cuerpo que explica *por qué* el mercado no premia
      la evidencia sin que suene a null result — analistas experimentados discriminan,
      el promedio no. Eso convierte el H-market rechazado en un hallazgo con mecanismo,
      y es el puente natural al capítulo 7 (si el mercado promedio no filtra, alguien
      necesita la herramienta). Es un párrafo de cuatro frases; no hay nada ahí que
      repita algo ya dicho en el cuerpo, así que no aplica el criterio de corte (un
      párrafo se corta si repite algo ya dicho, no si cita a alguien).

---

## Regla general de poda (aplica a todo el Bloque B y C)

- **Lo único que se borra de verdad es texto repetido.** Un caveat/frase/fórmula dicho
  dos o tres veces en el cuerpo (capability caveat, caveat del denominador, "never input
  to clustering", "AI washing ≠ prueba legal", "W ≠ fraud classifier", la fórmula de W
  repetida en §5.1, el párrafo que explica Table 12 antes de Table 12, el recap de
  teorías en §8.4 que ya está en §1.4, etc.) se deja dicho **una sola vez**, no cero
  veces. No es un criterio de "esto es prescindible", es un criterio de "esto ya está
  dicho en otro lugar del cuerpo".
- **Ningún número desaparece de la tesis.** Cuando un ítem dice "sacar los p-values /
  correlaciones / porcentajes de la prosa", significa sacarlos de la prosa, no del
  documento. Regla única: todo número que sale de la prosa tiene que seguir existiendo
  en una figura, tabla o apéndice existente; si no existe ahí, **crear la tabla/entrada
  en el apéndice antes de borrar el número de la prosa**, nunca después ni "por ahora".
  No asumir que ya está en una tabla — verificarlo (grep por el número/la figura/tabla
  citada) antes de cortar. Esto aplica en particular a los ítems 37 (p-values de §6.4 →
  Figure 23 / Table 33) y 18 (r=.928/.677/.234 del refit 10-K-only → Appendix E); ambos
  quedan pendientes de esa verificación explícita, no dar por hecho que ya están cubiertos.
- Al terminar la poda, si se quiere cerrar sin duda: comparar el PDF anterior contra el
  podado y extraer todos los números del cuerpo viejo para confirmar que cada uno sigue
  apareciendo en algún lugar de la versión nueva (cuerpo o apéndice). Es un cruce
  mecánico, hacerlo antes de dar el Bloque B/C por terminado si hay dudas sobre algún
  ítem.

## Notas de proceso

- Cada checkbox es una edición de texto en `thesis.qmd` (y, cuando corresponda, mover
  contenido a los apéndices existentes — no crear apéndices nuevos sin verificar que no
  exista ya una sección equivalente).
- No tocar los datos en `data/` como efecto colateral de estos cambios (ver regla de
  `CLAUDE.md` sobre no borrar datos etiquetados por LLM).
- Después de cada bloque completo (A, B, o C), correr `./render.py` desde
  `thesis_document/` y revisar que no haya quedado ninguna referencia rota (`@fig-`,
  `@tbl-`, `@sec-`) ni texto que dependa de un párrafo que se movió al apéndice.
- Buscar globalmente "Chapter", "Section", "above", "below", "earlier", "later" después
  de cada tanda de recortes — mover prosa a un apéndice a menudo deja una referencia
  cruzada colgando en el cuerpo (regla ya presente en `CLAUDE.md`).
