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
- [ ] **2. Executive Summary — reescribir para lector ejecutivo.** Eliminar β, p,
      "pooled per-1,000-word rates without firm-year fixed effects", "survives
      multiplicity correction". Quedarse con: problema, tres archetypes, gap
      call→filing, mercado no premia evidencia, framework de tres pasos. Los dos
      caveats metodológicos a una cláusula cada uno. **Modificado (el pase anterior solo
      comprimió, no simplificó el registro) — Prioridad: Máxima**
- [ ] **3. Capítulo 8 completo — objetivo 2 páginas.** §8.4 → 3-4 frases con la
      definición de AI washing de esta tesis, sin recap de las tres teorías. §8.2 −25%,
      o fusionar con §8.3 (Synthesis). **Modificado (agrega meta de páginas) — Prioridad:
      Máxima**
- [ ] **4. §8.1 — responder RQ1-RQ4 explícitamente.** Una línea cada una, además del
      veredicto H-operational/H-market. Cierra el criterio "coherent development as set
      out in the introduction". **Nuevo — Prioridad: Alta**
- [ ] **5. §1.1 — reordenar apertura.** Partir del decision problem (investors/boards/
      regulators) y recién después la difusión post-ChatGPT. **Original — Prioridad: Alta**
- [ ] **6. §1.3 + roadmap §1.5 — reordenar contribuciones.** framework → evidencia →
      arquitectura de medición. El roadmap debe decir explícitamente que los capítulos
      2-6 construyen el framework aplicado en el capítulo 7 — si no, la intro promete un
      orden que el cuerpo no entrega. **Modificado — Prioridad: Alta**
- [ ] **7. §1.4.2 — recortar definición teórica.** Agregar 1 línea de implicancia
      managerial por cada teoría citada. **Original — Prioridad: Alta**
- [ ] **8. §1.4.3 — reescribir literature gaps.** Como cuatro preguntas que un
      decision-maker no puede responder hoy (no como vacíos académicos abstractos).
      **Original — Prioridad: Media**
- [ ] **9. §1.5 (RQs) — reemplazar las cuatro colas "Derived from X theory…" por una
      tabla de 4 filas** (RQ | Teoría | Expected outcome | Capítulo). No borrar el mapa
      RQ→teoría (criterio "adequate use of theories in framing the topic"); la tabla
      resuelve también el ítem 1 (expected outcomes) y el ítem 4 (§8.1 respondiendo cada
      RQ) al mismo tiempo. **Corregido (no eliminar, convertir a tabla) — Prioridad:
      Alta**
- [ ] **10. Todo el texto + AI Statement — "we/our" → "this thesis" / primera persona
      singular.** La guía MIB insiste en que el trabajo es individual. **Nuevo —
      Prioridad: Media**
- [ ] **11. Cover letter — declarar uso de AI también ahí.** La guía lo exige en ambos
      lugares (cover letter y AI Statement). **Nuevo — Prioridad: Alta (cumplimiento)**
- [ ] **12. Figuras/tablas — medir páginas de figuras+tablas del cuerpo (42
      elementos).** Límite: 20 páginas. Si está cerca del límite, mover primero Table 3
      (prevalence) y Table 4 (distribution) al apéndice. **Nuevo — Prioridad: Alta
      (verificar antes de tocar el resto)**
- [ ] **12b. Figure 25 ("Incremental value of NLP layers over mention counts") —
      mover de Appendix E al cuerpo**, en §6.6 o como apertura de §8.1, un párrafo de
      cuatro frases. El cuerpo nunca la cita hoy. Es la respuesta más directa a la RQ
      principal (distinguir volume/posture/substance revela diferencias económicas
      reales) y la única prueba empírica de que contar keywords no basta: posture +
      activities suben R² en valuation +4.2 pp donde mention counts no aportan nada, y W
      no aporta nada (H-market matizada en una sola figura). **Nuevo — Prioridad:
      Máxima**

## Bloque B — poda mayor (robustez al apéndice)

- [ ] **13. §3.3.4 — cuerpo solo con el resultado.** k=3 estable bajo ambos bootstraps
      + una frase de validación temporal OOS. PCA, k=2, fixed-N, K-means, ARI/Jaccard →
      apéndice. **Original — Prioridad: Máxima**
- [ ] **14. §6.3 (robustness) — casi todo a Appendix E.** Conservar solo el punchline
      del placebo (asociación cross-sectional, no call-induced). **Original — Prioridad:
      Máxima**
- [ ] **15. §6.2 — sacar "martingale processes" y la justificación técnica.** Ecuación
      ANCOVA pasa a opcional/appendix. Detalle → Appendix D/E. **Original — Prioridad:
      Máxima**
- [ ] **16. §7.2.1 — comprimir solo la defensa de W, Simonian queda intacto.**
      La triple aclaración de que W no define cuadrante / no es fraud classifier / solo
      rankea se comprime a una frase: "D y S determinan el cuadrante; W prioriza revisión
      dentro de él y no es un fraud label." La referencia a Simonian (CFA Institute) que
      ancla el framework a un estándar de industria NO se corta — es lo que muestra
      "adequate use of models derived from the literature" para la rúbrica. **Corregido
      (pase anterior iba a cortar Simonian también; no corresponde) — Prioridad: Máxima**
- [ ] **17. §2.4 — recortar a la mitad.** False negatives y borderline passages →
      Appendix B. **Original — Prioridad: Muy alta**
- [ ] **18. §3.3.2.1 (refit 10-K-only) — una frase en el cuerpo.** Correlaciones y
      z-scores → apéndice. **Original — Prioridad: Muy alta**
- [ ] **19. §7.1 — Table 15 + casos en prosa. Decisión a tomar al tocar la sección:**
      (a) solo Microsoft y Welltower en prosa, JPM/GS y Apple solo en tabla, o (b) los
      cuatro casos a 2-3 líneas cada uno en vez de dos mini-casos largos. Los worked
      examples son lo más MBA de la tesis. Si hay que elegir entre cortar acá o en 3.3.4
      (ítem 13), cortar en 3.3.4. **Corregido (decisión abierta, preferir (b) si el
      espacio alcanza) — Prioridad: Muy alta**
- [ ] **20. §2.8 — dos frases.** Human audit valida producción; cross-model agreement
      es check secundario. **Original — Prioridad: Muy alta**
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
