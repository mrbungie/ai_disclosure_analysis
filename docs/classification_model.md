A continuación lo dejaría como **diseño final autocontenido**, listo para pegar en metodología, documentación técnica o para pasárselo a otra persona/modelo y pedir crítica.

# Diseño final: extracción semántica de disclosures de IA basada en frames

## Objetivo

El objetivo es transformar párrafos de disclosures corporativos sobre inteligencia artificial en una representación estructurada y reproducible que preserve las relaciones semánticas importantes entre:

* quién realiza o recibe la acción,
* qué tipo de IA se menciona,
* en qué estado temporal se encuentra,
* dónde se aplica,
* qué conceptos de interés para la tesis aparecen,
* qué tan concreta o sustentada está la afirmación,
* cómo se presenta retóricamente,
* y qué oraciones del párrafo sirven como evidencia.

La extracción se realizará a nivel de **párrafo**, pero la unidad semántica producida por el modelo será el **frame**.

El frame no corresponde necesariamente a una oración ni a un único concepto. Representa una **proposición semánticamente coherente** dentro del párrafo.

---

## 1. Unidad de entrada

Cada párrafo será previamente segmentado en oraciones y entregado al modelo con identificadores numéricos estables:

```text
[0] We currently use generative AI in our customer-service operations.
[1] The system has reduced average handling time by approximately 20%.
[2] We plan to expand these capabilities to additional markets during 2026.
```

El modelo no necesita reproducir el texto de evidencia. Sólo debe devolver los identificadores de las oraciones relevantes.

Esto permite mantener trazabilidad sin depender de extracción exacta de spans.

---

## 2. Unidad de salida: semantic frame

Un párrafo puede producir:

```text
0..N frames
```

Cada frame representa una proposición coherente sobre IA.

Conceptualmente:

```text
Frame
│
├── subject
├── ai_type
├── temporal
├── domain
│
├── concepts[]
│
├── specificity
├── rhetoric
│
└── evidence
     └── sentence_ids[]
```

Las dimensiones principales responden a preguntas distintas:

| Campo         | Pregunta                                              |
| ------------- | ----------------------------------------------------- |
| `subject`     | ¿De quién se está hablando?                           |
| `ai_type`     | ¿Qué clase de IA?                                     |
| `temporal`    | ¿Ya ocurre, está planeado, se espera o es hipotético? |
| `domain`      | ¿Dónde se aplica?                                     |
| `concepts`    | ¿Qué está afirmando concretamente la firma?           |
| `specificity` | ¿Qué evidencia hace más concreta la afirmación?       |
| `rhetoric`    | ¿Cómo está presentada discursivamente?                |
| `evidence`    | ¿Qué oraciones sustentan el frame?                    |

---

## 3. Regla central de segmentación

La regla principal es:

> **Crear el mínimo número de frames necesario para preservar correctamente las relaciones semánticas.**

No se debe crear un frame nuevo simplemente porque una misma proposición contiene varios conceptos.

Por ejemplo:

```text
[0] We currently use generative AI in customer service,
    reducing handling times and operating costs.
```

debería producir un solo frame con:

```text
ai_type = generative
temporal = realized

concepts = [
    deployed,
    productivity_outcome,
    cost_outcome
]
```

En cambio:

```text
[0] We currently use predictive ML for fraud detection,
    while we expect generative AI to improve customer service.
```

requiere al menos dos frames:

```text
Frame A
predictive_ml ↔ realized

Frame B
generative ↔ expected
```

Si se combinaran, se perdería la correspondencia correcta entre tipo de IA y temporalidad.

Por tanto:

> **Split a frame only when combining the information would incorrectly associate subject, AI type, temporal status, domain, concepts, or evidence across distinct propositions.**

---

# 4. Schema Pydantic

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# Closed vocabularies
# ============================================================

Subject = Literal[
    "firm",
    "suppliers_or_partners",
    "customers_or_market",
]

AIType = Literal[
    "generative",
    "predictive_ml",
    "unspecified",
]

Temporal = Literal[
    "realized",
    "planned",
    "expected",
    "hypothetical",
]

Domain = Literal[
    "internal",
    "customer_facing",
    "unspecified",
]


# ============================================================
# Thesis concepts
# ============================================================

Concept = Literal[
    # --------------------------------------------------------
    # Adoption / implementation
    # --------------------------------------------------------
    "deployed",
    "pilot_or_testing",
    "exploring",
    "use_stage_unspecified",
    "expansion_or_scaling",

    # --------------------------------------------------------
    # Capabilities / investment
    # --------------------------------------------------------
    "proprietary_ai",
    "third_party_ai",
    "ai_infrastructure",
    "ai_talent",
    "ai_investment",

    # --------------------------------------------------------
    # Outcomes
    # --------------------------------------------------------
    "productivity_outcome",
    "cost_outcome",
    "revenue_outcome",
    "customer_outcome",

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------
    "risk_cybersecurity",
    "risk_privacy",
    "risk_regulatory_or_legal",
    "risk_intellectual_property",
    "risk_bias_or_fairness",
    "risk_reliability_or_accuracy",
    "risk_competitive_or_disruption",
    "risk_workforce",
    "risk_operational_dependence",

    # --------------------------------------------------------
    # Governance
    # --------------------------------------------------------
    "gov_board_oversight",
    "gov_management_oversight",
    "gov_ai_policy_or_framework",
    "gov_technical_controls",
    "gov_human_oversight",
    "gov_vendor_governance",
]


# ============================================================
# Specificity / grounding evidence
# ============================================================

class SpecificityEvidence(BaseModel):
    """
    Observable evidence that makes the frame more concrete or grounded.

    These fields are not substantive thesis concepts.
    They describe the level of specificity supporting the proposition.

    Multiple values can be True simultaneously.
    """

    business_process: bool = Field(
        description=(
            "True when a concrete business process, organizational function, "
            "workflow, or operational activity is identified."
        )
    )

    product_or_system: bool = Field(
        description=(
            "True when a specific AI-enabled product, model, system, platform, "
            "tool, or application is identified."
        )
    )

    vendor_or_partner: bool = Field(
        description=(
            "True when a specific vendor, technology provider, supplier, "
            "partner, or external AI provider is identified."
        )
    )

    quantified_metric: bool = Field(
        description=(
            "True when the frame contains an explicit numerical metric, "
            "percentage, monetary amount, quantity, or measured result."
        )
    )

    date_or_timeline: bool = Field(
        description=(
            "True when the frame includes an explicit date, period, deadline, "
            "milestone, implementation schedule, or timeline."
        )
    )


# ============================================================
# Rhetorical signals
# ============================================================

class RhetoricalSignals(BaseModel):
    """
    Describes how the proposition is rhetorically communicated.

    These fields are conceptually independent from adoption,
    outcomes, risks, governance, and specificity.
    """

    promotional: bool = Field(
        description=(
            "True when the proposition uses strongly positive, transformational, "
            "superiority-oriented, leadership-oriented, revolutionary, or "
            "similarly promotional language about AI."
        )
    )

    strategic_importance: bool = Field(
        description=(
            "True when the proposition explicitly frames AI as central, "
            "critical, material, strategically important, or a strategic priority."
        )
    )


# ============================================================
# Evidence linkage
# ============================================================

class FrameEvidence(BaseModel):
    """
    Links the semantic frame to the numbered sentences supplied in the input.

    Example input:

        [0] ...
        [1] ...
        [2] ...

    The extractor returns sentence identifiers only.
    It should not reproduce the original text.
    """

    sentence_ids: list[int] = Field(
        min_length=1,
        description=(
            "Indices of all numbered sentences necessary to support this frame."
        ),
    )


# ============================================================
# Core semantic unit
# ============================================================

class AIFrame(BaseModel):
    """
    One semantically coherent proposition about AI.

    A frame binds together:

        subject
        AI type
        temporal framing
        application domain
        one or more thesis concepts
        specificity evidence
        rhetorical characteristics
        source evidence

    Multiple concepts can belong to the same frame.

    Do NOT create one frame per concept.

    Create separate frames only when keeping the information together
    would destroy an important semantic relationship.

    Example 1:

        [0] We currently use generative AI in customer service,
            reducing handling times by 20%.

    One frame:

        subject = firm
        ai_type = generative
        temporal = realized
        domain = customer_facing

        concepts = [
            deployed,
            productivity_outcome,
            customer_outcome
        ]

    Example 2:

        [0] We currently use predictive ML for fraud detection,
            but expect generative AI to improve customer service.

    At least two frames are required because:

        predictive_ml <-> realized

    and:

        generative <-> expected

    are separate semantic pairings.
    """

    subject: Subject = Field(
        description=(
            "Whose AI activity, capability, outcome, risk, or governance "
            "arrangement is being described."
        )
    )

    ai_type: AIType = Field(
        description=(
            "Type of AI referred to in this frame. "
            "'generative' includes generative AI, LLMs, and foundation-model uses. "
            "'predictive_ml' refers to traditional predictive or discriminative "
            "machine-learning applications. "
            "'unspecified' is used when the disclosure refers only to AI generally."
        )
    )

    temporal: Temporal = Field(
        description=(
            "'realized': already occurred, currently exists, or is ongoing. "
            "'planned': concrete intention, commitment, announced action, "
            "or implementation plan. "
            "'expected': management expectation, forecast, target, or anticipated outcome. "
            "'hypothetical': may, might, could, potentially, or another possibility "
            "without a concrete commitment."
        )
    )

    domain: Domain = Field(
        description=(
            "'internal': employees, internal operations, internal decision-making, "
            "or internal business processes. "
            "'customer_facing': products, services, interfaces, customer interactions, "
            "or other externally facing applications. "
            "'unspecified': the application domain is unclear, absent, or not informative."
        )
    )

    concepts: list[Concept] = Field(
        min_length=1,
        description=(
            "All thesis concepts asserted within this same semantic frame. "
            "Concepts are additive and multiple concepts may coexist."
        ),
    )

    specificity: SpecificityEvidence = Field(
        description=(
            "Observable evidence indicating how concretely this proposition "
            "is grounded in the disclosure."
        )
    )

    rhetoric: RhetoricalSignals = Field(
        description=(
            "Rhetorical characteristics associated specifically with this frame."
        )
    )

    evidence: FrameEvidence = Field(
        description=(
            "References to the numbered input sentences supporting this frame."
        )
    )


# ============================================================
# Paragraph-level output
# ============================================================

class ParagraphExtraction(BaseModel):
    """
    Structured extraction from one paragraph.

    Input paragraphs are pre-segmented into numbered sentences.

    Example:

        [0] ...
        [1] ...
        [2] ...

    The extractor returns zero or more semantic AI frames.

    Zero frames is valid when the paragraph does not contain
    information relevant to the ontology.
    """

    frames: list[AIFrame] = Field(
        default_factory=list,
        description=(
            "All distinct semantic AI frames supported by the paragraph. "
            "Use the minimum number of frames necessary to preserve "
            "the relevant semantic pairings."
        ),
    )
```

# 5. Interpretación de las dimensiones

## `subject`

Permite distinguir entre afirmaciones sobre la propia empresa y afirmaciones sobre su entorno.

```text
firm
suppliers_or_partners
customers_or_market
```

Esto es importante para evitar interpretar:

> “Our customers are increasingly adopting AI”

como evidencia de adopción de IA por parte de la firma.

---

## `ai_type`

Distingue como mínimo:

```text
generative
predictive_ml
unspecified
```

Esto permite analizar longitudinalmente la transición desde machine learning tradicional hacia GenAI.

Por ejemplo:

```text
predictive ML + realized
GenAI + expected
```

pueden coexistir dentro de una empresa y tienen significados muy distintos.

---

## `temporal`

La temporalidad describe el estatus de **cada frame**, no del párrafo completo:

```text
realized
planned
expected
hypothetical
```

Ejemplos:

```text
"We currently use..."
→ realized

"We will deploy..."
→ planned

"We expect AI to increase..."
→ expected

"AI could improve..."
→ hypothetical
```

Esto permite separar adopción realmente materializada de narrativa forward-looking.

---

## `concepts`

Los conceptos son las variables sustantivas del instrumento.

Se agrupan conceptualmente en:

```text
Adoption / implementation
Capabilities / investment
Outcomes
Risks
Governance
```

Pero la extracción no obliga a elegir sólo una dimensión.

Un mismo frame puede contener:

```text
deployed
productivity_outcome
cost_outcome
```

simultáneamente.

---

# 6. Specificity evidence

Los llamados originalmente `anchors` se separan explícitamente de los conceptos porque representan otra dimensión.

No describen **qué** dice la empresa.

Describen **qué tan concretamente lo dice**.

Por ejemplo:

```text
business_process
product_or_system
vendor_or_partner
quantified_metric
date_or_timeline
```

Una afirmación:

> “AI will transform our organization”

podría tener:

```text
specificity:
    business_process = false
    product_or_system = false
    vendor_or_partner = false
    quantified_metric = false
    date_or_timeline = false
```

Mientras que:

> “Our Azure OpenAI customer-service assistant reduced average handling time by 20% in 2025”

podría tener:

```text
business_process = true
product_or_system = true
vendor_or_partner = true
quantified_metric = true
date_or_timeline = true
```

Esta capa permite posteriormente construir medidas de **specificity / grounding** sin pedir al LLM que produzca directamente una puntuación subjetiva.

---

# 7. Rhetorical signals

La retórica también se mantiene separada de los conceptos y de la especificidad.

Actualmente se extraen:

```text
promotional
strategic_importance
```

Por ejemplo:

> “AI will revolutionize our industry and establish us as a technology leader.”

puede producir:

```text
promotional = true
strategic_importance = true
```

independientemente de si además existen evidencias concretas de implementación.

Esto permite estudiar combinaciones particularmente relevantes para la tesis, como:

```text
promotional + high specificity

promotional + low specificity

strategic + realized

strategic + hypothetical
```

---

# 8. Evidence linkage

Cada frame debe vincularse a una o más oraciones del párrafo:

```python
evidence.sentence_ids = [0, 1]
```

La relación es muchos-a-muchos:

```text
Paragraph
│
├── Sentence 0 ──────┐
├── Sentence 1 ──────┼── Frame A
├── Sentence 2 ──┐   │
│                └── Frame B
```

Una oración puede sustentar múltiples frames.

Un frame puede requerir múltiples oraciones.

Por ahora no es necesario extraer spans exactos. Los `sentence_ids` proporcionan suficiente trazabilidad para:

* auditoría humana,
* validación,
* debugging,
* recuperación posterior del texto,
* análisis de errores.

Los spans pueden añadirse posteriormente si aportan valor.

---

# 9. Ejemplo completo

Input:

```text
[0] We currently use generative AI tools from Microsoft in our customer-service operations.
[1] These systems have reduced average handling time by approximately 20%.
[2] We plan to expand the platform to additional European markets during 2026.
[3] Generative AI may also create new privacy and regulatory risks.
```

Output conceptual:

```json
{
  "frames": [
    {
      "subject": "firm",
      "ai_type": "generative",
      "temporal": "realized",
      "domain": "customer_facing",
      "concepts": [
        "deployed",
        "third_party_ai",
        "productivity_outcome"
      ],
      "specificity": {
        "business_process": true,
        "product_or_system": true,
        "vendor_or_partner": true,
        "quantified_metric": true,
        "date_or_timeline": false
      },
      "rhetoric": {
        "promotional": false,
        "strategic_importance": false
      },
      "evidence": {
        "sentence_ids": [0, 1]
      }
    },
    {
      "subject": "firm",
      "ai_type": "generative",
      "temporal": "planned",
      "domain": "customer_facing",
      "concepts": [
        "expansion_or_scaling"
      ],
      "specificity": {
        "business_process": true,
        "product_or_system": true,
        "vendor_or_partner": false,
        "quantified_metric": false,
        "date_or_timeline": true
      },
      "rhetoric": {
        "promotional": false,
        "strategic_importance": false
      },
      "evidence": {
        "sentence_ids": [2]
      }
    },
    {
      "subject": "firm",
      "ai_type": "generative",
      "temporal": "hypothetical",
      "domain": "unspecified",
      "concepts": [
        "risk_privacy",
        "risk_regulatory_or_legal"
      ],
      "specificity": {
        "business_process": false,
        "product_or_system": false,
        "vendor_or_partner": false,
        "quantified_metric": false,
        "date_or_timeline": false
      },
      "rhetoric": {
        "promotional": false,
        "strategic_importance": false
      },
      "evidence": {
        "sentence_ids": [3]
      }
    }
  ]
}
```

---

# 10. Uso downstream

El frame es la capa de medición primaria.

Los constructs de nivel superior **no son producidos directamente por el modelo**.

La transformación sería:

```text
Raw disclosure
      ↓
Paragraphs
      ↓
Semantic frames
      ↓
Deterministic aggregation
      ↓
Firm-year features
      ↓
Disclosure archetypes / clusters
```

Por ejemplo, a partir de los frames se pueden derivar:

```text
share_realized_ai_claims
share_planned_ai_claims
share_expected_ai_claims
share_hypothetical_ai_claims

share_realized_genai
share_realized_predictive_ml

share_deployment_claims
share_outcome_claims
share_risk_claims
share_governance_claims

share_quantified_claims
mean_specificity_anchors

share_promotional_claims
share_promotional_unanchored_claims

share_realized_outcomes
share_hypothetical_outcomes

share_realized_governance
share_planned_governance
```

Y combinaciones especialmente relevantes para AI washing:

```text
promotional
+
hypothetical
+
low specificity
```

frente a:

```text
realized
+
operational evidence
+
quantification
+
governance
```

La clasificación final de una firma como un determinado archetype no será impuesta por el LLM. Surgirá del análisis de estas características agregadas.

---

# 11. Principio metodológico

El principio del diseño es:

> **El LLM debe realizar extracción semántica estructurada sobre propiedades relativamente observables, mientras que los constructs teóricos y los archetypes se construyen posteriormente de forma determinística o estadística.**

Esto reduce la dependencia de juicios globales del modelo como:

```text
"Is this company AI washing?"
```

y reemplaza esa decisión por observaciones auditables como:

```text
generative AI
realized
deployed
productivity outcome
quantified
named system
promotional = false
```

El resultado es una capa intermedia rica, trazable y reutilizable que permite modificar posteriormente las definiciones de credibility, specificity, maturity o AI washing **sin tener que volver a interpretar todo el corpus desde cero**.

