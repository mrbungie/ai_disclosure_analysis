Sí po XD. Ahí **Pydantic no pinta nada**. Para el filtro barato previo al extractor estructurado, haría una pipeline puramente de retrieval/scoring:

```text
10M párrafos
    ↓
normalización de texto
    ↓
┌────────────────────┬─────────────────────┐
│                    │                     │
│  lexical matcher   │ semantic embeddings │
│                    │                     │
└──────────┬─────────┴──────────┬──────────┘
           │                    │
           ↓                    ↓
   lexical features      semantic similarities
           │                    │
           └──────────┬─────────┘
                      ↓
              deterministic score
                      ↓
                candidate / drop
```

Yo lo diseñaría así:

### 1. Léxico fuerte

Términos que por sí solos hacen pasar el párrafo:

```python
STRONG_AI_TERMS = [
    "artificial intelligence",
    "generative ai",
    "genai",
    "machine learning",
    "deep learning",
    "large language model",
    "large language models",
    "foundation model",
    "foundation models",
    "neural network",
    "neural networks",
]
```

Y algunos nombres/modelos si te interesan:

```python
STRONG_AI_PRODUCTS = [
    "chatgpt",
    "openai",
    "gpt-4",
    "copilot",
    "gemini",
    "claude",
]
```

No necesitas LLM para esto. A 10M párrafos usaría Aho-Corasick/Hyperscan/regex compilado, según el stack.

### 2. Léxico débil

Términos que **no bastan solos**, porque tienen muchísimo ruido:

```python
WEAK_AI_TERMS = [
    "algorithm",
    "predictive",
    "prediction",
    "recommendation",
    "classifier",
    "automation",
    "automated decision",
    "computer vision",
    "natural language processing",
    "nlp",
    "model",
]
```

`model`, por ejemplo, jamás debería pasar solo porque en 10-K vas a tener `business model`, `valuation model`, etc.

### 3. Anchors semánticos

Aquí sí partiría de los conceptos de tu schema final, pero como **frases de retrieval**, no como labels a extraer.

Por ejemplo:

```python
SEMANTIC_ANCHORS = {
    "ai_use": [
        "The company is using or deploying artificial intelligence in business operations.",
        "Artificial intelligence is integrated into a product, service, or workflow.",
    ],

    "ai_exploration": [
        "The company is experimenting with, piloting, evaluating, or exploring artificial intelligence.",
    ],

    "ai_capability": [
        "The company develops proprietary artificial intelligence models, systems, or platforms.",
        "The company relies on third-party artificial intelligence providers or models.",
        "The company invests in AI infrastructure, computing, data, or talent.",
    ],

    "ai_outcome": [
        "Artificial intelligence improves productivity or operational efficiency.",
        "Artificial intelligence reduces costs.",
        "Artificial intelligence generates revenue or commercial growth.",
        "Artificial intelligence improves customer experience or personalization.",
    ],

    "ai_risk": [
        "Artificial intelligence creates cybersecurity or privacy risks.",
        "Artificial intelligence creates regulatory, legal, copyright, or intellectual-property risks.",
        "Artificial intelligence may produce inaccurate, biased, unreliable, or harmful outputs.",
        "Artificial intelligence creates competitive or workforce disruption.",
    ],

    "ai_governance": [
        "The company has governance, policies, oversight, controls, or human review for artificial intelligence.",
    ],

    "ai_strategy": [
        "Artificial intelligence is described as strategically important or transformative for the company.",
    ],
}
```

Embebes estos anchors **una sola vez**.

Después, para cada párrafo:

```python
paragraph_vec = encoder.encode(paragraph)

scores = {
    anchor_name: max_cosine(paragraph_vec, anchor_vectors)
    for anchor_name, anchor_vectors in anchors.items()
}
```

### 4. Lo importante: no usaría un solo umbral

Tendría tres rutas de entrada:

```python
candidate = (
    strong_lexical_match
    or max_semantic_score >= HIGH_THRESHOLD
    or (
        weak_lexical_match
        and max_semantic_score >= LOW_THRESHOLD
    )
)
```

Por ejemplo, **conceptualmente**, no como thresholds definitivos:

```python
HIGH_THRESHOLD = 0.65
LOW_THRESHOLD = 0.45
```

Esos valores los calibras con tu gold set.

### 5. Incluso agregaría negativos semánticos

Esto puede ser muy útil con financial filings porque existen palabras horriblemente ambiguas como `model`, `learning`, `automation`.

Podrías tener:

```python
NEGATIVE_ANCHORS = [
    "The paragraph discusses a business model rather than machine learning.",
    "The paragraph discusses financial valuation models.",
    "The paragraph discusses economic forecasting without artificial intelligence.",
    "The paragraph discusses ordinary software automation without machine learning or AI.",
]
```

Entonces algo tipo:

```python
ai_score = max_positive_similarity - max_negative_similarity
```

puede funcionar mejor que cosine absoluto.

### 6. La salida del filtro tampoco necesita schema sofisticado

Guardaría columnas:

```text
paragraph_id
strong_lexical_match
weak_lexical_match
matched_terms
max_semantic_score
best_semantic_anchor
second_best_semantic_anchor
negative_similarity
candidate
```

Quizá también todos los scores si el almacenamiento no te molesta:

```text
score_ai_use
score_ai_capability
score_ai_outcome
score_ai_risk
score_ai_governance
...
```

Eso después sirve muchísimo para debugging y sampling.

### 7. Y recién después entra tu Pydantic

```text
10M paragraphs
        ↓
lexical + embedding retrieval
        ↓
candidate paragraphs
        ↓
Frontier/SLM
        ↓
ParagraphExtraction / AIFrame Pydantic
```

Es decir:

**retrieval = texto + vectores + reglas.**

**measurement = structured LLM + Pydantic.**

No mezclaría ambas capas. Ahí tenías razón.

