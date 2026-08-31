# detection/001_expanded_lexicon

Proposer diagnosis and rationale:
Analysis of 000_seed false negatives on the search split showed that 10-K paragraphs discussing AI/ML
frequently reference specific AI technologies, products, and domains without explicitly using the exact
trigrams 'artificial intelligence' or 'machine learning' or the standalone word 'ai'.
Specifically:
- Product names like Copilot (row 71: 'Copilot for Microsoft 365')
- Autonomous vehicle technologies (row 291: 'advanced driver-assistance systems (ADAS)')
- Predictive modeling and intelligent search (row 51: 'predictive models')
- Data science initiatives (row 426: 'data science')
- LLMs, NLP, computer vision, neural networks, foundation models, GPT/OpenAI references.

This candidate adds targeted patterns covering core AI/ML terms, architectures, prominent AI products,
and application domains while ensuring high precision (zero false positives on the negative set).
