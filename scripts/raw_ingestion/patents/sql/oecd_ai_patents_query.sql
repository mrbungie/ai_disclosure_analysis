
WITH matched_assignees AS (
  SELECT
    ticker,
    cik,
    company_name,
    assignee_regex
  FROM `skillforge-dev-502010.thesis_patents.sp500_patent_aliases_seed`
),
-- assignee_harmonized.name repeats across every patent a firm holds (millions of rows
-- collapse to ~5M distinct strings in the 2015-2026 filing window). Running the 549-way
-- REGEXP_CONTAINS once per DISTINCT name instead of once per (patent, assignee) row cuts
-- the expensive regex evaluation ~33x; the second pass below re-joins by plain string
-- equality (a hash join), which is cheap regardless of table size.
distinct_assignee_names AS (
  SELECT DISTINCT assignee_harmonized.name AS name
  FROM `patents-public-data.patents.publications`,
  UNNEST(assignee_harmonized) AS assignee_harmonized
  WHERE filing_date >= 20150101 AND filing_date <= 20260901
),
matched_assignee_names AS (
  SELECT
    da.name AS assignee_name,
    m.ticker,
    m.cik,
    m.company_name
  FROM distinct_assignee_names da
  JOIN matched_assignees m
    ON REGEXP_CONTAINS(UPPER(da.name), m.assignee_regex)
),
-- Candidate patents belonging to S&P 500 assignees
candidate_patents AS (
  SELECT
    mn.ticker,
    mn.cik,
    mn.company_name,
    p.publication_number,
    p.filing_date,
    p.publication_date,
    DIV(p.filing_date, 10000) AS filing_year,
    DIV(p.publication_date, 10000) AS publication_year,
    -- Extract distinct CPC main groups for this patent
    ARRAY(
      SELECT DISTINCT SPLIT(code, '/')[OFFSET(0)]
      FROM UNNEST(p.cpc)
      WHERE code IS NOT NULL AND INSTR(code, '/') > 0
    ) AS cpc_groups,
    -- Concatenate title and abstract text
    LOWER(
      CONCAT(
        COALESCE((SELECT STRING_AGG(text, ' ') FROM UNNEST(p.title_localized)), ''),
        ' ',
        COALESCE((SELECT STRING_AGG(text, ' ') FROM UNNEST(p.abstract_localized)), '')
      )
    ) AS text_combined
  FROM `patents-public-data.patents.publications` p,
  UNNEST(p.assignee_harmonized) AS ah
  JOIN matched_assignee_names mn
    ON ah.name = mn.assignee_name
  WHERE p.filing_date >= 20150101 AND p.filing_date <= 20260901
),
-- Deduplicate patent by publication_number and firm
deduped_patents AS (
  SELECT
    ticker,
    cik,
    company_name,
    publication_number,
    filing_year,
    publication_year,
    cpc_groups,
    text_combined
  FROM candidate_patents
  QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker, publication_number) = 1
),
-- Evaluate OECD 2025 AI conditions
patent_evaluation AS (
  SELECT
    ticker,
    cik,
    company_name,
    publication_number,
    filing_year,
    publication_year,
    -- Condition 1: Core AI (at least one CPC in 5 core groups)
    EXISTS(
      SELECT 1 FROM UNNEST(cpc_groups) g
      WHERE g IN ('G06N3', 'G06N5', 'G06N7', 'G06N20', 'G06F18')
    ) AS is_core_ai,
    -- Condition 2: Related AI (at least one CPC in 95 related groups AND >= 1 OECD keyword)
    (
      EXISTS(
        SELECT 1 FROM UNNEST(cpc_groups) g
        WHERE g IN ('B60W2040', 'B60W2050', 'B60W2400', 'B60W2420', 'B60W2422', 'B60W2520', 'B60W2540', 'B60W2552', 'B60W2554', 'B60W2555', 'B60W2556', 'B60W2720', 'B60W2754', 'B60W2756', 'B60W30', 'B60W40', 'B60W50', 'B60W60', 'B64U10', 'B64U20', 'B64U2101', 'B64U2201', 'B64U30', 'B64U40', 'B64U50', 'B64U60', 'B64U70', 'B64U80', 'G06F15', 'G06F16', 'G06F17', 'G06F2111', 'G06F2113', 'G06F2119', 'G06F2207', 'G06F2209', 'G06F2216', 'G06F2218', 'G06F30', 'G06F40', 'G06F5', 'G06F7', 'G06J1', 'G06N99', 'G06Q10', 'G06Q30', 'G06Q40', 'G06Q50', 'G06Q99', 'G06T1', 'G06T11', 'G06T13', 'G06T15', 'G06T17', 'G06T19', 'G06T2200', 'G06T2201', 'G06T2207', 'G06T2210', 'G06T2211', 'G06T2215', 'G06T2219', 'G06T3', 'G06T5', 'G06T7', 'G06T9', 'G06V10', 'G06V20', 'G06V2201', 'G06V30', 'G06V40', 'G16B10', 'G16B15', 'G16B20', 'G16B25', 'G16B30', 'G16B35', 'G16B40', 'G16B45', 'G16B5', 'G16B50', 'G16C10', 'G16C20', 'G16C60', 'G16H10', 'G16H15', 'G16H20', 'G16H30', 'G16H40', 'G16H50', 'G16H70', 'G16H80', 'G16Y20', 'G16Y40', 'G16Z99')
      )
      AND (
        REGEXP_CONTAINS(text_combined, r'(?i)\baction\s+recognition\b|\bactivity\s+recognition\b|\badaboost\b|\badaptive\s+boosting\b|\badversarial\s+learning\b|\badversarial\s+network\b|\bai\s+algorithm\b|\bai\s+chipset\b|\balgorithmic\s+of\s+ai\b|\bambient\s+intelligence\b|\banswer\s+set\s+program\b|\bant\s+colony\b|\bant\s+colony\s+optimisation\b|\bartificial\s+bee\s+colony\s+algorithm\b|\bartificial\s+evolution\b|\bartificial\s+intelligence\b|\bartificial\s+neural\s+network\b|\bassociation\s+rule\b|\battention.{1,30}transformer\b|\battention\s+based\b|\bautoencoder\b|\bautomated\s+planning\b|\bautomated\s+scheduling\b|\bautonomic\s+computing\b|\bautonomous\s+car\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bautonomous\s+vehicle\b|\bautonomous\s+weapon\b|\bbackpropagation\b|\bbayes\s+network\b|\bbayesian\s+learning\b|\bbayesian\s+network\b|\bbee\s+colony\b|\bblind\s+signal\s+separation\b|\bbootstrap\s+aggregation\b|\bbrain\s+computer\s+interface\b|\bbrownboost\b|\bclassification\s+tree\b|\bcluster\s+analysis\b|\bcognitive\s+automation\b|\bcognitive\s+computing\b|\bcognitive\s+insight\s+system\b|\bcognitive\s+modelling\b|\bcognitive\s+robotic\b|\bcollaborative\s+filtering\b|\bcollision\s+avoidance\b|\bcommunity\s+detection\b|\bcomputational\s+intelligence\b|\bcomputational\s+pathology\b|\bcomputer\s+vision\b|\bconnectionis\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bconvolutional\s+neural\b|\bconvolutional\s+neural\s+network\b|\bdecision\s+model\b|\bdecision\s+tree\b|\bdeep\s+belief\s+network\b|\bdeep\s+convolutional\s+neural\s+network\b|\bdeep\s+learning\b|\bdeep\s+learning\s+model\b|\bdeep\s+neural\b|\bdeep\s+neural\s+network\b|\bdeep\s+reinforcement\s+learning\b|\bdictionary\s+learning\b|\bdifferential\s+evolution\s+algorithm\b|\bdiffusion\s+model\b|\bdimensionality\s+reduction\b|\bdriverless\s+car\b|\bdriverless\s+vehicle\b|\bdynamic\s+time\s+warping\b|\bembedding\b|\bemotion\s+recognition\b|\bensemble\s+learning\b|\bevolutionary\s+algorithm\b|\bevolutionary\s+computation\b|\bevolutionary\s+robotic\b|\bexpert\s+system\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bextreme\s+machine\s+learning\b|\bface\s+recognition\b|\bfacial\s+expression\s+recognition\b|\bfactorisation\s+machine\b|\bfeature\s+engineering\b|\bfeature\s+extraction\b|\bfeature\s+learning\b|\bfeature\s+selection\b|\bfederated\s+learning\b|\bfirefly\s+algorithm\b|\bfuzzy\s+c\b|\bfuzzy\s+environment\b|\bfuzzy\s+logic\b|\bfuzzy\s+number\b|\bfuzzy\s+set\b|\bfuzzy\s+system\b|\bgaussian\s+mixture\s+model\b|\bgaussian\s+process\b|\bgeneration\s+model\b|\bgenerative\s+adversarial\s+network\b|\bgenerative\s+ai\b|\bgenetic\s+algorithm\b|\bgenetic\s+programming\b|\bgesture\s+recognition\b|\bgradient\s+boosting\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bgradient\s+tree\s+boosting\b|\bgraph\s+convolutional\s+network\b|\bgraph\s+neural\b|\bgraph\s+neural\s+network\b|\bgraphical\s+model\b|\bgravitational\s+search\s+algorithm\b|\bhebbian\s+learning\b|\bhidden\s+markov\s+model\b|\bhierarchical\s+clustering\b|\bhigh\s+dimensional\s+data\b|\bhigh\s+dimensional\s+feature\b|\bhigh\s+dimensional\s+input\b|\bhigh\s+dimensional\s+model\b|\bhough\s+transformer\b|\bhuman\s+action\s+recognition\b|\bhuman\s+activity\s+recognition\b|\bhuman\s+aware\s+artificial\s+intelligence\b|\bhuman\s+robot\s+interaction\b|\bhumanoid\s+robot\b|\bhybrid\s+ai\b|\bimage\s+classification\b|\bimage\s+processing\b|\bimage\s+recognition\b|\bimage\s+retrieval\b|\bimage\s+segmentation\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bindependent\s+component\s+analysis\b|\binductive\s+logic\b|\binductive\s+logic\s+programm\b|\binductive\s+monitoring\b|\binstance\s+based\s+learning\b|\bintelligence\s+augmentation\b|\bintelligent\s+agent\b|\bintelligent\s+classifier\b|\bintelligent\s+geometric\s+computing\b|\bintelligent\s+infrastructure\b|\bintelligent\s+software\s+agent\b|\bintelligent\s+system\b|\bintuitionistic\s+fuzzy\s+set\b|\bk\s+means\b|\bkernel\s+learning\b|\blanguage.{1,30}transformer\b|\blanguage\s+model\b|\blarge\s+language\b|\blarge\s+language\s+model\b|\blatent\s+dirichlet\s+allocation\b|\blatent\s+semantic\s+analysis\b|\blatent\s+variable\b|\blayered\s+control\s+system\b|\blearning.{1,30}transformer\b|\blearning\s+algorithm\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\blearning\s+automata\b|\blearning\s+model\b|\blink\s+prediction\b|\bllm\b|\blogistic\s+regression\b|\blogitboost\b|\blong\s+short\s+term\s+memory\b|\blpboost\b|\bmachine\s+intelligence\b|\bmachine\s+learning\b|\bmachine\s+learning\s+model\b|\bmachine\s+translation\b|\bmachine\s+vision\b|\bmadaboost\b|\bmapreduce\b|\bmarkovian\b|\bmemetic\s+algorithm\b|\bmeta\s+learning\b|\bmodel\s+training\b|\bmotion\s+planning\b|\bmulti\s+agent\s+system\b|\bmulti\s+label\s+classification\b|\bmulti\s+layer\s+perceptron\b|\bmulti\s+objective\s+evolutionary\s+algorithm\b|\bmulti\s+objective\s+optimisation\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bmulti\s+sensor\s+fusion\b|\bmulti\s+task\s+learning\b|\bmultilayer\s+perceptron\b|\bmultinomial\s+naïve\s+bayes\b|\bnaïve\s+bayes\s+classifier\b|\bnatural\s+gradient\b|\bnatural\s+language\b|\bnatural\s+language\s+generation\b|\bnatural\s+language\s+processing\b|\bnatural\s+language\s+understanding\b|\bnearest\s+neighbour\s+algorithm\b|\bneural\s+architecture\s+search\b|\bneural\s+machine\s+translation\b|\bneural\s+network\b|\bneural\s+network\s+cnn\b|\bneural\s+network\s+model\b|\bneural\s+radiance\s+field\b|\bneural\s+turing\b|\bneural\s+turing\s+machine\b|\bneuromorphic\s+architecture\b|\bneuromorphic\s+computing\b|\bneuromorphic\s+engineering\b|\bnon\s+negative\s+matrix\s+factorisation\b|\bobject\s+detection\b|\bobject\s+recognition\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bobstacle\s+avoidance\b|\bone\s+class\s+learning\b|\bone\s+shot\s+learning\b|\boptical\s+character\s+recognition\b|\boptimisation\s+for\s+learning\b|\boptimization\s+for\s+learning\b|\bparticle\s+swarm\s+optimisation\b|\bpattern\s+recognition\b|\bpedestrian\s+detection\b|\bplanning\s+algorithm\b|\bpolicy\s+gradient\s+method\b|\bq\s+learning\b|\brandom\s+field\b|\brandom\s+forest\b|\brankboost\b|\brecommender\s+system\b|\brecurrent\s+neural\s+network\b|\bregression\s+tree\b|\breinforcement\s+learning\b|\brelational\s+learning\b|\brepresentation\s+learning\b|\brobot\s+interaction\b|\brobot\s+learning\b|\brough\s+set\b|\brule\s+based\s+learning\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\brule\s+learning\b|\bself\s+driving\s+car\b|\bself\s+organising\s+map\b|\bself\s+organising\s+structure\b|\bsemi\s+supervised\s+learning\b|\bsemi\s+supervised\s+training\b|\bsensor\s+data\s+fusion\b|\bsensor\s+fusion\b|\bsentiment\s+analysis\b|\bservice\s+robot\b|\bsimilarity\s+learning\b|\bsimultaneous\s+localisation\s+mapping\b|\bsingle\s+linkage\s+clustering\b|\bsparse\s+representation\b|\bspectral\s+clustering\b|\bspeech\s+recognition\b|\bspeech\s+synthesis\b|\bspeech\s+to\s+text\b|\bstacked\s+generalisation\b|\bstatistical\s+relational\s+learning\b|\bstochastic\s+gradient\b|\bstochastic\s+gradient\s+descent\b|\bsupervised\s+learning\b|\bsupervised\s+training\b|\bsupport\s+vector\s+machine\b')
        OR REGEXP_CONTAINS(text_combined, r'(?i)\bsupport\s+vector\s+regression\b|\bswarm\s+intelligence\b|\bswarm\s+optimization\b|\bt\s+s\s+fuzzy\s+system\b|\btakagi\s+sugeno\s+fuzzy\s+system\b|\btemporal\s+difference\s+learning\b|\btext\s+mining\b|\btext\s+to\s+speech\b|\btopic\s+model\b|\btotalboost\b|\btraining\s+data\b|\btrajectory\s+planning\b|\btrajectory\s+tracking\b|\btransfer\s+learning\b|\btrust\s+region\s+policy\s+optimisation\b|\bunmanned\s+aerial\s+vehicle\b|\bunsupervised\s+learning\b|\bunsupervised\s+training\b|\bvariational\s+inference\b|\bvconvolutional\s+neural\b|\bvector\s+machine\b|\bvirtual\s+assistant\b|\bvisual\s+servoing\b|\bxgboost\b')
      )
    ) AS is_related_ai
  FROM deduped_patents
),
classified_patents AS (
  SELECT
    ticker,
    cik,
    company_name,
    publication_number,
    filing_year,
    publication_year,
    is_core_ai,
    is_related_ai,
    (is_core_ai OR is_related_ai) AS is_ai_oecd
  FROM patent_evaluation
)
-- Aggregate by firm and filing year
SELECT
  ticker,
  cik,
  company_name,
  filing_year AS year,
  COUNT(DISTINCT publication_number) AS total_patents,
  COUNT(DISTINCT IF(is_ai_oecd, publication_number, NULL)) AS ai_patents_oecd,
  COUNT(DISTINCT IF(is_core_ai, publication_number, NULL)) AS ai_patents_core,
  COUNT(DISTINCT IF(is_related_ai, publication_number, NULL)) AS ai_patents_related,
  ROUND(SAFE_DIVIDE(
    COUNT(DISTINCT IF(is_ai_oecd, publication_number, NULL)),
    COUNT(DISTINCT publication_number)
  ), 4) AS ai_patent_intensity
FROM classified_patents
WHERE filing_year BETWEEN 2015 AND 2026
GROUP BY 1, 2, 3, 4
ORDER BY ticker, year
