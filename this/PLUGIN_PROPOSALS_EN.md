# XplagiaX — Proposed Plugin Architecture Expansion

**Version:** v3.5+ Plugin Roadmap | **Date:** 2026-03-21 | **Author:** Pavel Santos Nunez

---

## Current Coverage Map

| Active Plugin | Detection Target | Known Blind Spot |
|---|---|---|
| `detector_final.py` | Raw AI text (41 model classes via ModernBERT ensemble) | Paraphrased/obfuscated AI text |
| `stylometric_profiler.py` | Writing style fingerprint (burstiness, vocabulary richness, hapax) | Single-document only, no author baseline comparison |
| `hallucination_profile.py` | Veracity anomalies (25-dim zero-resource feature vector) | No external fact-checking; statistical signals only |
| `reasoning_profiler.py` | Chain-of-thought / reasoning model traces (o1, R1, QwQ) | Only catches explicit CoT patterns; misses implicit reasoning |
| `watermark_decoder.py` | Statistical watermarks (Kirchenbauer et al. scheme) | Only known schemes; experimental, noisy |

---

## Proposed Plugins

### P0 — Critical Priority (Anti-Evasion)

#### 1. `paraphrase_detector.py` — Obfuscation & Rewrite Detection

**Problem:** QuillBot, Undetectable.ai, HIX Bypass, and manual rewording are the #1 evasion technique in K-12 and post-secondary settings. The current ModernBERT ensemble catches raw AI output with high accuracy but drops to near-random performance on paraphrased text. Research shows DIPPER-style paraphrasing reduces DetectGPT accuracy from 70.3% to 4.6%.

**Why This Is Necessary:** Without this plugin, XplagiaX has a critical architectural blind spot that any student can exploit in under 30 seconds. A student generates text with ChatGPT, pastes it into QuillBot (free tier), and the paraphrased output bypasses the ModernBERT ensemble entirely. This isn't a theoretical gap — it's the most common evasion workflow documented in academic integrity literature (Krishna et al., NeurIPS 2023). Every commercial competitor that fails to address paraphrasing becomes irrelevant within one semester of student adoption. For XplagiaX to be credible in the SD5 pilot and any institutional sale, this plugin is non-negotiable. The 4-model ensemble investment is wasted if the output can be trivially circumvented by a free browser extension. This plugin transforms XplagiaX from "catches raw AI text" to "catches AI involvement regardless of post-processing" — a fundamentally different product category.

**What it detects:** Statistical signatures that survive paraphrasing — clause-depth distribution, synonym substitution density anomalies (paraphrase tools over-substitute), sentence-start diversity, transition word entropy, passive/active ratio stability, paragraph template repetition.

**Feature vector (~12 dimensions):**
- Clause-depth distribution (mean, std)
- Synonym density anomaly score
- Sentence-start bigram diversity
- Transition word entropy
- Passive-to-active ratio stability
- Paragraph structural template repetition score
- Lexical cohesion gradient (inter-sentence)
- Hedging-to-assertion ratio uniformity
- Function word distribution fingerprint (4 features)

**Integration pattern:**
```
PluginOrchestrator → .vectorize(text) → ParaphraseClassifier
→ additional_analyses["paraphrase"]
→ ForensicReportGenerator → HTML section
```

**Dependency:** CPU-only, zero external models. Optional: DistilBERT embeddings for semantic cohesion features.

---

#### 2. `authorship_drift_profiler.py` — Longitudinal Style Comparison

**Problem:** For repeated submissions (e.g., KCA use case, SD5 pilot), comparing a new document against the student's established baseline is the strongest integrity signal. A sudden style shift — vocabulary jump, formality register change, sentence complexity spike — indicates external authorship.

**Why This Is Necessary:** Single-document analysis has a fundamental ceiling — it can only compare the text against generic AI/human population statistics. It cannot answer the most critical question an instructor asks: "Did THIS student write THIS text?" Authorship drift is the only signal that provides per-student contextualization. In the KCA case with Colleen Morrison, the existing detector flagged the text as AI but could not demonstrate that the writing was inconsistent with the student's prior work. With a baseline profile built from 3-5 previous assignments, drift detection produces evidence that is intuitive and defensible in parent-teacher meetings: "This submission's vocabulary sophistication is 2.4 standard deviations above your child's established writing profile." This converts XplagiaX from a probabilistic classifier into an authorship verification system — a category upgrade that directly supports MarkTrack Pro's keystroke-based authorship thesis. Schools that adopt XplagiaX for ongoing monitoring (not one-shot detection) need this capability. It also creates a competitive moat: the baseline requirement means switching costs increase with every assignment processed.

**What it detects:** Cosine distance between the submission's stylometric vector and the student's established profile. Identifies WHERE in the text the style diverges (paragraph-level drift curve).

**Implementation:** Wraps existing `StylometricProfiler.build_profile()` and `compare_texts()` into the pipeline. Outputs a per-paragraph drift score and an overall divergence index.

**Feature output:**
- Overall cosine distance from baseline
- Per-paragraph drift curve (array of distances)
- Divergence breakpoints (positions where style shifts significantly)
- Top 5 features contributing to divergence
- Risk classification: CONSISTENT / MILD DRIFT / SIGNIFICANT DIVERGENCE

**Integration pattern:**
```
PluginOrchestrator.__init__(baseline_profile=StyleProfile)
→ .compare(text, baseline) → drift_result
→ additional_analyses["authorship_drift"]
→ ForensicReportGenerator → HTML section with drift chart
```

**Dependency:** Requires a pre-built `StyleProfile` per author. CPU-only.

---

### P1 — High Value

#### 3. `reference_validator.py` — Citation & Source Existence Checker

**Problem:** AI fabricates citations in 18-55% of cases depending on model version. A fabricated citation is binary, verifiable proof of AI involvement — the most actionable signal for academic integrity officers.

**Why This Is Necessary:** Every other detection signal in XplagiaX is probabilistic — "this text is 87% likely AI-generated." Probabilistic verdicts create room for dispute ("my writing style is just formal"). A fabricated citation is binary and irrefutable: the paper either exists in CrossRef or it doesn't. There is no gray area, no argument, no appeal. Research from NeurIPS 2025 found over 100 hallucinated citations across 53 accepted papers that passed peer review by 3+ reviewers (Goldman, 2026). In K-12 and post-secondary settings, citation fabrication rates reach 39-55% with ChatGPT 3.5 and 18-28% with GPT-4 (JMIR, 2024). For XplagiaX's target market — academic integrity officers who need defensible evidence for formal proceedings — a reference validator provides the strongest possible proof artifact. It also addresses a gap no competitor in the K-12 space currently fills: Turnitin checks for plagiarism against existing sources but does not verify whether cited sources actually exist. This is a clear differentiation point for XplagiaX sales conversations.

**What it detects:**
- Non-existent references (phantom citations)
- Incorrect DOIs, volume/issue numbers, page ranges
- Chimeric references (elements from multiple real papers merged into one)
- Author-title mismatches
- Date anachronisms (citing future publications)

**Implementation:** Regex + heuristic extraction for APA/MLA/Chicago formats. Validation via CrossRef API (free, 50 req/sec) and OpenAlex API (fully open). Returns: `exists` / `not_found` / `ambiguous` / `chimeric` per citation.

**Dependency:** Requires HTTP access to CrossRef/OpenAlex. Extraction is CPU-only.

---

#### 4. `register_consistency_profiler.py` — Intra-Document Register Analysis

**Problem:** AI-assisted writing often shows unnatural formality shifts: casual intro → hyper-formal body → casual conclusion, or sudden vocabulary sophistication jumps mid-paragraph. This is distinct from `stylometric_profiler` (which profiles the whole document) — this plugin profiles segments and measures internal consistency.

**Why This Is Necessary:** The most common real-world AI misuse pattern is not "100% AI-generated essay" but "human-written intro and conclusion with AI-generated body paragraphs." Students write the first and last paragraphs themselves, then paste ChatGPT output for the analytical middle. The current pipeline's whole-document metrics average these segments together, diluting the signal. Register consistency analysis detects exactly this pattern by measuring formality, vocabulary sophistication, and syntactic complexity at the paragraph level and flagging internal inconsistencies. A document where paragraph 1 scores a Flesch-Kincaid grade of 8 and paragraph 3 scores 14 has a register shift that is invisible to whole-document analysis but obvious to segment-level profiling. This plugin also catches a second pattern: students who use ChatGPT to "improve" their draft — the AI polishes some paragraphs to a level of formality inconsistent with the student's own sections. For instructors, the visual output (a per-paragraph formality curve showing the abrupt jump) is immediately intuitive and requires no statistical literacy to interpret.

**Feature vector (~8 dimensions):**
- Heylighen F-score per paragraph (formality index)
- Vocabulary sophistication index per segment
- Syntactic complexity gradient
- Hedging density variance across sections
- Readability grade variance (Flesch-Kincaid per paragraph)
- Register shift breakpoints
- Formality coefficient of variation
- Clause subordination depth variance

**Dependency:** CPU-only. Optional: spaCy for syntactic depth.

---

#### 5. `prompt_signature_profiler.py` — Prompt Pattern Reconstruction

**Problem:** Certain prompt patterns leave detectable traces. "Write an essay about X in 500 words" produces text that mirrors the prompt structure — exactly 5 paragraphs, topic sentence density, conclusion-mirrors-intro.

**Why This Is Necessary:** Even when paraphrasing tools successfully scramble the lexical surface of AI text, the structural fingerprint of the original prompt survives. A 5-paragraph essay with perfectly balanced 100-word paragraphs, each starting with a topic sentence, and a conclusion that mirrors the introduction is not how humans naturally write — it's how ChatGPT responds to "Write a 500-word essay about X." This structural signature is orthogonal to every other detection signal in the pipeline: the ModernBERT ensemble detects token-level AI patterns, the stylometric profiler measures vocabulary features, the reasoning profiler catches CoT markers, but none of them detect prompt-level structural conformance. This plugin fills a unique analytical dimension. It is also the hardest signal for students to defeat — even if they paraphrase every sentence and swap vocabulary, the 5-paragraph template, the enumeration precision, and the meta-discourse markers ("In this essay we will explore...") persist because changing document structure requires genuine compositional effort that most evasion tools don't perform. For K-12 specifically, prompt signatures are highly prevalent because students use simple, template-inducing prompts.

**Feature vector (~10 dimensions):**
- Template compliance score (5-paragraph essay, list-of-N, pros-and-cons)
- Enumeration precision (does text enumerate exactly N items when topic doesn't naturally decompose?)
- Topic-sentence density per paragraph
- Conclusion-mirrors-intro ratio
- Paragraph count vs. topic complexity mismatch
- Instructional echo detection (residual prompt language)
- Word count conformance to round numbers
- Section balance uniformity (all paragraphs ~same length)
- Meta-discourse ratio ("In this essay", "In conclusion")
- Greeting/sign-off absence score

**Dependency:** CPU-only.

---

### P2 — Strategic Additions

#### 6. `multilingual_transfer_detector.py` — Cross-Lingual Evasion Detection

**Problem:** Evasion technique: generate in English → translate to target language. Translation artifacts are detectable: unnatural collocations, calque patterns, preposition misuse, article placement anomalies. Critical for LCI Education's multilingual network.

**Why This Is Necessary:** LCI Education operates across multiple countries and languages. The current pipeline is English-centric — the ModernBERT ensemble was trained on English corpora, and the stylometric/reasoning profilers use English stopword lists and English-language regex patterns. Students in French, Spanish, or Portuguese programs can generate text in English (where AI models are strongest), run it through Google Translate or DeepL, and submit the translation. The current pipeline has zero detection capability against this workflow. Translation artifacts are linguistically distinct from paraphrasing artifacts: calque patterns (literal translations of English idioms that don't exist in the target language), preposition misuse (English preposition logic applied to romance languages), and unnatural article placement are all detectable with language-specific feature extractors. For XplagiaX to be viable as an enterprise product for LCI's global network — not just the English-language campuses — this plugin is required. It also opens the Latin American and European education markets where AI detection tools are even scarcer than in North America.

**Feature vector:** Collocation naturalness score, preposition distribution anomaly, article usage patterns, calque density, false cognate frequency.

**Dependency:** spaCy with multilingual models.

---

#### 7. `metadata_forensics.py` — Document Metadata Analysis

**Problem:** Analyzes .docx/.pdf container metadata rather than text content. Detects: creation-to-last-modified gap (copy-paste indicator), application signature, edit time vs. word count ratio, revision count.

**Why This Is Necessary:** All other plugins in XplagiaX analyze text content. Metadata forensics operates on a completely independent evidence channel — the document container itself. A .docx file whose creation timestamp is 11:42 PM, last-modified timestamp is 11:43 PM, has 1 revision, and contains 2,000 words was not written by a human — it was pasted. A document created by "Google Docs" but submitted as .docx has a different metadata fingerprint than one created in Microsoft Word. A PDF generated directly by a Python script (common in AI pipelines) carries application metadata like "ReportLab" or "FPDF" instead of "Microsoft Word" or "LibreOffice." These signals are invisible to text analysis and impossible to spoof without specialized tools that students don't typically possess. Metadata evidence is also highly compelling in formal proceedings because it is objective and non-probabilistic: the timestamp either shows a 1-minute edit window or it doesn't. Combined with MarkTrack Pro's keystroke tracking, metadata forensics creates a multi-layered behavioral evidence chain that goes far beyond content analysis alone.

**Dependency:** python-docx, PyPDF2. No ML models.

---

#### 8. `perplexity_profiler.py` — Reference-LM Perplexity Analysis

**Problem:** Per-token perplexity curve using GPT-2 small as reference LM. Currently the pipeline approximates perplexity via statistical features. A proper burstiness-aware perplexity curve is a stronger signal.

**Why This Is Necessary:** Perplexity — how "surprised" a language model is by each token — is the foundational signal underlying DetectGPT, Fast-DetectGPT, Binoculars, and most zero-shot detection methods. The current XplagiaX pipeline approximates perplexity indirectly through statistical proxies (entropy, burstiness), but these are coarse-grained and lose the per-token granularity that makes perplexity curves so powerful. A proper perplexity profiler produces a curve showing exactly WHERE in the text the language model is unsurprised (low perplexity = likely AI-generated) versus surprised (high perplexity = likely human-written). This per-token resolution enables hybrid detection: even in a document where 70% is human-written, the 30% AI-generated section will show a distinct perplexity valley that paragraph-level metrics miss. The perplexity curve is also one of the most interpretable signals for educators — "this section is highly predictable to a language model, meaning it follows the statistical patterns AI models produce" is a clear explanation. Adding this plugin also makes XplagiaX's detection stack comparable to the research-grade systems (DetectGPT, Binoculars) used in peer-reviewed AI detection papers, strengthening the product's academic credibility.

**Dependency:** GPT-2 small (~500MB). GPU recommended.

---

## Implementation Roadmap

| Phase | Plugins | Timeline | ROI |
|---|---|---|---|
| Phase 1 | `paraphrase_detector`, `authorship_drift_profiler` | Immediate | Highest — closes #1 evasion gap |
| Phase 2 | `reference_validator`, `register_consistency_profiler` | Next sprint | High — binary proof for citations |
| Phase 3 | `prompt_signature_profiler`, `metadata_forensics`, `multilingual_transfer_detector` | Q2 2026 | Medium — differentiation |
| Phase 4 | `perplexity_profiler` | When GPU budget allows | Medium — enhanced detection |

---

## PluginConfig Extension

```python
@dataclass
class PluginConfig:
    # ... existing flags ...
    enable_paraphrase:       bool = True    # P0 — anti-evasion
    enable_authorship_drift: bool = False   # P0 — requires baseline
    enable_reference_check:  bool = False   # P1 — requires network
    enable_register:         bool = True    # P1 — formality analysis
    enable_prompt_signature: bool = True    # P1 — prompt trace
    enable_metadata:         bool = False   # P2 — requires file path
    enable_multilingual:     bool = False   # P2 — requires spaCy multi
    enable_perplexity:       bool = False   # P2 — requires GPT-2
    baseline_profile:        Optional[Any] = None  # for drift comparison
```
