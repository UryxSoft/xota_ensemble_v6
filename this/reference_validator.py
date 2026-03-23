"""
reference_validator.py  (XplagiaX Plugin — v1.0)
===================================================
Citation existence verification and fabrication detection.

Research Basis
──────────────
  - CheckIfExist (2026): Cascade validation via CrossRef/S2/OpenAlex
  - SemanticCite (2025): Semantic claim verification
  - Spinellis (2025): Ornamental reference detection
  - JMIR (2024): ChatGPT citation fabrication rates (18-55%)

Architecture
────────────
  1. ReferenceExtractor: Regex parser for APA/MLA/Chicago/inline refs
  2. ReferenceValidator: Cascade API validation (CrossRef → S2 → OpenAlex)
  3. ChimeraDetector: Cross-database author mismatch detection
  4. OrnamentalDetector: Bibliography vs inline citation cross-check
  5. ReferenceRiskClassifier: Severity profile for ForensicReportGenerator

Interface Contract (matches PluginOrchestrator pattern)
───────────────────────────────────────────────────────
  validator = ReferenceValidator()
  stats     = validator.compute_stats(text)      # → Dict[str, Any]
  vec       = validator.vectorize(text)           # → np.ndarray shape (12,)
  names     = validator.feature_names()           # → tuple of 12 strings

Dependency: Requires HTTP access to CrossRef/OpenAlex APIs.
            Extraction is CPU-only. Validation requires network.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# VECTOR SCHEMA
# ============================================================================

REFERENCE_VECTOR_DIM: int = 12

_VECTOR_SCHEMA: Tuple[Tuple[str, int], ...] = (
    ("total_references",        0),
    ("verified_ratio",          1),   # score > 80
    ("fabricated_ratio",        2),   # score < 50 (ghost citations)
    ("chimeric_ratio",          3),   # title match + author mismatch
    ("partial_match_ratio",     4),   # score 50-80
    ("ornamental_ratio",        5),   # in bib but no inline citation
    ("mean_confidence",         6),   # avg confidence score [0-100]
    ("min_confidence",          7),   # worst reference score
    ("author_mismatch_count",   8),   # suspicious authors detected
    ("doi_fabrication_count",   9),   # invalid/fabricated DOIs
    ("date_anomaly_count",     10),   # future or impossible years
    ("missing_inline_ratio",   11),   # refs in bib without inline cites
)

FEATURE_NAMES: Tuple[str, ...] = tuple(name for name, _ in _VECTOR_SCHEMA)

assert len(_VECTOR_SCHEMA) == REFERENCE_VECTOR_DIM
assert sorted(idx for _, idx in _VECTOR_SCHEMA) == list(range(REFERENCE_VECTOR_DIM))


# ============================================================================
# CONSTANTS
# ============================================================================

_CROSSREF_API    = "https://api.crossref.org/works"
_OPENALEX_API    = "https://api.openalex.org/works"
_S2_API          = "https://api.semanticscholar.org/graph/v1/paper/search"

_RATE_LIMIT_MS   = 800    # 800ms between API calls (CheckIfExist spec)
_MAX_CANDIDATES  = 3      # Top N candidates from each API
_REQUEST_TIMEOUT = 10     # seconds

# Confidence score thresholds (from CheckIfExist paper)
_VERIFIED_THRESHOLD   = 80   # score > 80 → verified match
_FALLBACK_THRESHOLD   = 70   # score < 70 → trigger cascade fallback
_EXISTS_THRESHOLD     = 50   # score > 50 → exists with issues

# Chimera detection thresholds
_TITLE_MATCH_HIGH     = 0.80  # title similarity > 80% to consider a match
_AUTHOR_MISMATCH_THR  = 0.90  # author similarity < 90% → chimera suspect

# Penalties (from CheckIfExist paper)
_PENALTY_TITLE_MISMATCH   = -20
_PENALTY_AUTHOR_MISMATCH  = -20
_PENALTY_JOURNAL_MISMATCH = -15
_PENALTY_FALSE_AUTHOR     = -15
_BONUS_CROSS_VALIDATED    = +10

_CURRENT_YEAR = datetime.now().year

# User-Agent for API requests (polite crawling)
_USER_AGENT = "XplagiaX-ReferenceValidator/1.0 (mailto:research@xplagiax.com)"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ParsedReference:
    """A reference extracted from text."""
    raw_text: str
    authors: List[str] = field(default_factory=list)
    title: str = ""
    year: Optional[int] = None
    journal: str = ""
    volume: str = ""
    pages: str = ""
    doi: str = ""
    position: int = 0            # position in the reference list
    is_inline: bool = False      # True if extracted from inline citation
    has_inline_citation: bool = False  # True if cited in the text body


@dataclass
class ValidationResult:
    """Result of validating a single reference."""
    reference: ParsedReference
    confidence_score: float = 0.0        # 0-100
    status: str = "not_checked"          # verified|partial|not_found|chimeric|error
    matched_title: str = ""
    matched_authors: List[str] = field(default_factory=list)
    matched_doi: str = ""
    matched_year: Optional[int] = None
    matched_journal: str = ""
    suspicious_authors: List[str] = field(default_factory=list)
    confirmed_authors: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    sources_checked: List[str] = field(default_factory=list)
    is_chimeric: bool = False
    is_ornamental: bool = False
    doi_valid: bool = True
    date_anomaly: bool = False


# ============================================================================
# STRING SIMILARITY (Levenshtein — from CheckIfExist methodology)
# ============================================================================

def _levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Levenshtein similarity ratio [0, 1].
    1.0 = identical, 0.0 = completely different.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    max_len = max(len1, len2)
    return 1.0 - (matrix[len1][len2] / max_len) if max_len > 0 else 1.0


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison (CheckIfExist methodology):
    lowercase + remove non-alphanumeric characters.
    """
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()


def _extract_surnames(author_str: str) -> List[str]:
    """
    Extract surnames from author string.
    Handles: "Smith, J.", "J. Smith", "Smith et al.", "Smith, John and Doe, Jane"
    """
    # Remove "et al." and similar
    cleaned = re.sub(r'\bet\s+al\.?', '', author_str, flags=re.IGNORECASE)

    # Split on common delimiters
    parts = re.split(r'\s*(?:,\s*(?:and|&)\s*|;\s*|\s+and\s+|\s*&\s*)', cleaned)

    surnames = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # "Smith, J." → Smith
        if ',' in part:
            surname = part.split(',')[0].strip()
        else:
            # "J. Smith" or "John Smith" → take last word
            words = part.split()
            surname = words[-1] if words else part
        surname = re.sub(r'[^a-zA-Z\-]', '', surname)
        if len(surname) >= 2:
            surnames.append(surname.lower())

    return surnames


# ============================================================================
# REFERENCE EXTRACTOR
# ============================================================================

class ReferenceExtractor:
    """
    Extract references from text using regex patterns.

    Supports: APA, MLA, Chicago, numbered references, inline citations.
    Does NOT use GROBID or anystyle (pure regex, CPU-only).
    """

    # ── Bibliography section patterns ─────────────────────────────
    # Match "References", "Bibliography", "Works Cited", etc.
    _BIB_HEADER = re.compile(
        r'^\s*(?:References|Bibliography|Works?\s+Cited|Literature|'
        r'Bibliograf[íi]a|R[ée]f[ée]rences)\s*$',
        re.IGNORECASE | re.MULTILINE,
    )

    # APA-style: Author, A. B. (2023). Title. Journal, vol(num), pages.
    _APA_REF = re.compile(
        r'([A-Z][a-zA-Z\-]+(?:\s*,\s*[A-Z]\.(?:\s*[A-Z]\.)*)'
        r'(?:\s*(?:,|&|and)\s*[A-Z][a-zA-Z\-]+(?:\s*,\s*[A-Z]\.(?:\s*[A-Z]\.)*)*)*'
        r'(?:\s*et\s+al\.?)?)'       # Authors
        r'\s*\((\d{4})\)\.\s*'        # (Year).
        r'([^.]+\.)',                  # Title.
        re.MULTILINE,
    )

    # Numbered reference: [1] Author... (year)... Title...
    _NUMBERED_REF = re.compile(
        r'\[(\d+)\]\s*(.+?)(?:\n\n|\n(?=\[\d+\])|\Z)',
        re.DOTALL,
    )

    # DOI pattern
    _DOI = re.compile(
        r'(?:doi:\s*|https?://doi\.org/)(10\.\d{4,}/[^\s,;]+)',
        re.IGNORECASE,
    )

    # Inline citations: (Smith, 2023), (Smith & Doe, 2023), (Smith et al., 2023)
    _INLINE_CITATION = re.compile(
        r'\(([A-Z][a-zA-Z\-]+(?:\s*(?:&|and|et\s+al\.?|,)\s*'
        r'[A-Z]?[a-zA-Z\-]*)*\s*,?\s*\d{4}[a-z]?)\)',
    )

    # Year extraction
    _YEAR = re.compile(r'\b((?:19|20)\d{2})\b')

    def extract(self, text: str) -> Tuple[List[ParsedReference], List[str]]:
        """
        Extract references from text.

        Returns
        -------
        (bibliography_refs, inline_citations)
            bibliography_refs: ParsedReference objects from the reference section
            inline_citations: raw inline citation strings found in text body
        """
        refs = []
        inline_cites = []

        # ── Find bibliography section ─────────────────────────────
        bib_match = self._BIB_HEADER.search(text)
        if bib_match:
            bib_text = text[bib_match.end():]
            body_text = text[:bib_match.start()]
        else:
            # No explicit bibliography header — try the last 30% of text
            split_point = int(len(text) * 0.7)
            bib_text = text[split_point:]
            body_text = text

        # ── Extract APA-style references ──────────────────────────
        for i, m in enumerate(self._APA_REF.finditer(bib_text)):
            authors_str, year_str, title = m.group(1), m.group(2), m.group(3)
            doi_match = self._DOI.search(m.group(0))
            ref = ParsedReference(
                raw_text=m.group(0).strip(),
                authors=_extract_surnames(authors_str),
                title=title.strip().rstrip('.'),
                year=int(year_str),
                doi=doi_match.group(1) if doi_match else "",
                position=i,
            )
            refs.append(ref)

        # ── Extract numbered references if APA found nothing ──────
        if not refs:
            for m in self._NUMBERED_REF.finditer(bib_text):
                raw = m.group(2).strip()
                year_m = self._YEAR.search(raw)
                doi_m = self._DOI.search(raw)
                ref = ParsedReference(
                    raw_text=raw,
                    year=int(year_m.group(1)) if year_m else None,
                    doi=doi_m.group(1) if doi_m else "",
                    position=int(m.group(1)),
                )
                # Try to extract title (first sentence-like chunk)
                title_m = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', raw)
                if title_m:
                    ref.title = title_m.group(1).strip()
                else:
                    # Take longest phrase-like segment
                    sentences = re.split(r'[.!?]', raw)
                    if sentences:
                        ref.title = max(sentences, key=len).strip()[:200]
                refs.append(ref)

        # ── Extract inline citations from body ────────────────────
        for m in self._INLINE_CITATION.finditer(body_text):
            inline_cites.append(m.group(1).strip())

        # ── Cross-reference: mark which bib refs have inline cites ─
        for ref in refs:
            ref.has_inline_citation = self._has_inline_match(ref, inline_cites, body_text)

        return refs, inline_cites

    @staticmethod
    def _has_inline_match(ref: ParsedReference, inline_cites: List[str],
                          body_text: str) -> bool:
        """Check if a bibliography ref is cited in the text body."""
        if not ref.authors:
            return True  # can't verify, assume cited
        primary_author = ref.authors[0] if ref.authors else ""
        year_str = str(ref.year) if ref.year else ""

        # Check inline citations
        for cite in inline_cites:
            cite_lower = cite.lower()
            if primary_author in cite_lower:
                if not year_str or year_str in cite:
                    return True

        # Check body text for author name near year
        if primary_author:
            pattern = re.compile(
                rf'\b{re.escape(primary_author)}\b.*?\b{re.escape(year_str)}\b',
                re.IGNORECASE | re.DOTALL,
            )
            if pattern.search(body_text[:5000]):  # only check first 5000 chars
                return True

        return False


# ============================================================================
# API CLIENT (CrossRef, Semantic Scholar, OpenAlex)
# ============================================================================

class _APIClient:
    """
    HTTP client for academic APIs with rate limiting.

    Implements the CheckIfExist cascade: CrossRef → S2 → OpenAlex.
    Rate limited to 800ms between requests (CheckIfExist spec).
    """

    def __init__(self, rate_limit_ms: int = _RATE_LIMIT_MS,
                 timeout: int = _REQUEST_TIMEOUT) -> None:
        self._rate_limit = rate_limit_ms / 1000.0
        self._timeout = timeout
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get_json(self, url: str) -> Optional[dict]:
        """GET request returning parsed JSON, or None on failure."""
        self._throttle()
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                TimeoutError, OSError) as exc:
            logger.debug("API request failed: %s → %s", url[:100], exc)
            return None

    def search_crossref(self, query: str, rows: int = _MAX_CANDIDATES
                        ) -> List[dict]:
        """Search CrossRef for matching works."""
        q = urllib.parse.quote(query[:300])
        url = f"{_CROSSREF_API}?query={q}&rows={rows}&select=DOI,title,author,container-title,published-print,published-online"
        data = self._get_json(url)
        if data and "message" in data and "items" in data["message"]:
            return data["message"]["items"][:rows]
        return []

    def search_openalex(self, query: str, rows: int = _MAX_CANDIDATES
                        ) -> List[dict]:
        """Search OpenAlex for matching works."""
        q = urllib.parse.quote(query[:300])
        url = f"{_OPENALEX_API}?search={q}&per_page={rows}"
        data = self._get_json(url)
        if data and "results" in data:
            return data["results"][:rows]
        return []

    def search_semantic_scholar(self, query: str, rows: int = _MAX_CANDIDATES
                                 ) -> List[dict]:
        """Search Semantic Scholar for matching papers."""
        q = urllib.parse.quote(query[:300])
        url = f"{_S2_API}?query={q}&limit={rows}&fields=title,authors,year,externalIds,venue"
        data = self._get_json(url)
        if data and "data" in data:
            return data["data"][:rows]
        return []

    def lookup_doi(self, doi: str) -> Optional[dict]:
        """Look up a specific DOI on CrossRef."""
        url = f"{_CROSSREF_API}/{urllib.parse.quote(doi)}"
        data = self._get_json(url)
        if data and "message" in data:
            return data["message"]
        return None


# ============================================================================
# METADATA NORMALIZERS (extract comparable fields from API responses)
# ============================================================================

def _crossref_to_meta(item: dict) -> dict:
    """Normalize CrossRef API response to standard metadata dict."""
    title = ""
    if item.get("title"):
        title = item["title"][0] if isinstance(item["title"], list) else item["title"]

    authors = []
    for a in item.get("author", []):
        surname = a.get("family", "")
        if surname:
            authors.append(surname.lower())

    year = None
    for date_field in ["published-print", "published-online", "created"]:
        dp = item.get(date_field, {}).get("date-parts", [[]])
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break

    journal = ""
    ct = item.get("container-title", [])
    if ct:
        journal = ct[0] if isinstance(ct, list) else ct

    return {
        "title": title, "authors": authors, "year": year,
        "journal": journal, "doi": item.get("DOI", ""),
    }


def _openalex_to_meta(item: dict) -> dict:
    """Normalize OpenAlex API response."""
    title = item.get("display_name", "") or item.get("title", "")
    authors = []
    for authorship in item.get("authorships", []):
        a = authorship.get("author", {})
        name = a.get("display_name", "")
        if name:
            parts = name.split()
            if parts:
                authors.append(parts[-1].lower())

    year = item.get("publication_year")
    journal = ""
    loc = item.get("primary_location", {})
    if loc:
        src = loc.get("source", {})
        if src:
            journal = src.get("display_name", "")

    doi = item.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    return {"title": title, "authors": authors, "year": year,
            "journal": journal, "doi": doi}


def _s2_to_meta(item: dict) -> dict:
    """Normalize Semantic Scholar API response."""
    title = item.get("title", "")
    authors = []
    for a in item.get("authors", []):
        name = a.get("name", "")
        if name:
            parts = name.split()
            if parts:
                authors.append(parts[-1].lower())

    year = item.get("year")
    journal = item.get("venue", "")
    doi = ""
    ext = item.get("externalIds", {})
    if ext:
        doi = ext.get("DOI", "")

    return {"title": title, "authors": authors, "year": year,
            "journal": journal, "doi": doi}


# ============================================================================
# CONFIDENCE SCORER (CheckIfExist algorithm)
# ============================================================================

def _compute_confidence(ref: ParsedReference, candidate: dict) -> Tuple[float, List[str]]:
    """
    Compute confidence score [0-100] for a reference-candidate match.

    Implements CheckIfExist scoring with Levenshtein similarity
    and penalty system.

    Returns (score, issues_list).
    """
    issues = []
    score = 0.0

    # ── Title similarity ──────────────────────────────────────────
    ref_title = _normalize_text(ref.title)
    cand_title = _normalize_text(candidate.get("title", ""))
    title_sim = _levenshtein_ratio(ref_title, cand_title) if ref_title and cand_title else 0.0
    score += title_sim * 40  # title worth 40 points

    if title_sim < 0.5:
        issues.append(f"Title mismatch (similarity: {title_sim:.0%})")
        score += _PENALTY_TITLE_MISMATCH

    # ── Author similarity ─────────────────────────────────────────
    ref_authors = set(ref.authors)
    cand_authors = set(candidate.get("authors", []))

    if ref_authors and cand_authors:
        matched = ref_authors & cand_authors
        author_sim = len(matched) / max(len(ref_authors), 1)
        score += author_sim * 30  # authors worth 30 points

        if author_sim < _AUTHOR_MISMATCH_THR and title_sim > _TITLE_MATCH_HIGH:
            issues.append(f"Possible chimeric reference: title matches but authors differ")
            score += _PENALTY_AUTHOR_MISMATCH

        # Identify suspicious authors
        unmatched = ref_authors - cand_authors
        for ua in unmatched:
            issues.append(f"Suspicious author: '{ua}' not found in matched work")
            score += _PENALTY_FALSE_AUTHOR
    elif ref_authors:
        issues.append("Could not verify authors (no author data from API)")
        score += 10  # partial credit

    # ── Year similarity ───────────────────────────────────────────
    ref_year = ref.year
    cand_year = candidate.get("year")
    if ref_year and cand_year:
        if ref_year == cand_year:
            score += 15  # year worth 15 points
        elif abs(ref_year - cand_year) <= 1:
            score += 10
            issues.append(f"Year off by 1 ({ref_year} vs {cand_year})")
        else:
            issues.append(f"Year mismatch ({ref_year} vs {cand_year})")
            score += 5

    # ── Journal similarity ────────────────────────────────────────
    ref_journal = _normalize_text(ref.journal)
    cand_journal = _normalize_text(candidate.get("journal", ""))
    if ref_journal and cand_journal:
        journal_sim = _levenshtein_ratio(ref_journal, cand_journal)
        score += journal_sim * 15  # journal worth 15 points
        if journal_sim < 0.5:
            issues.append(f"Journal mismatch (similarity: {journal_sim:.0%})")
            score += _PENALTY_JOURNAL_MISMATCH

    return max(0.0, min(100.0, score)), issues


# ============================================================================
# REFERENCE VALIDATOR — Main Plugin Class
# ============================================================================

class ReferenceValidator:
    """
    Validate academic references against CrossRef/OpenAlex/Semantic Scholar.

    Implements CheckIfExist cascade architecture:
    1. Extract references from text
    2. Search CrossRef for top 3 candidates
    3. Score each candidate with Levenshtein + penalties
    4. If score < 70, fallback to Semantic Scholar + OpenAlex
    5. Cross-validate authors across databases
    6. Detect chimeric references, ornamental references, date anomalies

    Usage::

        validator = ReferenceValidator()

        # Full analysis (for PluginOrchestrator)
        stats = validator.compute_stats(text)

        # Feature vector (for ML)
        vec = validator.vectorize(text)
    """

    __slots__ = ("_extractor", "_api", "_enable_network", "_min_refs")

    def __init__(self, enable_network: bool = True,
                 rate_limit_ms: int = _RATE_LIMIT_MS,
                 min_refs: int = 1) -> None:
        """
        Parameters
        ----------
        enable_network : If False, only extracts references without API validation.
                         Useful for testing or offline analysis.
        rate_limit_ms  : Milliseconds between API calls (default: 800).
        min_refs       : Minimum references needed for meaningful analysis.
        """
        self._extractor = ReferenceExtractor()
        self._api = _APIClient(rate_limit_ms=rate_limit_ms)
        self._enable_network = enable_network
        self._min_refs = min_refs

    @staticmethod
    def feature_names() -> Tuple[str, ...]:
        """Return ordered tuple of feature names matching vectorize() output."""
        return FEATURE_NAMES

    def vectorize(self, text: str) -> np.ndarray:
        """Extract 12-dimensional feature vector from text."""
        vec = np.zeros(REFERENCE_VECTOR_DIM, dtype=np.float64)
        stats = self.compute_stats(text)
        for name, idx in _VECTOR_SCHEMA:
            vec[idx] = stats.get(name, 0.0)
        return vec

    def compute_stats(self, text: str) -> Dict[str, Any]:
        """
        Full reference validation analysis.

        Returns dict with 12 numeric features plus metadata:
          - validation_results: list of ValidationResult dicts
          - inline_citations: list of inline citation strings found
          - references_extracted: count of refs found
          - network_enabled: whether API validation was performed
        """
        result: Dict[str, Any] = {name: 0.0 for name, _ in _VECTOR_SCHEMA}
        result["validation_results"] = []
        result["inline_citations"] = []
        result["references_extracted"] = 0
        result["network_enabled"] = self._enable_network

        if not text or len(text.split()) < 50:
            return result

        # ── Step 1: Extract references ────────────────────────────
        refs, inline_cites = self._extractor.extract(text)
        result["inline_citations"] = inline_cites
        result["references_extracted"] = len(refs)
        result["total_references"] = float(len(refs))

        if len(refs) < self._min_refs:
            return result

        # ── Step 2: Validate each reference ───────────────────────
        validations: List[ValidationResult] = []
        for ref in refs:
            if self._enable_network:
                vr = self._validate_reference(ref)
            else:
                vr = self._offline_validate(ref)
            validations.append(vr)

        # ── Step 3: Compute aggregate features ────────────────────
        total = len(validations)
        if total == 0:
            return result

        scores = [v.confidence_score for v in validations]
        verified = sum(1 for v in validations if v.status == "verified")
        fabricated = sum(1 for v in validations if v.status == "not_found")
        chimeric = sum(1 for v in validations if v.is_chimeric)
        partial = sum(1 for v in validations if v.status == "partial")
        ornamental = sum(1 for v in validations if v.is_ornamental)
        author_mismatches = sum(len(v.suspicious_authors) for v in validations)
        doi_fabs = sum(1 for v in validations if not v.doi_valid)
        date_anomalies = sum(1 for v in validations if v.date_anomaly)
        missing_inline = sum(1 for v in validations
                             if not v.reference.has_inline_citation and not v.reference.is_inline)

        result["verified_ratio"] = verified / total
        result["fabricated_ratio"] = fabricated / total
        result["chimeric_ratio"] = chimeric / total
        result["partial_match_ratio"] = partial / total
        result["ornamental_ratio"] = ornamental / total
        result["mean_confidence"] = float(np.mean(scores))
        result["min_confidence"] = float(min(scores))
        result["author_mismatch_count"] = float(author_mismatches)
        result["doi_fabrication_count"] = float(doi_fabs)
        result["date_anomaly_count"] = float(date_anomalies)
        result["missing_inline_ratio"] = missing_inline / total if total > 0 else 0.0

        # Serialize validation results for reporting
        result["validation_results"] = [
            self._vr_to_dict(v) for v in validations
        ]

        return result

    def _validate_reference(self, ref: ParsedReference) -> ValidationResult:
        """
        Validate a single reference using cascade API architecture.

        Step 1: If DOI present, look up directly
        Step 2: Search CrossRef with title/author query
        Step 3: If score < 70, fallback to S2 + OpenAlex
        Step 4: Cross-validate authors across sources
        """
        vr = ValidationResult(reference=ref)

        # ── Pre-checks ────────────────────────────────────────────
        # Date anomaly
        if ref.year is not None:
            if ref.year > _CURRENT_YEAR + 1:
                vr.date_anomaly = True
                vr.issues.append(f"Future publication year: {ref.year}")
            elif ref.year < 1900:
                vr.date_anomaly = True
                vr.issues.append(f"Implausible publication year: {ref.year}")

        # DOI validation
        if ref.doi:
            doi_result = self._api.lookup_doi(ref.doi)
            if doi_result:
                meta = _crossref_to_meta(doi_result)
                score, issues = _compute_confidence(ref, meta)
                vr.confidence_score = score
                vr.issues.extend(issues)
                vr.matched_title = meta["title"]
                vr.matched_authors = meta["authors"]
                vr.matched_doi = meta["doi"]
                vr.matched_year = meta["year"]
                vr.matched_journal = meta["journal"]
                vr.sources_checked.append("crossref-doi")

                if score >= _VERIFIED_THRESHOLD:
                    vr.status = "verified"
                    self._check_ornamental(ref, vr)
                    return vr
            else:
                vr.doi_valid = False
                vr.issues.append(f"DOI not found: {ref.doi}")

        # ── CrossRef search ───────────────────────────────────────
        query = self._build_query(ref)
        if not query:
            vr.status = "error"
            vr.issues.append("Insufficient data to build search query")
            return vr

        cr_results = self._api.search_crossref(query)
        vr.sources_checked.append("crossref")

        best_score = 0.0
        best_meta = None
        best_issues = []

        for item in cr_results:
            meta = _crossref_to_meta(item)
            score, issues = _compute_confidence(ref, meta)
            if score > best_score:
                best_score = score
                best_meta = meta
                best_issues = issues

        if best_meta and best_score >= _VERIFIED_THRESHOLD:
            vr.confidence_score = best_score
            vr.issues.extend(best_issues)
            vr.status = "verified"
            vr.matched_title = best_meta["title"]
            vr.matched_authors = best_meta["authors"]
            vr.matched_doi = best_meta.get("doi", "")
            vr.matched_year = best_meta.get("year")
            vr.matched_journal = best_meta.get("journal", "")
            self._check_ornamental(ref, vr)
            return vr

        # ── Fallback: Semantic Scholar + OpenAlex ─────────────────
        if best_score < _FALLBACK_THRESHOLD:
            # Semantic Scholar
            s2_results = self._api.search_semantic_scholar(query)
            vr.sources_checked.append("semantic_scholar")
            for item in s2_results:
                meta = _s2_to_meta(item)
                score, issues = _compute_confidence(ref, meta)
                if score > best_score:
                    best_score = score
                    best_meta = meta
                    best_issues = issues

            # OpenAlex
            oa_results = self._api.search_openalex(query)
            vr.sources_checked.append("openalex")
            for item in oa_results:
                meta = _openalex_to_meta(item)
                score, issues = _compute_confidence(ref, meta)
                if score > best_score:
                    best_score = score
                    best_meta = meta
                    best_issues = issues

        # ── Cross-validate authors (chimera detection) ────────────
        if best_meta and best_score >= _EXISTS_THRESHOLD:
            self._cross_validate_authors(ref, vr, best_meta, cr_results,
                                          s2_results if 'semantic_scholar' in vr.sources_checked else [],
                                          oa_results if 'openalex' in vr.sources_checked else [])

        # ── Final classification ──────────────────────────────────
        if best_meta:
            vr.confidence_score = best_score
            vr.issues.extend(best_issues)
            vr.matched_title = best_meta["title"]
            vr.matched_authors = best_meta["authors"]
            vr.matched_doi = best_meta.get("doi", "")
            vr.matched_year = best_meta.get("year")
            vr.matched_journal = best_meta.get("journal", "")

        if best_score >= _VERIFIED_THRESHOLD:
            vr.status = "verified"
        elif best_score >= _EXISTS_THRESHOLD:
            vr.status = "partial"
        else:
            vr.status = "not_found"

        # Check for chimeric pattern
        if best_meta:
            ref_title_n = _normalize_text(ref.title)
            cand_title_n = _normalize_text(best_meta.get("title", ""))
            title_sim = _levenshtein_ratio(ref_title_n, cand_title_n)
            ref_authors = set(ref.authors)
            cand_authors = set(best_meta.get("authors", []))
            if ref_authors and cand_authors:
                author_sim = len(ref_authors & cand_authors) / max(len(ref_authors), 1)
                if title_sim > _TITLE_MATCH_HIGH and author_sim < _AUTHOR_MISMATCH_THR:
                    vr.is_chimeric = True
                    vr.status = "chimeric"
                    vr.issues.append(
                        "CHIMERIC: Title matches an existing work but authors "
                        "do not correspond — elements from multiple papers may "
                        "have been combined."
                    )

        self._check_ornamental(ref, vr)
        return vr

    def _offline_validate(self, ref: ParsedReference) -> ValidationResult:
        """Offline validation — heuristic checks without API access."""
        vr = ValidationResult(reference=ref)
        vr.sources_checked.append("offline-heuristic")

        # Date anomaly
        if ref.year is not None:
            if ref.year > _CURRENT_YEAR + 1:
                vr.date_anomaly = True
                vr.issues.append(f"Future publication year: {ref.year}")
                vr.confidence_score = 20
            elif ref.year < 1900:
                vr.date_anomaly = True
                vr.issues.append(f"Implausible year: {ref.year}")
                vr.confidence_score = 20
            else:
                vr.confidence_score = 50  # can't verify without network

        # DOI format check
        if ref.doi:
            if not re.match(r'^10\.\d{4,}/', ref.doi):
                vr.doi_valid = False
                vr.issues.append(f"Invalid DOI format: {ref.doi}")
                vr.confidence_score = max(vr.confidence_score - 20, 0)

        vr.status = "not_checked"
        self._check_ornamental(ref, vr)
        return vr

    @staticmethod
    def _check_ornamental(ref: ParsedReference, vr: ValidationResult) -> None:
        """Check if reference is ornamental (in bib but not cited in text)."""
        if not ref.has_inline_citation and not ref.is_inline:
            vr.is_ornamental = True
            vr.issues.append(
                "ORNAMENTAL: Reference appears in bibliography but is not "
                "cited in the text body — a pattern characteristic of "
                "AI-generated reference lists."
            )

    def _cross_validate_authors(self, ref: ParsedReference,
                                 vr: ValidationResult,
                                 best_meta: dict,
                                 cr_items: list,
                                 s2_items: list,
                                 oa_items: list) -> None:
        """
        Cross-validate authors across multiple databases.
        Authors confirmed in 2+ sources → confirmed.
        Authors in query but not in any source → suspicious.
        """
        all_authors_from_apis: set = set()

        # Collect all authors from all API results
        for item in cr_items:
            meta = _crossref_to_meta(item)
            all_authors_from_apis.update(meta["authors"])
        for item in s2_items:
            meta = _s2_to_meta(item)
            all_authors_from_apis.update(meta["authors"])
        for item in oa_items:
            meta = _openalex_to_meta(item)
            all_authors_from_apis.update(meta["authors"])

        # Cross-validate
        for author in ref.authors:
            if author in all_authors_from_apis:
                vr.confirmed_authors.append(author)
            else:
                vr.suspicious_authors.append(author)

        # Apply bonus/penalty
        if vr.confirmed_authors:
            vr.confidence_score = min(100, vr.confidence_score + _BONUS_CROSS_VALIDATED)
        if vr.suspicious_authors:
            for sa in vr.suspicious_authors:
                vr.issues.append(f"Author '{sa}' not confirmed in any database")

    @staticmethod
    def _build_query(ref: ParsedReference) -> str:
        """Build a search query from reference components."""
        parts = []
        if ref.title:
            parts.append(ref.title[:150])
        if ref.authors:
            parts.append(" ".join(ref.authors[:3]))
        if ref.year:
            parts.append(str(ref.year))
        return " ".join(parts)

    @staticmethod
    def _vr_to_dict(vr: ValidationResult) -> dict:
        """Serialize ValidationResult to dict for JSON export."""
        return {
            "raw_text": vr.reference.raw_text[:200],
            "title": vr.reference.title[:150],
            "authors": vr.reference.authors,
            "year": vr.reference.year,
            "doi": vr.reference.doi,
            "confidence_score": round(vr.confidence_score, 1),
            "status": vr.status,
            "matched_title": vr.matched_title[:150] if vr.matched_title else "",
            "matched_doi": vr.matched_doi,
            "suspicious_authors": vr.suspicious_authors,
            "confirmed_authors": vr.confirmed_authors,
            "issues": vr.issues[:5],
            "is_chimeric": vr.is_chimeric,
            "is_ornamental": vr.is_ornamental,
            "doi_valid": vr.doi_valid,
            "date_anomaly": vr.date_anomaly,
            "sources_checked": vr.sources_checked,
        }


# ============================================================================
# REFERENCE RISK CLASSIFIER (for ForensicReportGenerator)
# ============================================================================

class ReferenceRiskClassifier:
    """
    Severity-profile classifier for reference validation results.

    DISPLAY/REPORTING LAYER ONLY — produces per-feature severity levels
    and human-readable explanations. The ai_score is derived from
    majority-vote of severity levels (Late Fusion compliant).
    """

    _HIGH   = 0.65
    _MEDIUM = 0.35

    _EXPLANATIONS = {
        "fabricated_ratio": {
            "display": "Fabricated Citation Ratio",
            "high": "{v:.0%} of references could not be found in any academic database (CrossRef, Semantic Scholar, OpenAlex). These are likely AI-hallucinated citations.",
            "medium": "{v:.0%} of references could not be verified. Some may be real but not indexed, or may be fabricated.",
            "low": "All or nearly all references ({v:.0%} unverifiable) were found in academic databases.",
        },
        "chimeric_ratio": {
            "display": "Chimeric Citation Ratio",
            "high": "{v:.0%} of references appear chimeric — the title matches a real paper but the authors don't correspond. This is a hallmark of AI citation fabrication.",
            "medium": "{v:.0%} of references show partial author-title mismatches.",
            "low": "No chimeric references detected ({v:.0%}).",
        },
        "ornamental_ratio": {
            "display": "Ornamental Reference Ratio",
            "high": "{v:.0%} of bibliography entries are never cited in the text body. AI often generates 'decorative' reference lists.",
            "medium": "Some references ({v:.0%}) lack corresponding inline citations.",
            "low": "References are well-integrated with inline citations ({v:.0%} ornamental).",
        },
        "mean_confidence": {
            "display": "Mean Verification Confidence",
            "high": "Very low mean confidence ({v:.0f}/100) across all references — most could not be reliably matched to real publications.",
            "medium": "Moderate confidence ({v:.0f}/100) — some references verified, others uncertain.",
            "low": "High confidence ({v:.0f}/100) — references match well against academic databases.",
        },
    }

    def classify(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Produce severity profile + display-only ai_score."""
        severity = self._severity_profile(stats)
        score = self._score_from_severity(severity, stats)
        level = self._level(score)

        return {
            "ai_score": score,
            "risk_level": level,
            "severity_profile": severity,
            "interpretation": self._interpretation(score, stats),
            "feature_details": self._feature_details(stats),
            "total_references": int(stats.get("total_references", 0)),
            "validation_results": stats.get("validation_results", []),
        }

    def _severity_profile(self, stats: Dict[str, Any]) -> Dict[str, str]:
        profile = {}
        for feat in self._EXPLANATIONS:
            val = stats.get(feat, 0.0)
            profile[feat] = self._feat_level(feat, val)
        return profile

    def _score_from_severity(self, severity: Dict[str, str],
                             stats: Dict[str, Any]) -> float:
        refs = stats.get("total_references", 0)
        if refs < 1:
            return 0.0

        votes = []
        for feat, level in severity.items():
            if level == "high":
                votes.append(1.0)
            elif level == "medium":
                votes.append(0.5)
            else:
                votes.append(0.0)

        if not votes:
            return 0.0

        return round(min(1.0, max(0.0, float(np.mean(votes)))), 4)

    def _level(self, score: float) -> str:
        if score <= 0.0:
            return "NO REFERENCES FOUND"
        if score >= self._HIGH:
            return "HIGH — Fabricated Citations Detected"
        if score >= self._MEDIUM:
            return "MEDIUM — Some Citations Unverifiable"
        return "LOW — Citations Appear Legitimate"

    def _interpretation(self, score: float, stats: Dict[str, Any]) -> str:
        total = int(stats.get("total_references", 0))
        fab = stats.get("fabricated_ratio", 0.0)
        chim = stats.get("chimeric_ratio", 0.0)
        orn = stats.get("ornamental_ratio", 0.0)
        mean_conf = stats.get("mean_confidence", 0.0)

        if total == 0:
            return "No references were found in the text for validation."

        if score >= self._HIGH:
            parts = [f"CRITICAL: Of {total} references analysed, {fab:.0%} could not "
                     f"be found in academic databases (mean confidence: {mean_conf:.0f}/100)."]
            if chim > 0:
                parts.append(f"{chim:.0%} appear chimeric (mixed elements from different papers).")
            if orn > 0:
                parts.append(f"{orn:.0%} are ornamental (in bibliography but never cited in text).")
            parts.append("Fabricated citations are strong evidence of AI-generated content.")
            return " ".join(parts)

        if score >= self._MEDIUM:
            return (f"Of {total} references, some could not be fully verified "
                    f"(mean confidence: {mean_conf:.0f}/100). This may indicate "
                    f"AI-assisted writing or citation errors. Manual verification recommended.")

        return (f"All {total} references verified with high confidence "
                f"(mean: {mean_conf:.0f}/100). Citations appear legitimate and "
                f"properly attributed.")

    def _feature_details(self, stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        details = {}
        for feat, info in self._EXPLANATIONS.items():
            val = stats.get(feat, 0.0)
            level = self._feat_level(feat, val)
            template = info.get(level, "")
            details[feat] = {
                "display_name": info["display"],
                "value": round(val, 4) if isinstance(val, float) else val,
                "level": level,
                "explanation": template.format(v=val) if "{v" in template else template,
            }
        return details

    @staticmethod
    def _feat_level(feat: str, val: float) -> str:
        if feat == "mean_confidence":
            # Inverse: lower confidence = higher risk
            if val < 40: return "high"
            if val < 70: return "medium"
            return "low"

        # Direct: higher ratio = higher risk
        if feat in ("fabricated_ratio", "chimeric_ratio"):
            if val > 0.3: return "high"
            if val > 0.1: return "medium"
            return "low"

        if feat == "ornamental_ratio":
            if val > 0.5: return "high"
            if val > 0.2: return "medium"
            return "low"

        return "low"
