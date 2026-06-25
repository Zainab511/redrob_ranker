#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker
================================================
Single command: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Architecture:
  Stage 0: Plausibility / honeypot filter  (hard flags, rule-based)
  Stage 1: BM25 retrieval                  (lexical, cuts 100k → ~2k)
  Stage 2: Compound JD-grounded scoring    (multi-component weighted score)
  Stage 3: Reasoning generation            (template-grounded, no hallucination)

Designed to run in <5 min, 16GB RAM, CPU-only, no network.
"""

import argparse
import csv
import gc
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("ERROR: rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 22)

# Technology release year floor — for impossible-skill-duration detection
TECH_RELEASE_YEAR = {
    "Pinecone": 2021, "Weaviate": 2019, "Qdrant": 2021, "Milvus": 2019,
    "LoRA": 2021, "PEFT": 2022, "Fine-tuning LLMs": 2018,
    "Hugging Face Transformers": 2018, "BentoML": 2019, "LangChain": 2022,
    "Prompt Engineering": 2020, "Sentence Transformers": 2019,
    "Kubeflow": 2018, "Haystack": 2020, "MLflow": 2018,
}

# Pure IT-services companies — disqualifier per JD (entire career, no product-co history)
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini",
    "tech mahindra", "hcl", "hcltech", "mindtree", "ltimindtree",
    "l&t infotech", "mphasis", "persistent systems", "birlasoft", "genpact",
}

# Fictional filler companies — neutral (not product, not services)
FICTIONAL_COMPANIES = {
    "wayne enterprises", "initech", "pied piper", "globex inc", "globex",
    "acme corp", "acme", "dunder mifflin", "hooli", "stark industries",
}

# AI/ML titles that clearly align with the JD
AI_ALIGNED_TITLES = {
    "ml engineer": 1.0,
    "machine learning engineer": 1.0,
    "senior machine learning engineer": 1.0,
    "staff machine learning engineer": 1.0,
    "senior ai engineer": 1.0,
    "lead ai engineer": 1.0,
    "ai engineer": 0.95,
    "ai specialist": 0.9,
    "applied ml engineer": 1.0,
    "senior applied scientist": 1.0,
    "search engineer": 1.0,
    "recommendation systems engineer": 1.0,
    "nlp engineer": 0.95,
    "senior nlp engineer": 1.0,
    "data scientist": 0.8,
    "senior data scientist": 0.85,
    "ai research engineer": 0.85,  # research flag but still positive
    "computer vision engineer": 0.5,  # JD says CV-only without NLP/IR is weak fit
    "junior ml engineer": 0.55,  # likely too junior
    "senior software engineer (ml)": 0.9,
}

# Skills the JD explicitly wants (hard requirements vs nice-to-have)
REQUIRED_SKILLS = {
    "embeddings", "vector search", "information retrieval",
    "sentence transformers", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "bm25", "semantic search",
    "hugging face transformers", "python",
}

PREFERRED_SKILLS = {
    "nlp", "machine learning", "deep learning", "recommendation systems",
    "fine-tuning llms", "lora", "peft", "mlops", "mlflow", "kubeflow",
    "learning to rank", "xgboost", "lightgbm", "langchain", "haystack",
    "a/b testing", "evaluation framework", "ranking", "retrieval",
    "reinforcement learning from human feedback", "rlhf",
    "prompt engineering", "bentoml", "fastapi", "spark", "airflow",
}

# Disqualifying signals from the JD text
PURE_RESEARCH_RE = re.compile(
    r"(academic (lab|research|position)|phd (student|candidate|researcher)|"
    r"research (only|position|scientist at (deepmind|openai|google brain|fair|msr))|"
    r"no production (deployment|experience))",
    re.IGNORECASE,
)

LANGCHAIN_TOURIST_RE = re.compile(
    r"(langchain|llamaindex).{0,60}(my (first|only)|side project|tutorial|beginner|started (using|learning))",
    re.IGNORECASE,
)

ASPIRATIONAL_RE = re.compile(
    r"(interested in transitioning|want to (do more|transition)|"
    r"self.directed (ml|ai) project|kaggle competition|"
    r"learning modern ml|building competence on the ml|"
    r"wouldn.t call myself|comfortable with the modeling work but|"
    r"completed a couple of self.directed)",
    re.IGNORECASE,
)

NO_RECENT_CODE_RE = re.compile(
    r"(moved into (architecture|tech lead|engineering manager)|"
    r"no longer (write|coding|hands.on)|"
    r"primarily (architect|strategic|managerial))",
    re.IGNORECASE,
)

TITLE_CHASER_SIGNALS = re.compile(
    r"(senior|staff|principal|lead|director).{0,30}(promotion|title|leveling)",
    re.IGNORECASE,
)

# Ownership language in career history — indicates actually BUILT systems
OWNERSHIP_RE = re.compile(
    r"(built|build|developed|created|launched|implemented|designed and (built|deployed)|"
    r"architected|owned|led (the )?development|shipped)"
    r".{0,80}"
    r"(ranking|recommendation|retrieval|search (relevance|engine|quality)|"
    r"matching (algorithm|system)|embedding.based|vector (search|index|database))",
    re.IGNORECASE,
)

# BM25 query — what we're searching for in candidate text
JD_QUERY_TERMS = [
    "ranking", "retrieval", "recommendation", "embeddings", "vector", "search",
    "nlp", "machine learning", "python", "evaluation", "ndcg", "a/b test",
    "production", "shipped", "built", "deployed", "hybrid search", "bm25",
    "fine-tuning", "sentence-transformers", "faiss", "pinecone", "weaviate",
    "opensearch", "elasticsearch", "reranking", "semantic", "llm",
]

# Preferred locations per JD
PREFERRED_LOCATIONS = {"pune", "noida", "delhi", "gurugram", "gurgaon", "new delhi"}
ACCEPTABLE_LOCATIONS = {"hyderabad", "mumbai", "bangalore", "bengaluru", "delhi ncr"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def days_since(d: date) -> int:
    if d is None:
        return 9999
    return max(0, (TODAY - d).days)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def is_services(company: str) -> bool:
    return company.strip().lower() in SERVICES_COMPANIES


def is_fictional(company: str) -> bool:
    return company.strip().lower() in FICTIONAL_COMPANIES


def load_candidates(path: str):
    """Stream candidates from .jsonl or .jsonl.gz"""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return candidates


def candidate_text(c: dict) -> str:
    """Flatten all text fields into a single string for BM25 indexing."""
    p = c.get("profile", {}) or {}
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for ch in (c.get("career_history") or []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    for s in (c.get("skills") or []):
        parts.append(s.get("name", ""))
    for cert in (c.get("certifications") or []):
        parts.append(cert.get("name", ""))
    return " ".join(p for p in parts if p).lower()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0: PLAUSIBILITY / HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def honeypot_penalty(c: dict) -> float:
    """
    Returns a penalty multiplier: 1.0 = clean, 0.0 = certain honeypot.
    We use a graduated scale so marginal candidates aren't hard-excluded
    (which would risk over-flagging legitimate noisy profiles).
    Only very strong, multi-check violations get pushed near 0.
    """
    flags = 0
    profile = c.get("profile", {}) or {}
    career = c.get("career_history") or []
    skills = c.get("skills") or []
    yoe = safe_float(profile.get("years_of_experience"))

    # Check 1: skill duration exceeds technology's existence (hard evidence)
    for s in skills:
        sname = s.get("name", "")
        dur = safe_float(s.get("duration_months"))
        if sname in TECH_RELEASE_YEAR:
            max_plausible = (TODAY.year - TECH_RELEASE_YEAR[sname]) * 12 + 6
            if dur > max_plausible:
                flags += 2  # strong signal — worth double

    # Check 2: stacked "expert" proficiency with near-zero duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and safe_float(s.get("duration_months")) <= 3
    )
    if expert_zero >= 2:
        flags += 1

    # Check 3: years_of_experience vs career_history sum — >4yr mismatch
    career_sum_years = sum(safe_float(ch.get("duration_months")) for ch in career) / 12
    if yoe > 0 and abs(yoe - career_sum_years) > 4:
        flags += 1

    # Check 4: current role tenure exceeds total stated experience
    for ch in career:
        if ch.get("is_current") and yoe > 0:
            if safe_float(ch.get("duration_months")) > yoe * 12 + 3:
                flags += 1

    # Graduated penalty
    if flags == 0:
        return 1.0
    elif flags == 1:
        return 0.85   # mild suspicion — don't hard-punish noise
    elif flags == 2:
        return 0.5    # moderate — push down but don't exclude (could be noise)
    else:
        return 0.1    # 3+ flags — almost certainly a honeypot, near-exclude


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A: TITLE & CAREER TRAJECTORY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def title_score(c: dict) -> tuple[float, str]:
    """
    Returns (score 0-1, signal_label).
    Scores based on current title alignment with JD.
    Includes title-chaser detection (frequent short-tenure hops with escalating titles).
    """
    profile = c.get("profile", {}) or {}
    career = c.get("career_history") or []
    title = (profile.get("current_title") or "").strip().lower()

    base = AI_ALIGNED_TITLES.get(title, 0.0)
    label = f"title={title}"

    # Title-chaser detection: >= 3 jobs each < 18 months, with different titles
    short_tenures = [
        ch for ch in career
        if safe_float(ch.get("duration_months")) < 18
    ]
    titles_in_short = set(ch.get("title", "").strip().lower() for ch in short_tenures)
    if len(short_tenures) >= 3 and len(titles_in_short) >= 3:
        base *= 0.7
        label += ",title_chaser"

    return clamp(base), label


def company_type_score(c: dict) -> tuple[float, str]:
    """
    Per JD: services-only career = disqualifier. Product-co experience = good.
    Fictional/neutral companies are treated as product-co (benefit of the doubt).
    """
    profile = c.get("profile", {}) or {}
    career = c.get("career_history") or []

    current_co = (profile.get("current_company") or "").strip()
    all_companies = [current_co] + [ch.get("company", "") for ch in career]

    has_product_co = False
    all_services = True
    for co in all_companies:
        co_lower = co.strip().lower()
        if co_lower in SERVICES_COMPANIES:
            continue
        else:
            all_services = False
            if co_lower not in FICTIONAL_COMPANIES:
                has_product_co = True

    if all_services:
        return 0.2, "all_services_career"
    elif has_product_co:
        return 1.0, "product_co_experience"
    else:
        return 0.6, "neutral_companies"


def career_substance_score(c: dict) -> tuple[float, str]:
    """
    Scores based on what careers actually show — ownership vs. support vs. aspirational.
    The key differentiator vs naive keyword matching.
    """
    profile = c.get("profile", {}) or {}
    career = c.get("career_history") or []
    full_text = " ".join([
        profile.get("summary", ""),
        " ".join(ch.get("description", "") for ch in career)
    ])

    score = 0.5  # baseline — neutral
    label_parts = []

    # Strong positive: explicit ownership/shipping of ranking/retrieval/recommendation
    ownership_matches = OWNERSHIP_RE.findall(full_text)
    if ownership_matches:
        score += 0.35
        label_parts.append("built_ranking_system")

    # Negative: pure research language
    if PURE_RESEARCH_RE.search(full_text):
        score -= 0.4
        label_parts.append("pure_research")

    # Negative: aspirational/transitioning decoy persona
    if ASPIRATIONAL_RE.search(full_text):
        score -= 0.45
        label_parts.append("aspirational_only")

    # Negative: LangChain tourist (sub-12mo LLM-only without pre-LLM experience)
    if LANGCHAIN_TOURIST_RE.search(full_text):
        score -= 0.2
        label_parts.append("langchain_tourist")

    # Negative: no recent production code (moved to arch/tech-lead only)
    if NO_RECENT_CODE_RE.search(full_text):
        score -= 0.2
        label_parts.append("no_recent_code")

    # Mild positive: any mention of evaluation frameworks (NDCG, MRR, A/B)
    if re.search(r"\b(ndcg|mrr|map@|a/?b test|offline.online|recall@)\b", full_text, re.IGNORECASE):
        score += 0.1
        label_parts.append("eval_framework")

    label = ",".join(label_parts) if label_parts else "neutral"
    return clamp(score), label


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B: EXPERIENCE YEARS SCORE
# ─────────────────────────────────────────────────────────────────────────────

def experience_score(c: dict) -> tuple[float, str]:
    """
    JD says 5-9 years, but explicitly says "we'll consider outside the band."
    Peak score at 6-8 years, graceful degradation outside.
    Under 3 years = near-zero. Over 12 = slight penalty (risk of over-qualified/arch-only).
    """
    profile = c.get("profile", {}) or {}
    yoe = safe_float(profile.get("years_of_experience"))

    if yoe < 3:
        s = 0.1
    elif yoe < 5:
        s = 0.4 + 0.25 * ((yoe - 3) / 2)
    elif yoe <= 9:
        s = 0.85 + 0.15 * (1 - abs(yoe - 7) / 4)  # peak at 7
    elif yoe <= 12:
        s = 0.75 - 0.1 * ((yoe - 9) / 3)
    else:
        s = 0.55  # very senior — possible but risk of arch-only

    return clamp(s), f"yoe={yoe:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2C: SKILLS SCORE
# ─────────────────────────────────────────────────────────────────────────────

def skills_score(c: dict) -> tuple[float, str]:
    """
    Skills are the WEAKEST signal per our analysis — they're trivially gamed.
    We score on: required skill coverage, proficiency quality, duration authenticity.
    We deliberately down-weight raw skill count (keyword stuffer mitigation).
    """
    skills = c.get("skills") or []
    if not skills:
        return 0.0, "no_skills"

    skill_lookup = {}
    for s in skills:
        name = (s.get("name") or "").lower().strip()
        skill_lookup[name] = s

    # Required skill coverage (0-1)
    required_hits = sum(1 for sk in REQUIRED_SKILLS if sk in skill_lookup)
    required_coverage = required_hits / len(REQUIRED_SKILLS)

    # Preferred skill coverage (0-1)
    preferred_hits = sum(1 for sk in PREFERRED_SKILLS if sk in skill_lookup)
    preferred_coverage = preferred_hits / len(PREFERRED_SKILLS)

    # Quality multiplier: reward expert proficiency, penalize beginner-only
    proficiency_map = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    ai_relevant_skills = [s for s in skills if (s.get("name") or "").lower() in REQUIRED_SKILLS | PREFERRED_SKILLS]

    if ai_relevant_skills:
        avg_proficiency = sum(
            proficiency_map.get(s.get("proficiency", "intermediate"), 0.6)
            for s in ai_relevant_skills
        ) / len(ai_relevant_skills)
    else:
        avg_proficiency = 0.3

    # Duration authenticity: endorsements + reasonable duration
    endorsed_skills = sum(1 for s in ai_relevant_skills if safe_float(s.get("endorsements")) >= 2)
    endorse_ratio = min(1.0, endorsed_skills / max(1, len(ai_relevant_skills)))

    # Keyword stuffer penalty: if many AI skills but title is clearly non-AI
    profile = c.get("profile", {}) or {}
    title = (profile.get("current_title") or "").strip().lower()
    ai_skill_count = sum(1 for s in skills if (s.get("name") or "").lower() in REQUIRED_SKILLS | PREFERRED_SKILLS)
    is_non_ai_title = title not in AI_ALIGNED_TITLES
    stuffer_penalty = 0.4 if (is_non_ai_title and ai_skill_count >= 6) else 1.0

    score = (
        0.45 * required_coverage +
        0.25 * preferred_coverage +
        0.20 * avg_proficiency +
        0.10 * endorse_ratio
    ) * stuffer_penalty

    label = f"req_cov={required_hits}/{len(REQUIRED_SKILLS)},pref_hits={preferred_hits}"
    return clamp(score), label


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2D: BEHAVIORAL / REACHABILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_score(c: dict) -> tuple[float, str]:
    """
    JD explicitly instructs: "perfect-on-paper candidate who hasn't logged in
    for 6 months and has a 5% recruiter response rate... down-weight them."
    This is a MULTIPLIER on the base score, not an additive component.
    Range 0.2 (unreachable) to 1.1 (very active, explicitly job-seeking).
    """
    sig = c.get("redrob_signals", {}) or {}
    label_parts = []

    # 1. Recency of activity (0-1)
    la = parse_date(sig.get("last_active_date"))
    inactive_days = days_since(la)
    if inactive_days <= 30:
        recency = 1.0
    elif inactive_days <= 90:
        recency = 0.85
    elif inactive_days <= 180:
        recency = 0.65
    elif inactive_days <= 365:
        recency = 0.4
    else:
        recency = 0.2
    label_parts.append(f"inactive={inactive_days}d")

    # 2. Open to work flag — explicit signal
    open_flag = 1.15 if sig.get("open_to_work_flag") else 0.9
    if sig.get("open_to_work_flag"):
        label_parts.append("open_to_work")

    # 3. Recruiter response rate
    rrr = safe_float(sig.get("recruiter_response_rate"), 0.5)
    response_score = clamp(rrr)

    # 4. Interview completion rate
    icr = safe_float(sig.get("interview_completion_rate"), 0.5)
    interview_score = clamp(icr)

    # 5. Profile completeness
    pc = safe_float(sig.get("profile_completeness_score"), 50) / 100.0
    completeness = clamp(pc)

    # 6. Notice period (JD prefers <30 days, can buy out 30 days)
    notice = safe_float(sig.get("notice_period_days"), 60)
    if notice <= 30:
        notice_score = 1.0
        label_parts.append("notice_ok")
    elif notice <= 60:
        notice_score = 0.8
    elif notice <= 90:
        notice_score = 0.65
    else:
        notice_score = 0.45
        label_parts.append("long_notice")

    # 7. GitHub activity (only for non-sentinel values)
    gha = safe_float(sig.get("github_activity_score"), -1)
    if gha == -1:
        github_score = 0.5  # no signal — neutral
    else:
        github_score = clamp(gha / 100.0)
        if gha >= 60:
            label_parts.append("active_github")

    # Composite behavioral score
    behavioral = (
        0.30 * recency +
        0.20 * response_score +
        0.15 * interview_score +
        0.15 * completeness +
        0.10 * notice_score +
        0.10 * github_score
    ) * open_flag

    return clamp(behavioral, 0.2, 1.1), ",".join(label_parts)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2E: LOCATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def location_score(c: dict) -> tuple[float, str]:
    """
    Pune/Noida preferred. Hyderabad/Mumbai/Delhi NCR acceptable.
    International: possible but no visa sponsorship.
    """
    profile = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}
    country = (profile.get("country") or "").strip().lower()
    location = (profile.get("location") or "").strip().lower()
    willing_to_relocate = sig.get("willing_to_relocate", False)

    if country == "india":
        # Check if in preferred/acceptable city
        for city in PREFERRED_LOCATIONS:
            if city in location:
                return 1.0, f"preferred_loc={location}"
        for city in ACCEPTABLE_LOCATIONS:
            if city in location:
                return 0.85, f"acceptable_loc={location}"
        # Elsewhere in India — can relocate?
        if willing_to_relocate:
            return 0.75, f"india_relocate={location}"
        return 0.55, f"india_other={location}"
    else:
        # International — possible but penalty (no visa sponsorship)
        if willing_to_relocate:
            return 0.5, f"intl_relocate={country}"
        return 0.3, f"intl_no_relocate={country}"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2F: EDUCATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def education_score(c: dict) -> tuple[float, str]:
    """
    Light signal — JD doesn't emphasize education, so this is low-weight.
    Rewards tier_1/tier_2 institutions slightly; not a strong differentiator.
    """
    edu = c.get("education") or []
    if not edu:
        return 0.4, "no_edu"

    tier_scores = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}
    best_tier = max(tier_scores.get(e.get("tier", "unknown"), 0.5) for e in edu)
    return clamp(best_tier), f"edu_tier={max((e.get('tier','?') for e in edu), key=lambda t: tier_scores.get(t,0.5))}"


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SCORE
# ─────────────────────────────────────────────────────────────────────────────

# Component weights — must sum to 1.0 (behavioral is applied as a multiplier separately)
WEIGHTS = {
    "career_substance": 0.35,   # most important — what they actually did
    "title":           0.20,   # what role they're in now
    "skills":          0.15,   # what they claim (weakest — easily gamed)
    "experience":      0.15,   # years in band
    "company_type":    0.10,   # product vs services
    "location":        0.03,   # logistics
    "education":       0.02,   # light signal
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


def score_candidate(c: dict, bm25_score: float = 0.0) -> tuple[float, dict]:
    """
    Full compound score for a single candidate.
    Returns (final_score 0-1, component_breakdown dict).
    """
    hp = honeypot_penalty(c)

    t_score, t_label = title_score(c)
    co_score, co_label = company_type_score(c)
    cs_score, cs_label = career_substance_score(c)
    exp_score, exp_label = experience_score(c)
    sk_score, sk_label = skills_score(c)
    beh_score, beh_label = behavioral_score(c)
    loc_score, loc_label = location_score(c)
    edu_score, edu_label = education_score(c)

    # Weighted base score
    base = (
        WEIGHTS["career_substance"] * cs_score +
        WEIGHTS["title"] * t_score +
        WEIGHTS["skills"] * sk_score +
        WEIGHTS["experience"] * exp_score +
        WEIGHTS["company_type"] * co_score +
        WEIGHTS["location"] * loc_score +
        WEIGHTS["education"] * edu_score
    )

    # BM25 adds a small lexical-match bonus (max 5% uplift)
    bm25_bonus = 0.05 * clamp(bm25_score)

    # Behavioral is a multiplier on the full score (per JD instruction)
    # BUT: cap behavioral uplift based on experience score to prevent
    # very active but too-junior candidates from ranking above experienced ones.
    # If experience_score < 0.6 (roughly < 4 years), cap behavioral at 0.95 (no uplift).
    if exp_score < 0.6:
        beh_score_effective = min(beh_score, 0.95)
    else:
        beh_score_effective = beh_score

    final = (base + bm25_bonus) * beh_score_effective * hp

    breakdown = {
        "honeypot_penalty": round(hp, 3),
        "career_substance": (round(cs_score, 3), cs_label),
        "title": (round(t_score, 3), t_label),
        "skills": (round(sk_score, 3), sk_label),
        "experience": (round(exp_score, 3), exp_label),
        "company_type": (round(co_score, 3), co_label),
        "behavioral": (round(beh_score_effective, 3), beh_label),
        "location": (round(loc_score, 3), loc_label),
        "education": (round(edu_score, 3), edu_label),
        "bm25_bonus": round(bm25_bonus, 4),
        "final_score": round(final, 6),
    }
    return final, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: REASONING GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_ownership_snippet(c: dict) -> str:
    """Pull the actual matched ownership sentence from career history."""
    career = c.get("career_history") or []
    for ch in career:
        desc = ch.get("description", "")
        m = OWNERSHIP_RE.search(desc)
        if m:
            # Get a clean snippet: expand to sentence boundary, max 100 chars
            snippet = m.group(0)[:100].rstrip()
            return f'"{snippet}..."' if len(m.group(0)) > 100 else f'"{snippet}"'
    return ""


def generate_reasoning(c: dict, breakdown: dict) -> str:
    """
    Produces a specific, grounded 1-2 sentence reasoning string per candidate.
    Pulls from real fields — no hallucination risk. Varied by dominant driver.
    """
    profile = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}
    career = c.get("career_history") or []
    skills = c.get("skills") or []

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = safe_float(profile.get("years_of_experience"))
    country = (profile.get("country") or "").strip()

    cs_score, cs_label = breakdown["career_substance"]
    t_score, t_label = breakdown["title"]
    beh_score, beh_label = breakdown["behavioral"]
    hp = breakdown["honeypot_penalty"]
    sk_score, sk_label = breakdown["skills"]
    exp_score, exp_label = breakdown["experience"]
    co_score, co_label = breakdown["company_type"]

    rrr = safe_float(sig.get("recruiter_response_rate"), 0.5)
    notice = safe_float(sig.get("notice_period_days"), 60)
    la = parse_date(sig.get("last_active_date"))
    inactive_days = days_since(la)
    open_flag = sig.get("open_to_work_flag", False)
    gha = safe_float(sig.get("github_activity_score"), -1)

    # Get top AI skills this candidate has listed
    ai_skills_present = [
        s.get("name") for s in skills
        if (s.get("name") or "").lower() in REQUIRED_SKILLS and
        s.get("proficiency") in ("advanced", "expert")
    ][:3]

    ownership_snippet = extract_ownership_snippet(c)

    parts = []

    # ── Sentence 1: strongest positive signal ────────────────────────────────
    if "built_ranking_system" in cs_label and ownership_snippet:
        tech_context = f" with skills in {', '.join(ai_skills_present)}" if ai_skills_present else ""
        parts.append(
            f"{yoe:.0f}-year {title} at {company}{tech_context}; "
            f"career history shows direct production ownership: {ownership_snippet}."
        )
    elif "built_ranking_system" in cs_label:
        parts.append(
            f"{title} at {company} ({yoe:.0f} yrs) with confirmed hands-on "
            f"ownership of ranking/retrieval/recommendation systems in career history."
        )
    elif t_score >= 0.9 and "eval_framework" in cs_label:
        req_hits = sk_label.split(",")[0].replace("req_cov=", "")
        parts.append(
            f"{yoe:.0f}-year {title} at {company} with evaluation-framework experience "
            f"(NDCG/A-B testing) and {req_hits} of {len(REQUIRED_SKILLS)} required skills covered."
        )
    elif t_score >= 0.8:
        skill_str = f"; strong in {', '.join(ai_skills_present)}" if ai_skills_present else ""
        parts.append(
            f"{title} at {company} ({yoe:.0f} yrs){skill_str}; "
            f"title and experience band align well with the Senior AI Engineer role."
        )
    else:
        parts.append(
            f"{title} at {company} ({yoe:.0f} yrs); partial alignment — "
            f"scores on title fit and career substance are moderate."
        )

    # ── Sentence 2: availability OR concerns ─────────────────────────────────
    concerns = []
    positives = []

    if hp < 0.8:
        concerns.append("profile contains implausible experience claims (honeypot flag)")
    if "aspirational_only" in cs_label:
        concerns.append("summary reads as ML-transitioning rather than established practitioner")
    if "all_services_career" in co_label:
        concerns.append("entire career in IT-services firms (JD explicitly flags this as weaker fit)")
    if notice > 60:
        concerns.append(f"notice period {notice:.0f}d (JD prefers ≤30d)")
    if inactive_days > 180:
        concerns.append(f"last active {inactive_days}d ago — reachability uncertain")
    if rrr < 0.3:
        concerns.append(f"low recruiter response rate ({rrr:.0%})")
    if country not in ("India", "india", "") and not sig.get("willing_to_relocate"):
        concerns.append(f"based in {country}, no visa sponsorship available, not flagged willing to relocate")

    if not concerns:
        avail_parts = []
        if open_flag:
            avail_parts.append("actively open to work")
        avail_parts.append(f"responds to {rrr:.0%} of recruiter messages")
        if inactive_days <= 90:
            avail_parts.append(f"active {inactive_days}d ago")
        if notice <= 30:
            avail_parts.append(f"≤30d notice")
        if gha >= 60:
            avail_parts.append(f"GitHub activity score {gha:.0f}/100")
        parts.append("Availability: " + ", ".join(avail_parts) + ".")
    else:
        parts.append("Note: " + "; ".join(concerns) + ".")

    reasoning = " ".join(parts)
    if len(reasoning) > 450:
        reasoning = reasoning[:447] + "..."
    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl", help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to return")
    parser.add_argument("--bm25-pool", type=int, default=3000, help="BM25 pre-filter pool size before full scoring")
    parser.add_argument("--debug", action="store_true", help="Print score breakdowns for top-20")
    args = parser.parse_args()

    import time
    t0 = time.time()

    print(f"[1/4] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    print(f"[2/4] Building BM25 index and pre-filtering to top {args.bm25_pool}...")
    t1 = time.time()

    # Build corpus
    corpus_texts = [candidate_text(c).split() for c in candidates]
    bm25 = BM25Okapi(corpus_texts)
    del corpus_texts  # free memory
    gc.collect()

    # Score against JD query
    query = JD_QUERY_TERMS
    raw_bm25_scores = bm25.get_scores(query)
    del bm25
    gc.collect()

    # Normalise BM25 scores to 0-1
    bm25_max = raw_bm25_scores.max()
    bm25_scores_norm = raw_bm25_scores / bm25_max if bm25_max > 0 else raw_bm25_scores

    # Take top-N by BM25 for full scoring
    top_bm25_idx = raw_bm25_scores.argsort()[::-1][:args.bm25_pool]
    print(f"      BM25 done in {time.time()-t1:.1f}s, pre-filtered to {len(top_bm25_idx)} candidates")

    print(f"[3/4] Compound scoring {len(top_bm25_idx)} candidates...")
    t2 = time.time()

    scored = []
    for idx in top_bm25_idx:
        c = candidates[idx]
        bm25_s = float(bm25_scores_norm[idx])
        final_score, breakdown = score_candidate(c, bm25_score=bm25_s)
        scored.append((final_score, c, breakdown))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    top100 = scored[:args.top_n]
    print(f"      Scoring done in {time.time()-t2:.1f}s")

    if args.debug:
        print("\n=== TOP 20 SCORE BREAKDOWNS ===")
        for i, (score, c, bd) in enumerate(top100[:20]):
            p = c["profile"]
            print(f"\nRank {i+1}: {c['candidate_id']} | {p['current_title']} @ {p['current_company']} | {p['years_of_experience']}yrs")
            print(f"  Final score: {score:.4f}")
            for k, v in bd.items():
                if k != "final_score":
                    print(f"  {k}: {v}")

    print(f"[4/4] Writing submission CSV to {args.out}...")
    t3 = time.time()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (score, c, breakdown) in enumerate(top100, start=1):
            cid = c["candidate_id"]
            reasoning = generate_reasoning(c, breakdown)
            writer.writerow([cid, rank, round(score, 6), reasoning])

    total_time = time.time() - t0
    print(f"      Done in {time.time()-t3:.1f}s")
    print(f"\n✓ Submission written: {args.out}")
    print(f"✓ Total pipeline time: {total_time:.1f}s")
    print(f"✓ Top score: {top100[0][0]:.4f} | Bottom (rank 100) score: {top100[-1][0]:.4f}")

    # Quick honeypot self-check
    honeypot_flagged = sum(1 for _, c, bd in top100 if bd["honeypot_penalty"] < 0.5)
    print(f"✓ Honeypot check: {honeypot_flagged} candidates with penalty < 0.5 in top 100 (limit: 10)")
    if honeypot_flagged > 10:
        print(f"  ⚠ WARNING: honeypot rate {honeypot_flagged}% exceeds 10% disqualification threshold!")


if __name__ == "__main__":
    main()
