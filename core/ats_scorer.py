from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cau hinh trong so diem (tong = 100)
# ---------------------------------------------------------------------------
SECTION_WEIGHTS: Dict[str, float] = {
    "skills":     0.45,
    "experience": 0.30,
    "education":  0.15,
    "summary":    0.10,
}

SEMANTIC_MATCH_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Synonym map
# ---------------------------------------------------------------------------
SYNONYM_GROUPS: List[List[str]] = [
    ["machine learning", "ml", "ai", "artificial intelligence", "deep learning", "neural network"],
    ["nlp", "natural language processing", "text mining", "language model"],
    ["computer vision", "cv", "image processing", "object detection"],
    ["python", "py"],
    ["javascript", "js", "ecmascript"],
    ["typescript", "ts"],
    ["react", "reactjs", "react.js"],
    ["vue", "vuejs", "vue.js"],
    ["node", "nodejs", "node.js"],
    ["postgresql", "postgres", "psql"],
    ["mongodb", "mongo"],
    ["kubernetes", "k8s"],
    ["amazon web services", "aws"],
    ["google cloud platform", "gcp", "google cloud"],
    ["microsoft azure", "azure"],
    ["continuous integration", "ci/cd", "devops", "continuous deployment"],
    ["rest api", "restful", "rest", "web api"],
    ["sql", "relational database", "rdbms"],
    ["git", "version control", "github", "gitlab"],
    ["agile", "scrum", "kanban"],
    ["data science", "data analysis", "data analytics"],
    ["big data", "hadoop", "spark", "data engineering"],
    ["software engineer", "software developer", "programmer", "developer"],
    ["backend", "back-end", "server-side"],
    ["frontend", "front-end", "client-side", "ui development"],
    ["full stack", "fullstack", "full-stack"],
]

_SYNONYM_MAP: Dict[str, str] = {}
for _group in SYNONYM_GROUPS:
    _canonical = _group[0]
    for _term in _group:
        _SYNONYM_MAP[_term.lower()] = _canonical


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SkillMatchResult:
    jd_skill: str
    matched_cv_skill: Optional[str]
    similarity_score: float
    match_type: str   # "exact" | "synonym" | "semantic" | "missing"

    @property
    def is_matched(self) -> bool:
        return self.match_type != "missing"


@dataclass
class ATSResult:
    total_score: float
    section_scores: Dict[str, float]
    skill_matches: List[SkillMatchResult] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    matched_skills: List[str] = field(default_factory=list)
    skill_coverage: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 1),
            "section_scores": {k: round(v, 1) for k, v in self.section_scores.items()},
            "skill_coverage_pct": round(self.skill_coverage * 100, 1),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "skill_details": [
                {
                    "jd_skill": m.jd_skill,
                    "cv_skill": m.matched_cv_skill,
                    "score": round(m.similarity_score, 3),
                    "type": m.match_type,
                }
                for m in self.skill_matches
            ],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------
class EmbeddingEngine:
    """Sentence-Transformers voi TF-IDF fallback. Singleton."""
    _instance: Optional["EmbeddingEngine"] = None
    _model = None
    _use_sentence_transformers: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_sentence_transformers = True
            logger.info("Sentence-Transformers loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Sentence-Transformers unavailable ({e}). Fallback to TF-IDF.")
            self._use_sentence_transformers = False

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._use_sentence_transformers:
            return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        return vec.fit_transform(texts).toarray()

    def similarity(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        emb_a = self.encode(texts_a)
        emb_b = self.encode(texts_b)
        return cosine_similarity(emb_a, emb_b)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_synonyms(term: str) -> str:
    return _SYNONYM_MAP.get(term.lower(), term.lower())


def _extract_skills_from_text(text: str) -> List[str]:
    tokens = re.split(r"[,\n\r;|*\u2022\u00b7\u25aa\-\u2013\u2014/\\]", text)
    skills = []
    for tok in tokens:
        tok = tok.strip(" \t\r\n.()[]")
        tok = re.sub(r"\s+", " ", tok)
        if 1 <= len(tok.split()) <= 5 and len(tok) >= 2:
            skills.append(tok.lower())
    seen = set()
    unique = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Semantic Skill Matcher
# ---------------------------------------------------------------------------
class SemanticSkillMatcher:
    def __init__(self, threshold: float = SEMANTIC_MATCH_THRESHOLD):
        self.threshold = threshold
        self._engine = EmbeddingEngine()

    def match(self, jd_skills: List[str], cv_skills: List[str]) -> List[SkillMatchResult]:
        if not jd_skills or not cv_skills:
            return [SkillMatchResult(s, None, 0.0, "missing") for s in jd_skills]

        cv_normalized = [_normalize_text(s) for s in cv_skills]
        cv_canonical  = [_apply_synonyms(s) for s in cv_normalized]
        jd_normalized = [_normalize_text(s) for s in jd_skills]
        sim_matrix    = self._engine.similarity(jd_normalized, cv_normalized)

        results: List[SkillMatchResult] = []
        for idx, jd_skill in enumerate(jd_skills):
            jd_norm    = _normalize_text(jd_skill)
            jd_canon   = _apply_synonyms(jd_norm)
            best_idx   = int(np.argmax(sim_matrix[idx]))
            best_score = float(sim_matrix[idx][best_idx])
            best_cv    = cv_skills[best_idx]

            if jd_norm in cv_normalized:
                results.append(SkillMatchResult(jd_skill, best_cv, 1.0, "exact"))
            elif jd_canon in cv_canonical:
                syn_idx = cv_canonical.index(jd_canon)
                results.append(SkillMatchResult(jd_skill, cv_skills[syn_idx], 0.95, "synonym"))
            elif best_score >= self.threshold:
                results.append(SkillMatchResult(jd_skill, best_cv, best_score, "semantic"))
            else:
                results.append(SkillMatchResult(jd_skill, None, best_score, "missing"))

        return results


# ---------------------------------------------------------------------------
# Section Scorer
# ---------------------------------------------------------------------------
class SectionScorer:
    def __init__(self):
        self._engine = EmbeddingEngine()

    def score(self, cv_text: str, jd_text: str) -> float:
        if not cv_text.strip() or not jd_text.strip():
            return 0.0
        sim = self._engine.similarity([_normalize_text(cv_text)], [_normalize_text(jd_text)])
        return float(np.clip(sim[0][0], 0.0, 1.0))


# ---------------------------------------------------------------------------
# ATSScorer — main API
# ---------------------------------------------------------------------------
class ATSScorer:
    """
    Diem vao chinh cua Member 4.

    Cach dung::

        scorer = ATSScorer()
        result = scorer.score(cv_sections, jd_text, jd_skills)
        print(result.to_dict())
    """

    def __init__(self, threshold: float = SEMANTIC_MATCH_THRESHOLD):
        self._skill_matcher  = SemanticSkillMatcher(threshold=threshold)
        self._section_scorer = SectionScorer()

    def score(
        self,
        cv_sections: Dict[str, str],
        jd_text: str,
        jd_skills: Optional[List[str]] = None,
    ) -> ATSResult:
        """
        Cham diem CV theo JD.

        Args:
            cv_sections: output tu CVParser.extract_sections()
                         keys: "SKILLS", "EXPERIENCE", "EDUCATION", "OTHERS"
            jd_text:     Toan bo noi dung JD (plain text)
            jd_skills:   (tuy chon) ky nang JD da trich san; tu dong trich neu None
        """
        if not jd_skills:
            jd_skills = _extract_skills_from_text(jd_text)

        cv_skill_text = cv_sections.get("SKILLS", "") or cv_sections.get("skills", "")
        cv_skills     = _extract_skills_from_text(cv_skill_text)

        skill_matches = self._skill_matcher.match(jd_skills, cv_skills)
        skill_score   = self._compute_skill_score(skill_matches)

        jd_parts  = self._split_jd_sections(jd_text)
        exp_score = self._section_scorer.score(cv_sections.get("EXPERIENCE", ""), jd_parts.get("experience", jd_text))
        edu_score = self._section_scorer.score(cv_sections.get("EDUCATION", ""),  jd_parts.get("education", jd_text))
        sum_score = self._section_scorer.score(cv_sections.get("OTHERS", ""),     jd_text)

        section_scores_raw = {
            "skills":     skill_score,
            "experience": exp_score,
            "education":  edu_score,
            "summary":    sum_score,
        }

        total = sum(section_scores_raw[s] * w for s, w in SECTION_WEIGHTS.items()) * 100

        matched  = [m.jd_skill for m in skill_matches if m.is_matched]
        missing  = [m.jd_skill for m in skill_matches if not m.is_matched]
        coverage = len(matched) / len(jd_skills) if jd_skills else 0.0

        return ATSResult(
            total_score=round(total, 1),
            section_scores={k: v * 100 for k, v in section_scores_raw.items()},
            skill_matches=skill_matches,
            missing_skills=missing,
            matched_skills=matched,
            skill_coverage=coverage,
            summary=self._generate_summary(total, coverage, matched, missing),
        )

    @staticmethod
    def _compute_skill_score(matches: List[SkillMatchResult]) -> float:
        if not matches:
            return 0.0
        return sum(m.similarity_score for m in matches) / len(matches)

    @staticmethod
    def _split_jd_sections(jd_text: str) -> Dict[str, str]:
        sections: Dict[str, List[str]] = {
            "requirements": [], "experience": [], "education": [], "skills": [], "other": []
        }
        current = "other"
        mapping = {
            "requirement": "requirements", "qualification": "requirements",
            "experience": "experience",
            "education": "education", "degree": "education",
            "skill": "skills", "technical": "skills", "competenc": "skills",
        }
        for line in jd_text.splitlines():
            low = line.lower().strip()
            for kw, sec in mapping.items():
                if kw in low and len(low.split()) <= 6:
                    current = sec
                    break
            sections[current].append(line)
        return {k: "\n".join(v) for k, v in sections.items()}

    @staticmethod
    def _generate_summary(total: float, coverage: float, matched: List[str], missing: List[str]) -> str:
        if total >= 80:
            verdict = "Rat phu hop"
        elif total >= 60:
            verdict = "Kha phu hop"
        elif total >= 40:
            verdict = "Can cai thien"
        else:
            verdict = "Chua phu hop"
        lines = [
            f"Danh gia: {verdict}",
            f"Tong diem: {total:.1f}/100",
            f"Ty le ky nang dap ung: {coverage * 100:.0f}% ({len(matched)}/{len(matched)+len(missing)} ky nang)",
        ]
        if matched:
            lines.append(f"Ky nang phu hop: {', '.join(matched[:8])}{'...' if len(matched) > 8 else ''}")
        if missing:
            lines.append(f"Ky nang con thieu: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------
def score_cv(
    cv_sections: Dict[str, str],
    jd_text: str,
    jd_skills: Optional[List[str]] = None,
    threshold: float = SEMANTIC_MATCH_THRESHOLD,
) -> ATSResult:
    """
    Goi nhanh khong can khoi tao ATSScorer.

    Vi du::

        from core.ats_scorer import score_cv

        result = score_cv(
            cv_sections={"SKILLS": "Python, Machine Learning, Docker"},
            jd_text="We need AI engineers with experience in Python and cloud deployment.",
            jd_skills=["Python", "AI", "cloud deployment"],
        )
        print(result.to_dict())
    """
    return ATSScorer(threshold=threshold).score(cv_sections, jd_text, jd_skills)