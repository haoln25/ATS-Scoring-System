import streamlit as st
import pdfplumber
from docx import Document
import pandas as pd
from pathlib import Path
import sys
import os
import re
import hashlib
from typing import Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from core.cv_parser import (
    CVParser,
    join_spaced_letters,
    fix_merged_titles,
    fix_unicode,
    normalize_whitespace,
)
from core.ats_scorer import ATSScorer, SEMANTIC_MATCH_THRESHOLD

load_dotenv()

st.set_page_config(
    page_title="ATS Scoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .score-good {
        color: #28a745;
        font-weight: bold;
    }
    .score-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .score-danger {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FILE PROCESSING
# ============================================================================

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/[\w\-]+|github\.com/[\w\-/]+)",
    re.I,
)
PHONE_RE = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def clean_text(text: str) -> str:
    """Light cleanup so scoring/extraction is more stable."""
    text = join_spaced_letters(text or "")
    text = fix_merged_titles(text)
    text = fix_unicode(text)
    text = normalize_whitespace(text)
    return text.strip()


def mask_pii_for_llm(text: str) -> str:
    """Best-effort PII masking before sending to external LLM APIs."""
    if not text:
        return ""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = LINK_RE.sub("[LINK]", text)
    return text


def extract_skill_tokens(text: str) -> list[str]:
    """Skill tokenization similar to Member 4 scorer (keeps phrases up to 5 words)."""
    if not text:
        return []
    tokens = re.split(r"[,\n\r;|*\u2022\u00b7\u25aa\-\u2013\u2014/\\]", text)
    skills: list[str] = []
    for tok in tokens:
        tok = tok.strip(" \t\r\n.()[]")
        tok = re.sub(r"\s+", " ", tok)
        if 1 <= len(tok.split()) <= 5 and len(tok) >= 2:
            skills.append(tok)
    return _dedupe_preserve_order(skills)


def dedupe_entities(entities: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for ent in entities:
        label = str(ent.get("label", "")).strip()
        text = str(ent.get("text", "")).strip()
        if not label or not text:
            continue
        key = (label.lower(), text.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


@st.cache_resource(show_spinner=False)
def get_spacy_nlp(model_name: str):
    import spacy

    return spacy.load(model_name)


def extract_spacy_entities(text: str, spacy_model_name: str) -> list[dict]:
    """Optional spaCy extraction for extra entities (not used for scoring)."""
    try:
        nlp = get_spacy_nlp(spacy_model_name)
    except Exception as e:
        st.warning(
            f"Could not load spaCy model '{spacy_model_name}'. "
            f"Install with: python -m spacy download {spacy_model_name}. Error: {e}"
        )
        return []

    label_map = {
        "PERSON": "NAME",
        "ORG": "COMPANIES_WORKED_AT",
        "GPE": "LOCATION",
        "DATE": "GRADUATION_YEAR",
    }

    doc = nlp(text)
    ents: list[dict] = []
    for ent in doc.ents:
        mapped = label_map.get(ent.label_, ent.label_)
        ents.append({
            "text": ent.text,
            "label": mapped,
            "confidence": 0.7,
        })
    return ents


@st.cache_resource(show_spinner=False)
def get_ats_scorer(skill_threshold: float) -> ATSScorer:
    return ATSScorer(threshold=skill_threshold)


def run_analysis(
    cv_text: str,
    jd_text: str,
    *,
    skill_threshold: float,
    use_spacy: bool,
    spacy_model_name: Optional[str],
) -> dict[str, Any]:
    cv_clean = clean_text(cv_text)
    jd_clean = clean_text(jd_text)

    parser = CVParser()
    cv_sections = parser.extract_sections(cv_clean)

    scorer = get_ats_scorer(skill_threshold)
    ats_result = scorer.score(cv_sections=cv_sections, jd_text=jd_clean)
    ats_dict = ats_result.to_dict()

    ats_score = int(round(float(ats_dict.get("total_score", 0.0))))
    section_scores = ats_dict.get("section_scores", {})
    skill_coverage_pct = float(ats_dict.get("skill_coverage_pct", 0.0))
    matched_skills = ats_dict.get("matched_skills", [])
    missing_skills = ats_dict.get("missing_skills", [])
    skill_details = ats_dict.get("skill_details", [])

    def build_auto_summary_en(
        total_score: int,
        coverage_pct: float,
        matched: list[str],
        missing: list[str],
    ) -> str:
        if total_score >= 80:
            verdict = "Excellent fit"
        elif total_score >= 60:
            verdict = "Good fit"
        elif total_score >= 40:
            verdict = "Needs improvement"
        else:
            verdict = "Low fit"

        denom = len(matched) + len(missing)
        coverage_line = (
            f"Skill coverage: {coverage_pct:.0f}% ({len(matched)}/{denom} skills)" if denom else "Skill coverage: N/A"
        )

        lines = [
            f"Overall: {verdict}",
            f"Total score: {total_score}/100",
            coverage_line,
        ]
        if matched:
            lines.append(
                f"Top matched: {', '.join(matched[:8])}{'...' if len(matched) > 8 else ''}"
            )
        if missing:
            lines.append(
                f"Top missing: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"
            )
        return "\n".join(lines)

    # Build basic entities for UI (skills + a few PII-like fields from CV)
    cv_skills = extract_skill_tokens(cv_sections.get("SKILLS", ""))
    jd_skills = [d.get("jd_skill") for d in skill_details if d.get("jd_skill")]
    jd_skills = _dedupe_preserve_order([str(s) for s in jd_skills])

    cv_entities: list[dict] = [
        {"label": "SKILLS", "text": s, "confidence": 0.7} for s in cv_skills
    ]
    jd_entities: list[dict] = [
        {"label": "SKILLS", "text": s, "confidence": 0.7} for s in jd_skills
    ]

    emails = _dedupe_preserve_order(EMAIL_RE.findall(cv_clean))
    for email in emails[:3]:
        cv_entities.append({"label": "EMAIL_ADDRESS", "text": email, "confidence": 0.9})

    links = _dedupe_preserve_order(LINK_RE.findall(cv_clean))
    for link in links[:3]:
        cv_entities.append({"label": "LINKS", "text": link, "confidence": 0.85})

    if use_spacy and spacy_model_name:
        cv_entities.extend(extract_spacy_entities(cv_clean, spacy_model_name))
        jd_entities.extend(extract_spacy_entities(jd_clean, spacy_model_name))

    cv_entities = dedupe_entities(cv_entities)
    jd_entities = dedupe_entities(jd_entities)

    score_explanation = {
        "Total": f"{ats_score}/100",
        "Skills (45%)": f"{section_scores.get('skills', 0):.1f}/100 | coverage {skill_coverage_pct:.1f}%",
        "Experience (30%)": f"{section_scores.get('experience', 0):.1f}/100",
        "Education (15%)": f"{section_scores.get('education', 0):.1f}/100",
        "Summary (10%)": f"{section_scores.get('summary', 0):.1f}/100",
        "Skill matches": f"matched={len(matched_skills)} missing={len(missing_skills)}",
        "Auto summary": build_auto_summary_en(ats_score, skill_coverage_pct, matched_skills, missing_skills),
    }

    missing_keywords: dict[str, list[str]] = {}
    if missing_skills:
        missing_keywords["SKILLS"] = missing_skills
    if not emails:
        missing_keywords["EMAIL_ADDRESS"] = ["email address"]

    return {
        "cv_entities": cv_entities,
        "jd_entities": jd_entities,
        "ats_score": ats_score,
        "score_explanation": score_explanation,
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "missing_keywords": missing_keywords,
        "skill_details": skill_details,
    }

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    return text


def extract_text_from_docx(docx_file) -> str:
    text = ""
    try:
        doc = Document(docx_file)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""
    return text


def extract_text_from_txt(txt_file) -> str:
    try:
        return txt_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        return ""


def extract_cv_text(uploaded_file) -> str:
    if not uploaded_file:
        return ""
    
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""


def show_entities_table(entities, title: str):
    if not entities:
        st.info(f"No entities found for {title}")
        return
    
    # Group entities by label
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for e in entities:
        label = e.get("label", "")
        grouped[label].append(e)
    
    # Prepare data for display
    rows = []
    for label, items in grouped.items():
        if label == "SKILLS":
            # Consolidate skills into a single row
            skills_list = [item.get("text", "") for item in items]
            avg_confidence = sum(item.get("confidence", 0) for item in items) / len(items) if items else 0
            rows.append({
                "Label": label,
                "Entity": ", ".join(skills_list),
                "Confidence": f"{avg_confidence*100:.1f}%"
            })
        else:
            # Keep other labels separate
            for item in items:
                rows.append({
                    "Label": label,
                    "Entity": item.get("text", ""),
                    "Confidence": f"{item.get('confidence', 0)*100:.1f}%"
                })
    
    df = pd.DataFrame(rows)
    st.subheader(f"{title} ({len(entities)} entities)")
    st.dataframe(df, use_container_width=True, hide_index=True)


def get_improvement_suggestions(missing_keywords: dict) -> dict:
    suggestions = {
        "NAME": "Include your full name prominently at the top of your CV.",
        "EMAIL_ADDRESS": "Add your email address for easy contact.",
        "LOCATION": "Include your current location or willingness to relocate.",
        "DESIGNATION": "Clearly specify your job title and roles held.",
        "COMPANIES_WORKED_AT": "List all companies where you've worked with clear dates.",
        "COLLEGE_NAME": "Mention your educational institution names.",
        "GRADUATION_YEAR": "Include graduation dates for all degrees.",
        "SKILLS": "Add technical and professional skills relevant to the job.",
        "EXPERIENCE": "Detail your years of experience in relevant domains.",
        "PROJECTS": "Highlight significant projects and achievements.",
        "CERTIFICATIONS": "List relevant professional certifications.",
        "LANGUAGES": "Mention any languages you're proficient in.",
    }
    
    result = {}
    for label, keywords in missing_keywords.items():
        if keywords and label in suggestions:
            result[label] = {
                "suggestion": suggestions[label],
                "keywords": keywords[:5],  # Show top 5
            }
    
    return result


def show_suggestions(missing_keywords: dict):
    suggestions = get_improvement_suggestions(missing_keywords)
    
    if not suggestions:
        st.success("Your CV matches all key categories in the job description.")
        return
    
    st.warning(f"Missing {len(suggestions)} categories:")
    
    for label, data in suggestions.items():
        with st.expander(f"{label}"):
            st.write(data["suggestion"])
            st.write(f"**Required keywords:** {', '.join(data['keywords'])}")


def show_score_breakdown(explanation: dict):
    st.subheader("Score Breakdown")
    
    df = pd.DataFrame([
        {"Category": label, "Details": details}
        for label, details in explanation.items()
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def build_llm_prompt(
    ats_score: int,
    matched_skills: list[str],
    missing_skills: list[str],
    score_explanation: dict,
    cv_text: str,
    jd_text: str
) -> str:
    cv_preview = mask_pii_for_llm((cv_text or "")[:1800])
    jd_preview = (jd_text or "")[:1800]
    matched_preview = matched_skills[:20]
    missing_preview = missing_skills[:20]

    breakdown_lines = "\n".join([f"- {k}: {v}" for k, v in (score_explanation or {}).items()])
    matched_line = ", ".join(matched_preview) if matched_preview else "(Not clearly detected)"
    missing_line = ", ".join(missing_preview) if missing_preview else "(No notable missing skills)"

    return f"""
You are a senior technical recruiter and career coach.
Evaluate how well the candidate matches the job description using ONLY the data below.

OUTPUT REQUIREMENTS (mandatory):
1) Write in clear, natural English with a supportive but direct tone.
2) Output exactly 2 sections with these exact titles (no extra sections):
   AI CV Review
   Supplemental Learning Roadmap
3) Do NOT invent facts beyond the provided data. If you cannot conclude something, say "Insufficient data".
4) Prioritize the highest-impact actions first.

INPUT DATA:
- ATS score: {ats_score}/100
- Matched skills: {matched_line}
- Missing skills: {missing_line}
- Score breakdown:\n{breakdown_lines}

CV excerpt (PII-masked, truncated):
{cv_preview}

JD excerpt (truncated):
{jd_preview}

CONTENT GUIDELINES:
AI CV Review:
- Give an overall fit assessment aligned with the ATS score.
- List 3 strengths tied to matched skills and how to highlight them in the CV.
- List 3 critical gaps tied to missing skills and why they matter.
- Provide 5 concrete CV edits (from quick wins to harder changes) focused on ATS and credibility.

Supplemental Learning Roadmap:
- Provide a 30-60-90 day plan.
- For each phase include: (1) skills to learn (2) suggested resources (3) a mini-project idea (4) a self-evaluation method.
""".strip()


def generate_ai_feedback(
    provider: str,
    model_name: str,
    api_key: str,
    prompt: str
) -> str:
    if provider == "OpenAI (ChatGPT)":
        client = OpenAI(api_key=api_key)
        # Prefer Responses API when available (newer SDK), otherwise Chat Completions.
        if hasattr(client, "responses"):
            try:
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    temperature=0.4,
                    max_output_tokens=900,
                )
                output_text = getattr(response, "output_text", None)
                if output_text:
                    return str(output_text).strip()
            except Exception:
                pass

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=900,
        )
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 900,
        }
    )
    return (response.text or "").strip()


def get_fallback_provider(provider: str) -> str:
    return "Google Gemini" if provider == "OpenAI (ChatGPT)" else "OpenAI (ChatGPT)"


# ============================================================================
# BUTTON CALLBACKS
# ============================================================================

def load_example_cv():
    example_cv = """
Abhishek Jha
Email: abhishek@example.com
Location: Bengaluru, Karnataka

EXPERIENCE
Application Development Associate at Accenture (Nov 2017 - Present)
- Developing chatbot backend with Oracle PeopleSoft
- Training NLP models for user utterances
- Building REST APIs and microservices

EDUCATION
B.E in Information Science and Engineering
B.V.B College of Engineering, Hubli (2013-2017)

SKILLS
Python, Java, JavaScript, C++, Database Management, Oracle PeopleSoft,
Machine Learning, NLP, Docker, REST APIs, Git
    """
    st.session_state.cv_text = example_cv
    st.session_state.show_success_example = True


def load_abhishek_test():
    abhishek_cv = """Abhishek Jha Application Development Associate - Accenture Bengaluru, Karnataka - Email me on Indeed: indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a • To work for an organization which provides me the opportunity to improve my skills and knowledge for my individual and company's growth in best possible ways. Willing to relocate to: Bangalore, Karnataka WORK EXPERIENCE Application Development Associate Accenture - November 2017 to Present Role: Currently working on Chat-bot. Developing Backend Oracle PeopleSoft Queries for the Bot which will be triggered based on given input. Also, Training the bot for different possible utterances (Both positive and negative), which will be given as input by the user. EDUCATION B.E in Information science and engineering B.v.b college of engineering and technology - Hubli, Karnataka August 2013 to June 2017 SKILLS C, Database, Database Management System, Java ADDITIONAL INFORMATION Technical Skills Programming language: C, C++, Java • Oracle PeopleSoft • Internet Of Things • Machine Learning • Database Management System • Computer Networks • Operating System worked on: Linux, Windows, Mac"""
    jd_text = """POSITION: Senior Full Stack Developer COMPANY: TechCorp India REQUIREMENTS: Experience: 5-7 years of full-stack development experience, Experience with JavaScript, Python, Java, C++ Technical Skills Required: JavaScript, React.js, Node.js, Python, Java, PostgreSQL, MySQL, MongoDB, Docker, Git, REST API Design, Oracle Database Nice to Have: Machine Learning experience, NLP expertise, Chatbot development, Microservices architecture, Kubernetes"""
    st.session_state.cv_text = abhishek_cv
    st.session_state.jd_text = jd_text
    st.session_state.show_success_abhishek = True


def load_bullet_test():
    bullet_cv = """Skills: • Python • Java • JavaScript • C++ • Database Management • Oracle PeopleSoft • Machine Learning • NLP • Docker • REST APIs • Microservices • Git"""
    st.session_state.cv_text = bullet_cv
    st.session_state.show_success_bullet = True


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    
    st.title("Applicant Tracking System (ATS)")
    st.markdown("*Analyze your CV against job descriptions using AI-powered NER (Spacy + BERT)*")
    with st.sidebar:
        st.header("Configuration")
        
        use_spacy = st.checkbox(
            "Enable Spacy NER",
            value=False,
            help="Load Spacy model for additional entity detection (slower but more accurate)"
        )
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(SEMANTIC_MATCH_THRESHOLD),
            step=0.05,
            help="Semantic skill-match threshold (≈0.60 recommended). Higher = stricter matching."
        )
        
        show_raw_entities = st.checkbox(
            "Show Raw Entities",
            value=False,
            help="Display all extracted entities without filtering"
        )
        
        spacy_model = None
        if use_spacy:
            spacy_model = st.selectbox(
                "Spacy Model",
                options=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
                help="Select Spacy model for NER. Install with: python -m spacy download [model]"
            )

        st.divider()
        st.markdown("### LLM Integration")
        provider = st.selectbox(
            "LLM Provider",
            options=["OpenAI (ChatGPT)", "Google Gemini"],
            index=0
        )

        if provider == "OpenAI (ChatGPT)":
            default_key = os.getenv("OPENAI_API_KEY", "")
            default_model = "gpt-4o-mini"
            api_help = "Enter OpenAI API key or set OPENAI_API_KEY"
        else:
            default_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            default_model = "gemini-1.5-flash"
            api_help = "Enter Gemini API key or set GEMINI_API_KEY / GOOGLE_API_KEY"

        llm_model_name = st.text_input("Model name", value=default_model).strip()
        llm_api_key = st.text_input("API Key", value=default_key, type="password", help=api_help)
        enable_llm_fallback = st.checkbox(
            "Enable provider fallback",
            value=True,
            help="If the selected provider fails, automatically retry with the other provider."
        )
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This ATS system uses:
        - **Spacy NER** for linguistic entity extraction
        - **Semantic scoring** (Sentence-Transformers with TF-IDF fallback)
        - **Skill matching** (exact/synonym/semantic)
        
        **Score Calculation:**
        - Skills: 45%
        - Experience: 30%
        - Education: 15%
        - Summary: 10%
        """)
    # Input section
    st.header("Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your CV")
        uploaded_resume = st.file_uploader(
            "Upload CV/Resume",
            type=['pdf', 'docx', 'txt'],
            key="cv_upload"
        )
        
        # Option to paste CV text
        paste_cv = st.checkbox("Or paste CV text", key="paste_cv")
        
        if paste_cv:
            st.text_area(
                "Paste your CV text here",
                height=200,
                value=st.session_state.cv_text,
                placeholder="Paste your CV content...",
                key="cv_text"
            )
        elif uploaded_resume:
            cv_text = extract_cv_text(uploaded_resume)
            if cv_text:
                st.session_state.cv_text = cv_text
                st.success(f"Extracted {len(cv_text)} characters from CV.")
    
    with col2:
        st.subheader("Job Description")
        st.text_area(
            "Enter Job Description",
            height=200,
            value=st.session_state.jd_text,
            placeholder="Paste the job description here...",
            key="jd_text"
        )
    # Analysis section
    if st.button("Analyze CV Against Job Description", type="primary", use_container_width=True):
        if not st.session_state.cv_text:
            st.error("Please provide CV text (either by uploading or pasting)")
            return
        if not st.session_state.jd_text:
            st.error("Please provide a job description")
            return
        
        with st.spinner("Scoring CV against Job Description..."):
            try:
                # Use Member 4 scorer for stable ATS + skill match (no BERT weights required).
                skill_threshold = float(
                    max(0.0, min(1.0, conf_threshold if conf_threshold is not None else SEMANTIC_MATCH_THRESHOLD))
                )
                result = run_analysis(
                    st.session_state.cv_text,
                    st.session_state.jd_text,
                    skill_threshold=skill_threshold,
                    use_spacy=use_spacy,
                    spacy_model_name=spacy_model if use_spacy else None,
                )
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                return
        # ====== RESULTS DISPLAY ======
        st.divider()
        st.header("Analysis Results")
        # ATS Score - Main Display
        ats_score = result["ats_score"]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if ats_score >= 75:
                st.markdown(f"<div class='score-good'>ATS Score: {ats_score}/100</div>", 
                           unsafe_allow_html=True)
                st.success("Your CV is a good match for this job!")
            elif ats_score >= 50:
                st.markdown(f"<div class='score-warning'>ATS Score: {ats_score}/100</div>", 
                           unsafe_allow_html=True)
                st.warning("Your CV has some gaps. Consider the suggestions below.")
            else:
                st.markdown(f"<div class='score-danger'>ATS Score: {ats_score}/100</div>", 
                           unsafe_allow_html=True)
                st.error("Your CV needs significant improvements. See suggestions below.")
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Score Analysis",
            "Entity Comparison",
            "Suggestions",
            "Raw Data",
            "AI Coach"
        ])
        # Tab 1: Score Analysis
        with tab1:
            show_score_breakdown(result["score_explanation"])
            
            # Missing skills summary
            missing_skills = result["missing_skills"]
            if missing_skills:
                st.warning(f"Missing {len(missing_skills)} required skills:")
                st.write(", ".join(missing_skills[:10]))
        # Tab 2: Entity Comparison
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                show_entities_table(result["cv_entities"], "CV Entities")
            
            with col2:
                show_entities_table(result["jd_entities"], "Job Description Entities")
        # Tab 3: Suggestions
        with tab3:
            show_suggestions(result["missing_keywords"])
            
            # Additional tips
            st.divider()
            st.subheader("General Tips")
            st.markdown("""
            1. **Keywords Matter**: Include specific skills and technologies mentioned in the job description
            2. **Be Specific**: Use exact company names and job titles (avoid generic descriptions)
            3. **Quantify Experience**: Mention years of experience and metrics (e.g., "5+ years")
            4. **Format Clearly**: Use clear sections (Experience, Education, Skills) for better parsing
            5. **Use Industry Terms**: Use the same terminology as the job posting
            """)
        # Tab 4: Raw Data
        with tab4:
            st.subheader("Raw Extraction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "cv_entities_count": len(result["cv_entities"]),
                    "cv_entities": [
                        {"label": e["label"], "text": e["text"], "confidence": e["confidence"]}
                        for e in result["cv_entities"][:10]
                    ]
                })
            
            with col2:
                st.json({
                    "jd_entities_count": len(result["jd_entities"]),
                    "jd_entities": [
                        {"label": e["label"], "text": e["text"], "confidence": e["confidence"]}
                        for e in result["jd_entities"][:10]
                    ]
                })
            
            if show_raw_entities:
                st.write("**Full CV Entities:**")
                st.json(result["cv_entities"])
                st.write("**Full JD Entities:**")
                st.json(result["jd_entities"])

            st.divider()
            st.subheader("Skill Match Details")
            st.json(result.get("skill_details", [])[:30])

        with tab5:
            st.subheader("AI CV Review and Learning Roadmap")
            if not llm_api_key:
                st.info("Enter API key in the sidebar to enable GenAI analysis.")
            else:
                if not llm_model_name.strip():
                    st.error("Model name is required for LLM generation.")
                    return

                matched_skills = result.get("matched_skills", [])
                missing_skills = result.get("missing_skills", [])
                prompt = build_llm_prompt(
                    ats_score=ats_score,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    score_explanation=result.get("score_explanation", {}),
                    cv_text=st.session_state.cv_text,
                    jd_text=st.session_state.jd_text,
                )

                with st.expander("View Prompt Engineering", expanded=False):
                    st.code(prompt, language="text")

                prompt_fingerprint = hashlib.sha256(
                    f"{provider}|{llm_model_name}|{prompt}".encode("utf-8")
                ).hexdigest()
                cached_feedback = st.session_state.get("ai_feedback_text")
                cached_fp = st.session_state.get("ai_feedback_fingerprint")

                if cached_feedback and cached_fp == prompt_fingerprint:
                    st.markdown(cached_feedback)
                    st.download_button(
                        "Download AI Feedback",
                        data=cached_feedback,
                        file_name="ai_cv_review_and_learning_roadmap.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    with st.spinner("Calling LLM to generate personalized feedback..."):
                        try:
                            ai_feedback = generate_ai_feedback(
                                provider=provider,
                                model_name=llm_model_name,
                                api_key=llm_api_key,
                                prompt=prompt,
                            )

                            if not ai_feedback:
                                raise ValueError("Empty response from LLM.")

                            st.session_state.ai_feedback_text = ai_feedback
                            st.session_state.ai_feedback_fingerprint = prompt_fingerprint

                            st.markdown(ai_feedback)
                            st.download_button(
                                "Download AI Feedback",
                                data=ai_feedback,
                                file_name="ai_cv_review_and_learning_roadmap.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )
                        except Exception as e:
                            if not enable_llm_fallback:
                                st.error(f"Error calling {provider}: {e}")
                                return

                            fallback_provider = get_fallback_provider(provider)
                            fallback_key = (
                                os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
                                if fallback_provider == "Google Gemini"
                                else os.getenv("OPENAI_API_KEY", "")
                            )
                            fallback_model = (
                                "gemini-1.5-flash" if fallback_provider == "Google Gemini" else "gpt-4o-mini"
                            )

                            if not fallback_key:
                                st.error(
                                    f"Error calling {provider}: {e}. "
                                    f"Fallback to {fallback_provider} is unavailable because API key is missing."
                                )
                                return

                            st.warning(
                                f"{provider} failed ({e}). Retrying with {fallback_provider} automatically."
                            )
                            try:
                                fallback_feedback = generate_ai_feedback(
                                    provider=fallback_provider,
                                    model_name=fallback_model,
                                    api_key=fallback_key,
                                    prompt=prompt,
                                )
                                if not fallback_feedback:
                                    st.error(f"{fallback_provider} returned an empty response.")
                                    return

                                st.session_state.ai_feedback_text = fallback_feedback
                                st.session_state.ai_feedback_fingerprint = prompt_fingerprint

                                st.markdown(fallback_feedback)
                                st.download_button(
                                    "Download AI Feedback",
                                    data=fallback_feedback,
                                    file_name="ai_cv_review_and_learning_roadmap.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                )
                                st.info(f"Response generated successfully by {fallback_provider}.")
                            except Exception as fallback_error:
                                st.error(
                                    f"Error calling {provider}: {e}. "
                                    f"Fallback with {fallback_provider} also failed: {fallback_error}"
                                )
    
    # Test section in sidebar
    with st.sidebar:
        st.divider()
        st.header("Quick Testing")
        
        col_test1, col_test2 = st.columns(2)
        
        with col_test1:
            st.button("Example CV", on_click=load_example_cv, key="btn_example")
        
        with col_test2:
            st.button("Abhishek Test", on_click=load_abhishek_test, key="btn_abhishek")
        
        col_test3, col_test4 = st.columns(2)
        
        with col_test3:
            st.button("Bullet Format", on_click=load_bullet_test, key="btn_bullet")
        
        # Show success messages
        if st.session_state.get("show_success_example", False):
            st.success("Example CV loaded.")
            st.session_state.show_success_example = False
        
        if st.session_state.get("show_success_abhishek", False):
            st.success("Test data loaded. Click Analyze.")
            st.session_state.show_success_abhishek = False
        
        if st.session_state.get("show_success_bullet", False):
            st.success("Bullet-point test data loaded.")
            st.info("This tests the bullet-splitting preprocessing (should extract 12 skills individually)")
            st.session_state.show_success_bullet = False


if __name__ == "__main__":
    main()
