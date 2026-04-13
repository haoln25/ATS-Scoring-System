"""CV Parser — Trích xuất, làm sạch text từ CV và ẩn danh PII."""

import re
from typing import Tuple, Dict, List
from pathlib import Path

# Optional imports
spacy = None
pdfplumber = None
Document = None

try:
    import spacy
except ImportError:
    pass

try:
    import pdfplumber
except ImportError:
    pass

try:
    from docx import Document
except ImportError:
    pass

class CVParserError(Exception):
    pass

TECH_WHITELIST = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin',
    'django', 'flask', 'fastapi', 'react', 'reactjs', 'angular', 'vue', 'vuejs', 'nodejs', 'node.js', 'spring', 'springboot',
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle', 'sqlserver',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'ec2', 's3', 'lambda',
    'git', 'github', 'gitlab', 'jira', 'postman', 'figma', 'vscode', 'scrum', 'system', 'project', 'rebuild',
    'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'spacy', 'nltk', 'sklearn', 'scikit',
    'html', 'css', 'sass', 'graphql', 'rest', 'api', 'oauth', 'jwt', 'grpc', 'mqtt', 'http', 'https',
    'gpa', 'btech', 'mtech', 'be', 'me', 'bs', 'ms', 'phd', 'ielts', 'toefl',
    'fpt', 'viettel', 'vng', 'tiki', 'shopee', 'lazada', 'hcmc', 'hanoi', 'saigon', 'vietnam'
}

NER_SKIP_PHRASES = {
    'ho chi minh', 'hanoi', 'ha noi', 'saigon', 'vietnam', 'district', 'city',
    'university', 'college', 'institute', 'school', 'transport', 'technology',
    'developed', 'implemented', 'handled', 'collaborated', 'working', 'worked',
    'senior', 'junior', 'intern', 'developer', 'engineer'
}

SECTION_KEYWORDS = {
    'EDUCATION': ['education', 'academic', 'qualification', 'degree', 'certification', 'background'],
    'EXPERIENCE': ['experience', 'work history', 'employment', 'career', 'history'],
    'SKILLS':     ['skills', 'competencies', 'technical skills', 'expertise'],
}

def join_spaced_letters(text: str) -> str:
    """Nối các chữ cái đơn lẻ (R E S U M E -> RESUME)"""
    text = re.sub(r'(?<=\b[A-Z])\s{1,2}(?=[A-Z]\b)', '', text)
    text = re.sub(r'([A-Z])\s{3,}([A-Z])', r'\1 \2', text)
    return text

def fix_merged_titles(text: str) -> str:
    """Khắc phục các lỗi dính chữ do PDF và dính dòng tiêu đề"""
    
    joined_titles = {
        'ACADEMICBACKGROUND': 'ACADEMIC BACKGROUND',
        'WORKHISTORY': 'WORK HISTORY',
        'TECHNICALEXPERTISE': 'TECHNICAL EXPERTISE',
        'PROFILESUMMARY': 'PROFILE SUMMARY',
        'PROFESSIONALSUMMARY': 'PROFESSIONAL SUMMARY',
        'EMPLOYMENTHISTORY': 'EMPLOYMENT HISTORY'
    }
    for joined, fixed in joined_titles.items():
        text = text.replace(joined, fixed)
        
    titles = ['SUMMARY', 'EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 'BACKGROUND', 'HISTORY', 'EXPERTISE', 'PROFILE']
    for t in titles:
        text = re.sub(rf'\b({t})([A-Z])', r'\1\n\2', text)
        text = re.sub(rf'\b({t})([^A-Za-z0-9\s])', r'\1\n\2', text)
        
    return text

def fix_unicode(text: str) -> str:
    """Sửa lỗi font, ký tự đặc biệt và xóa rác trang trí"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('¢', '-').replace('”', '"').replace('“', '"')
    
    text = text.replace('\u20ac', '')
    text = re.sub(r'(\u00e2\u20ac\u201d){2,}', '--', text)
    text = text.replace('\u00e2\u20ac\u201d', '-')
    text = re.sub(r'(\u00e2\u20ac\u00a2){2,}', '--', text)
    text = text.replace('\u00e2\u20ac\u00a2', '-')
    text = re.sub(r'(\u00e2\u20ac){2,}', '--', text)
    text = text.replace('\u00e2\u20ac', '-')
    text = re.sub(r'(\u00e2\u20ac\u00a6){2,}', '...', text)
    text = text.replace('\u00e2\u20ac\u00a6', '...')
    text = re.sub(r'(\u00e2\u017e\u00a4){2,}', ' ', text)
    text = text.replace('\u00e2\u017e\u00a4', '*')
    text = text.replace('\u00e2\u017e\u00a1', ' ')
    text = text.replace('\u00e2\u20ac\u0153', '"').replace('\u00e2\u0153', '"')
    text = text.replace('\u00e2\u201e\u00a2', '(TM)')
    text = re.sub(r'(\u00e2\u2013\u00ba){2,}', ' ', text)
    text = text.replace('\u00e2\u2013\u00ba', ' ')
    text = text.replace('\u00e2\u2013\u00bc', ' ')
    text = text.replace('\u00e2\u2014\u2020', ' ')
    text = text.replace('\u00e2\u02dc\u2026', '*')
    text = text.replace('\u00e2', '').replace('\u017e', 'z').replace('\u00a4', '')
    text = text.replace('\u00a1', '!').replace('\u00a6', '|').replace('\u0153', '"')
    text = text.replace('\u2026', '...').replace('\u2013', '-').replace('\u2014', '-')
    
    text = re.sub(r'([|"\-~=*]\s*){3,}', ' ', text)
    
    text = re.sub(r'([!*•\-])([A-Z])', r'\1 \2', text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()


class CVParser:
    def __init__(self, nlp=None):
        """
        Initialize CVParser with optional spacy model.
        If nlp is None, some methods will be disabled.
        """
        self.nlp = nlp

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == '.pdf':
            if pdfplumber is None:
                raise ImportError("pdfplumber is required for PDF parsing. Install it with: pip install pdfplumber")
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() or '' for page in pdf.pages)
        elif ext == '.docx':
            if Document is None:
                raise ImportError("python-docx is required for DOCX parsing. Install it with: pip install python-docx")
            return "\n".join(p.text for p in Document(file_path).paragraphs)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def get_tokens(self, text: str) -> List[str]:
        if self.nlp is None:
            return text.split()
        return [t.text for t in self.nlp(text)]

    def get_sentences(self, text: str) -> List[str]:
        if self.nlp is None:
            return [s.strip() for s in text.split('\n') if s.strip()]
        return [s.text.strip() for s in self.nlp(text).sents]

    def remove_stopwords(self, text: str) -> str:
        if self.nlp is None:
            return text
        return " ".join(t.text for t in self.nlp(text) if not t.is_stop and not t.is_punct)

    def get_lemmas(self, text: str) -> List[str]:
        if self.nlp is None:
            return text.split()
        return [t.lemma_ for t in self.nlp(text) if not t.is_stop and not t.is_punct]

    def mask_pii(self, text: str) -> str:
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', '[PHONE]', text)
        
        if self.nlp is not None:
            entities = []
            for ent in self.nlp(text).ents:
                if ent.label_ == 'PERSON':
                    low = ent.text.lower()
                    if not any(kw in low for kw in TECH_WHITELIST) and not any(kw in low for kw in NER_SKIP_PHRASES):
                        entities.append((ent.start_char, ent.end_char))
            
            for start, end in sorted(entities, reverse=True):
                original_text = text[start:end]
                newlines = '\n' * original_text.count('\n')
                text = text[:start] + '[PERSON]' + newlines + text[end:]
            
        return text

    def extract_sections(self, text: str) -> Dict[str, str]:
        sections = {'EDUCATION': [], 'EXPERIENCE': [], 'SKILLS': [], 'OTHERS': []}
        current = 'OTHERS'
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped: continue
            
            found = None
            low = stripped.lower()
            if len(stripped.split()) <= 5:
                for sec, kws in SECTION_KEYWORDS.items():
                    if any(kw in low for kw in kws):
                        found = sec
                        break
            current = found or current
            sections[current].append(stripped)
            
        return {k: "\n".join(v) for k, v in sections.items()}

    def parse_cv(self, file_path: str) -> Tuple[str, Dict[str, str]]:
        raw = self.extract_text(file_path)
        
        text = join_spaced_letters(raw)
        text = fix_merged_titles(text)  
        text = fix_unicode(text)        
        text = normalize_whitespace(text)
        
        sections = self.extract_sections(text)
        clean_full = self.mask_pii(text)
        masked_sections = {k: self.mask_pii(v) for k, v in sections.items()}
        
        return clean_full, masked_sections


def parse_cv(file_path: str, nlp=None) -> Tuple[str, Dict[str, str]]:
    """Parse CV file. nlp parameter is optional (spacy model)."""
    return CVParser(nlp=nlp).parse_cv(file_path)