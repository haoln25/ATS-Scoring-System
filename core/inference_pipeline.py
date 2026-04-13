import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.cv_parser import CVParser
from core.entity_extractor import EntityExtractor
from core.ats_scorer import calculate_ats_score, get_missing_keywords


def split_by_bullets(text: str) -> str:
    bullet_patterns = [
        r'[•●◦◾▪▫■□◻]',
        r'(?:^|\s)[-*]\s+',
        r'(?:^|\s)[0-9]+\.\s+',
    ]
    
    for pattern in bullet_patterns:
        text = re.sub(pattern, '\n', text)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


def preprocess_text(text: str) -> str:
    replacements = {
        'â€"': '–',    'â€"': '—',    'â€˜': ''',    'â€™': ''',
        'â€œ': '"',    'â€': '"',     'â€¢': '•',    'â€¦': '…',
        'Ã©': 'é',     'Ã¨': 'è',     'Ã¢': 'â',     'Ã´': 'ô',
        'Ã¼': 'ü',     'Ã±': 'ñ',     'Ã‹': 'Ë',     'Ã¡': 'á',
        'Ãº': 'ú',     'Ã®': 'î',     'Ã€': 'À',     'Ã¬': 'ì',
        'Ã™': 'Ù',     'Ã': 'Í',     'Ã–': 'Ö',     'Ã': 'Á',
        'ÃŒ': 'Ì',     'Ã‰': 'É',     'Ã': 'Ï',     'Ã«': 'ë',
        'Ã³': 'ó',     'Ãž': 'Þ',     'Ãš': 'Ú',     'Ã¦': 'æ',
        'Ã˜': 'Ø',     'ÃŸ': 'ß',     'Ã°': 'ð',     'Ã­': 'í',
        'Ãµ': 'õ',     'Ã¥': 'å',     'Ã¯': 'ï',     'Ã£': 'ã',
        'Ã¤': 'ä',     'Ã¶': 'ö',
    }
    
    for wrong_char, correct_char in replacements.items():
        text = text.replace(wrong_char, correct_char)
    
    text = split_by_bullets(text)
    
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


COMMON_SKILLS = {
    # Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin', 'scala',
    'perl', 'r programming', 'matlab', 'groovy', 'dart', 'elixir', 'clojure', 'haskell', 'erlang', 'fortran',
    
    # Frontend
    'react', 'angular', 'vue', 'svelte', 'ember', 'backbone', 'knockout', 'polymer', 'preact', 'next.js', 'gatsby',
    'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'material-ui', 'webpack', 'vite', 'parcel',
    
    # Backend
    'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'rails', 'laravel', 'symfony',
    'asp.net', 'asp.net core', 'quarkus', 'micronaut', 'grpc', 'graphql', 'rest api',
    
    # Databases
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'mariadb', 'oracle', 'sql server',
    'sqlite', 'dynamodb', 'firestore', 'couchdb', 'neo4j', 'influxdb', 'timescaledb',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'linode', 'vultr',
    'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci', 'terraform',
    'ansible', 'puppet', 'chef', 'salt', 'cloudformation', 'helm', 'prometheus', 'grafana', 'elk stack',
    
    # Testing & QA
    'junit', 'pytest', 'jest', 'mocha', 'chai', 'rspec', 'testng', 'selenium', 'cypress', 'playwright',
    'jmeter', 'loadrunner', 'postman', 'soapui', 'appium', 'xcode', 'android studio',
    
    # DevTools
    'git', 'github', 'gitlab', 'bitbucket', 'subversion', 'svn', 'mercurial', 'jira', 'confluence',
    'slack', 'asana', 'trello', 'monday.com', 'notion', 'vscode', 'intellij', 'eclipse',
    
    # Data & ML
    'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'spark', 'hadoop',
    'jupyter', 'anaconda', 'machine learning', 'deep learning', 'nlp', 'computer vision',
    
    # Other
    'linux', 'windows', 'macos', 'unix', 'shell', 'bash', 'powershell', 'sql', 'xml', 'json',
    'soap', 'http', 'https', 'websocket', 'oauth', 'jwt', 'microservices', 'serverless',
    'architecture', 'design patterns', 'agile', 'scrum', 'kanban',
}

def extract_skills_rule_based(text: str) -> List[Dict[str, Any]]:
    skills = []
    text_lower = text.lower()
    
    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            skills.append({
                "text": skill,
                "label": "SKILLS",
                "confidence": 0.7,  # Rule-based: lower confidence than ML
            })
    
    return skills


SYNONYM_MAP = {
    # Programming Languages
    'python3': 'python', 'python 3': 'python',
    'js': 'javascript', 'nodejs': 'node.js',
    'react.js': 'react', 'reactjs': 'react',
    'typescript.js': 'typescript', 'ts': 'typescript',
    'c++': 'cpp', 'c# ': 'csharp', 'c#': 'csharp',
    '.net': 'dotnet', '.net core': 'dotnet',
    'golang': 'go', 'ruby on rails': 'ruby',
    
    # Frameworks
    'django rest framework': 'django', 'drf': 'django',
    'expressjs': 'express', 'express.js': 'express',
    'nextjs': 'next.js', 'next js': 'next.js',
    'vuejs': 'vue', 'vue.js': 'vue',
    'angularjs': 'angular', 'angular.js': 'angular',
    
    # Databases
    'postgre sql': 'postgresql', 'postgres': 'postgresql',
    'maria db': 'mariadb',
    
    # Cloud & DevOps
    'amazon web services': 'aws', 'ec2 instance': 'ec2',
    's3 bucket': 's3', 'google cloud': 'gcp',
    'microsoft azure': 'azure', 'k8s': 'kubernetes',
    'docker compose': 'docker', 'docker container': 'docker',
    
    # Company Name Variations
    'infosys limited': 'infosys', 'tcs': 'tata consultancy services',
    'cognizant': 'cognizant technology solutions',
    'ibm india': 'ibm', 'ibm global': 'ibm',
    
    # Skill variations
    'machine learning': 'ml', 'natural language processing': 'nlp',
    'sql database': 'sql', 'database management': 'databases',
    'web development': 'web', 'mobile development': 'mobile',
    'rest api': 'rest', 'rest apis': 'rest',
}

def apply_synonym_mapping(text: str) -> str:
    text_lower = text.lower().strip()
    return SYNONYM_MAP.get(text_lower, text_lower)


def is_duplicate(text: str, processed_list: List[Dict], similarity_threshold: float = 0.85) -> bool:
    text_lower = text.lower().strip()
    
    for item in processed_list:
        existing = item["text"].lower().strip()
        
        # Exact match
        if text_lower == existing:
            return True
        
        # Check if one is substring of other
        if text_lower in existing or existing in text_lower:
            return True
    
    return False


def is_valid_entity(text: str, label: str) -> bool:
    text = text.strip().lower()
    
    if not text or len(text) < 2:
        return False
    skip_words = {
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'january', 'feb', 'mar', 'apr', 'sept', 'oct', 'nov', 'dec',
        'which', 'where', 'who', 'what', 'when', 'why', 'how',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
    }
    
    if text in skip_words:
        return False
    
    if label == 'SKILLS':
        if text.count(' ') > 4:
            return False
        
        words = text.split()
        if len(words) > 4:
            return False
        
        sentence_starters = ['training the', 'developing', 'currently', 'also', 'working']
        for starter in sentence_starters:
            if text.startswith(starter):
                return False
    
    return True


def clean_bullet_artifacts(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    
    bullet_chars = r'[•●◦◾▪▫■□◻]'
    text = re.sub(f'{bullet_chars}.*$', '', text)
    text = text.strip()
    section_markers = [
        'WORK EXPERIENCE', 'EDUCATION', 'ADDITIONAL INFORMATION', 
        'Technical Skills', 'Non - Technical Skills', 'SKILLS',
        'REQUIREMENTS', 'Requirements:', 'Experience:', 'Location:',
        'POSITION:', 'COMPANY:', 'CONTACT'
    ]
    
    for marker in section_markers:
        if marker in text:
            idx = text.find(marker)
            if idx > 0 and len(text[:idx].strip()) > 2:
                text = text[:idx].strip()
                break
            elif idx == 0 and len(text) > len(marker):
                text = text[len(marker):].strip()
                break
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) > 80:
        words = text.split()
        if words:
            # Take just first word/2-word phrase if it looks like a skill
            if len(words[0]) > 2:
                return words[0]
            elif len(words) > 1 and len(words[1]) > 2:
                return f"{words[0]} {words[1]}"
        return ''
    fragment_indicators = [
        r'.*\b(role:|working\s+on|developed|according|triggered|given|input|user|which|development|associate|november|august|june|march|april).*',
        r'.*\b(for\s+the|in\s+the|to\s+the|from\s+the)\b.*',  # Prepositions indicating longer text
        r'.*\d{1,2}[:\.\s]+\d{2,4}.*',  # Date patterns
    ]
    
    for pattern in fragment_indicators:
        if re.match(pattern, text, re.IGNORECASE):
            # Possible contamination - try to extract first noun
            words = text.split()
            if len(words) > 2:
                # Take first 1-2 words only if short enough
                candidate = words[0]
                if len(candidate) > 2 and not re.search(r'\d', candidate):
                    return candidate
            return ''
    if len(text) < 2:
        return ''
    
    return text


def postprocess_entities(entities: List[Dict]) -> List[Dict]:
    processed = []
    
    for ent in entities:
        text = ent.get("text", "").strip()
        if not text:
            continue
        
        label = ent.get("label", "")
        confidence = ent.get("confidence", 0.5)
        
        text = clean_bullet_artifacts(text)
        if not text or len(text) < 2:
            continue
        if not is_valid_entity(text, label):
            continue
        
        text_normalized = apply_synonym_mapping(text)
        if len(text_normalized) < 2:
            continue
        if is_duplicate(text_normalized, processed):
            continue
        
        processed.append({
            "label": label,
            "text": text_normalized,
            "confidence": confidence,
        })
    
    return processed


def merge_entities(spacy_entities: List[Dict], bert_entities: List[Dict]) -> List[Dict]:
    merged = []
    merged.extend(bert_entities)
    for spacy_ent in spacy_entities:
        spacy_text = spacy_ent.get("text", "").lower().strip()
        spacy_label = spacy_ent.get("label", "")
        found = False
        for merged_ent in merged:
            merged_text = merged_ent.get("text", "").lower().strip()
            if spacy_text in merged_text or merged_text in spacy_text:
                found = True
                break
        
        if not found:
            merged.append(spacy_ent)
    
    return postprocess_entities(merged)


class ATSInferencePipeline:
    
    def __init__(self, 
                 bert_model_dir: str = None,
                 spacy_model_name: str = None):
        self.parser = CVParser()
        
        self.bert_extractor = EntityExtractor(model_dir=bert_model_dir)
        self.spacy_extractor = None
        if spacy_model_name:
            try:
                import spacy
                self.spacy_extractor = spacy.load(spacy_model_name)
            except ImportError:
                print(f"⚠ Spacy not installed. Install with: pip install spacy")
            except OSError:
                print(f"⚠ Spacy model '{spacy_model_name}' not found.")
                print(f"  Install with: python -m spacy download {spacy_model_name}")
            except Exception as e:
                print(f"⚠ Error loading Spacy model: {e}")
    
    def extract_with_spacy(self, text: str) -> List[Dict]:
        if self.spacy_extractor is None:
            return []
        
        doc = self.spacy_extractor(text)
        entities = []
        label_map = {
            "PERSON": "NAME",
            "ORG": "COMPANIES_WORKED_AT",
            "GPE": "LOCATION",
            "DATE": "GRADUATION_YEAR",
        }
        
        for ent in doc.ents:
            mapped_label = label_map.get(ent.label_, ent.label_)
            entities.append({
                "text": ent.text,
                "label": mapped_label,
                "confidence": 0.7,
            })
        
        return entities
    
    def extract_with_bert(self, text: str, use_rule_fallback: bool = True) -> List[Dict]:
        result = self.bert_extractor.extract_entities(text, conf=0.5)
        bert_entities = []
        for label, items in result.items():
            if isinstance(items, list):
                for item in items:
                    bert_entities.append({
                        "text": item if isinstance(item, str) else str(item),
                        "label": label,
                        "confidence": 0.6,
                    })
        
        if use_rule_fallback:
            rule_skills = extract_skills_rule_based(text)
            existing_texts = {e.get("text", "").lower().strip() for e in bert_entities if e.get("label") == "SKILLS"}
            for rule_skill in rule_skills:
                rule_text = rule_skill["text"].lower().strip()
                if rule_text not in existing_texts:
                    bert_entities.append(rule_skill)
        
        return bert_entities
    
    def run_inference(self, cv_text: str, jd_text: str) -> Dict[str, Any]:
        cv_clean = preprocess_text(cv_text)
        jd_clean = preprocess_text(jd_text)
        
        cv_spacy = []
        jd_spacy = []
        if self.spacy_extractor is not None:
            cv_spacy = self.extract_with_spacy(cv_clean)
            jd_spacy = self.extract_with_spacy(jd_clean)
        
        cv_bert = self.extract_with_bert(cv_clean)
        jd_bert = self.extract_with_bert(jd_clean)
        
        if cv_spacy or jd_spacy:
            cv_entities = merge_entities(cv_spacy, cv_bert)
            jd_entities = merge_entities(jd_spacy, jd_bert)
        else:
            cv_entities = postprocess_entities(cv_bert)
            jd_entities = postprocess_entities(jd_bert)
        
        ats_score, explanation = calculate_ats_score(cv_entities, jd_entities)
        
        missing = get_missing_keywords(cv_entities, jd_entities)
        missing_skills = missing.get("SKILLS", [])
        
        return {
            "cv_entities": cv_entities,
            "jd_entities": jd_entities,
            "ats_score": ats_score,
            "score_explanation": explanation,
            "missing_skills": missing_skills,
            "missing_keywords": missing,
        }
