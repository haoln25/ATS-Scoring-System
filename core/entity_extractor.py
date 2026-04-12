import re
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
LINK_RE    = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/[\w\-]+|github\.com/[\w\-/]+)", re.I)
YEAR_RE    = re.compile(r"\b(19|20)\d{2}\b")
NOISE_RE   = re.compile(r"^[\s\-|,.:;()\[\]{}'\"]+|[\s\-|,.:;()\[\]{}'\"]+$")
SK_HDR_RE  = re.compile(r"(SKILLS|Technical\s+Skills|Key\s+Skills|Core\s+Competencies|Technologies|Tech\s+Stack)", re.I)
NEXT_RE    = re.compile(r"\n\s*(EXPERIENCE|WORK|EDUCATION|PROJECTS|CERTIFICATIONS|AWARDS|SUMMARY|OBJECTIVE|PROFILE)", re.I)
EDU_HDR_RE = re.compile(r"(EDUCATION|Academic\s+Background|Qualifications|Academic\s+Details)", re.I)
EDU_CTX_RE = re.compile(r"(B\.Tech|M\.Tech|B\.E\.|B\.Sc|M\.Sc|MCA|MBA|M\.S\.|Ph\.D|Bachelor|Master|Graduated|IIT|NIT|BITS|University|College|Institute)", re.I)
YOE_RE2    = re.compile(r"\b(\d{1,2}\+?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:experience|exp|work|in\s+\w+))?)", re.I)


def extract_skills_text(text):
    m = SK_HDR_RE.search(text)
    if not m: 
        return None
    after = text[m.end():]
    nl = after.find("\n")
    sc = after[nl:] if nl != -1 else after
    mn = NEXT_RE.search(sc)
    if mn: 
        sc = sc[:mn.start()]
    sc = NOISE_RE.sub("", sc.strip())
    sc = re.sub(r"\n+", ", ", sc).strip(" ,")
    
    if len(sc) < 3:
        return None
    
    skills = []
    for skill in sc.split(","):
        skill_clean = skill.strip()
        skill_clean = re.sub(r"\s*\([^)]*\)\s*", "", skill_clean).strip()
        if len(skill_clean) >= 2:
            skills.append(skill_clean)
    
    return skills if skills else None


def extract_grad_years(text):
    m_edu = EDU_HDR_RE.search(text)
    edu   = text[m_edu.start():] if m_edu else text
    mn    = NEXT_RE.search(edu[len(m_edu.group()):] if m_edu else edu)
    if mn: 
        edu = edu[:len(m_edu.group()) + mn.start() if m_edu else mn.start()]
    years = []
    for m in YEAR_RE.finditer(edu):
        yr = int(m.group())
        if 1990 <= yr <= 2030 and m.group() not in years:
            window = text[max(0, m.start()-150):min(len(text), m.end()+150)]
            if m_edu or EDU_CTX_RE.search(window): 
                years.append(m.group())
    return years[:2]


def extract_yoe(text):
    for m in YOE_RE2.finditer(text[:500]): 
        return [m.group().strip()]
    for m in YOE_RE2.finditer(text): 
        return [m.group().strip()]
    return []


_NOISE_TOK = {":", ";", "-", ".", "|", "/", ",", "(", ")", "+", "*", "#", "@", "&", "—", "–"}


class EntityExtractor:
    
    def __init__(self, model_dir: str = None, device: str = None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models" / "bert"
        
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading tokenizer from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        
        print(f"Loading BERT model from {self.model_dir}...")
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else {}
        print(f"Model loaded successfully. Labels: {self.id2label}")
    
    def predict_entities(self, text: str, batch_size: int = 128, tta_runs: int = 0, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        words = re.findall(r"\S+", text)
        if not words: 
            return []
        
        n_labels = len(self.id2label) if self.id2label else 9
        acc = np.zeros((len(words), n_labels), dtype=np.float64)
        
        for run in range(tta_runs + 1):
            if run > 0:
                self.model.train()
            else:
                self.model.eval()
            
            for i in range(0, len(words), batch_size):
                chunk = words[i:i+batch_size]
                enc = self.tokenizer(
                    chunk, 
                    is_split_into_words=True, 
                    return_tensors="pt",
                    truncation=True, 
                    max_length=512, 
                    padding=False
                )
                
                with torch.no_grad():
                    logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
                
                probs = torch.softmax(logits, -1).cpu().numpy()
                seen = set()
                
                for ti, wid in enumerate(enc.word_ids(0)):
                    if wid is None or wid in seen: 
                        continue
                    seen.add(wid)
                    gw = i + wid
                    if gw < len(words): 
                        acc[gw] += probs[ti]
        
        acc /= (tta_runs + 1)
        self.model.eval()
        
        return [
            {
                "word": words[i], 
                "label": self.id2label.get(int(np.argmax(acc[i])), f"LABEL_{int(np.argmax(acc[i]))}"),
                "score": float(np.max(acc[i]))
            } 
            for i in range(len(words))
        ]
    
    def group_and_clean(self, preds: List[Dict], text: str, conf: float = 0.5) -> Dict[str, List[str]]:
        ents, cur_type, cur_words = {}, None, []
        
        def flush():
            nonlocal cur_type, cur_words
            if cur_type and cur_words:
                v = NOISE_RE.sub("", " ".join(cur_words)).strip()
                if len(v) >= 2 and v not in ents.get(cur_type, []):
                    ents.setdefault(cur_type, []).append(v)
        
        for wp in preds:
            lbl, sc, w = wp["label"], wp["score"], wp["word"]
            if w in _NOISE_TOK: 
                flush()
                cur_type, cur_words = None, []
                continue
            
            if lbl.startswith("B-") and sc >= conf:
                flush()
                cur_type, cur_words = lbl[2:], [w]
            elif lbl.startswith("I-") and cur_type == lbl[2:] and sc >= conf * 0.75:
                cur_words.append(w)
            else:
                flush()
                cur_type, cur_words = None, []
        
        flush()
        
        # Extract additional information using regex patterns
        sk = extract_skills_text(text)
        if sk: 
            ents["Skills"] = sk  # Now sk is already a list
        
        gy = extract_grad_years(text)
        if gy: 
            ents["Graduation Year"] = gy
        elif "Graduation Year" in ents:
            valid = [YEAR_RE.search(y).group() for y in ents["Graduation Year"]
                     if YEAR_RE.search(y) and 1990 <= int(YEAR_RE.search(y).group()) <= 2030]
            if valid: 
                ents["Graduation Year"] = list(dict.fromkeys(valid))[:2]
            else: 
                ents.pop("Graduation Year", None)
        
        yoe = extract_yoe(text)
        if yoe: 
            ents["Years of Experience"] = yoe
        
        emails = EMAIL_RE.findall(text)
        if emails: 
            ents["Email Address"] = list(dict.fromkeys(emails))
        
        links = LINK_RE.findall(text)
        if links: 
            ents["Links"] = list(dict.fromkeys(links))
        
        if "Location" in ents: 
            ents["Location"] = list(dict.fromkeys(ents["Location"]))[:3]
        
        # Clean up empty entries
        for g in list(ents.keys()):
            vals = [NOISE_RE.sub("", v).strip() for v in ents[g]]
            vals = [v for v in vals if len(v) >= 2]
            if vals: 
                ents[g] = vals
            else: 
                del ents[g]
        
        return ents
    
    def extract_entities(self, text: str, conf: float = 0.5) -> Dict[str, List[str]]:
        preds = self.predict_entities(text, conf_threshold=conf)
        return self.group_and_clean(preds, text, conf=conf)


# Global instance (lazy loaded)
_extractor = None

def get_extractor(model_dir: str = None) -> EntityExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor(model_dir=model_dir)
    return _extractor


def extract_entities(text: str, model_dir: str = None, conf: float = 0.5) -> Dict[str, List[str]]:
    extractor = get_extractor(model_dir=model_dir)
    return extractor.extract_entities(text, conf=conf)