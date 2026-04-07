# ATS Scoring System - Rebuild

## Cấu trúc dự án đề xuất (sau rebuild)

```
ATS-Scoring-System/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py       # Unicode, whitespace
│   │   ├── section_detector.py   # Detect CV sections
│   │   └── pii_masker.py         # Mask PII for logging
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   └── txt_parser.py
│   ├── ner/
│   │   ├── __init__.py
│   │   ├── bert_ner.py           # BERT NER inference
│   │   ├── spacy_ner.py          # spaCy NER inference
│   │   └── entity_merger.py      # Merge NER + rules
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── regex_patterns.py     # All regex patterns
│   │   ├── rule_engine.py        # Rule-based extraction
│   │   └── postprocessor.py      # Entity normalization
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── ats_calculator.py     # Score calculation
│   │   ├── weights_config.py     # Weight configuration
│   │   └── explainer.py          # Score explanation
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── suggestion_engine.py  # LLM integration
│   │   ├── prompts.py            # Prompt templates
│   │   └── safety.py             # PII filtering
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config.py
├── training/
│   ├── train_bert.py             # BERT training script
│   ├── train_spacy.py            # spaCy training script
│   ├── evaluate.py               # Model evaluation
│   └── error_analysis.py         # Error analysis tools
├── data/
│   ├── raw/
│   │   └── data.json             # Original dataset
│   ├── processed/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── synthetic/
│   │   └── synthetic_resumes.json
│   └── changelog.md              # Dataset version history
├── models/
│   ├── bert_ner/                 # BERT model artifacts
│   ├── spacy_ner/                # spaCy model artifacts
│   └── model_registry.json       # Model version registry
├── configs/
│   ├── training_config.yaml
│   ├── inference_config.yaml
│   └── scoring_weights.yaml
├── docker/
│   ├── Dockerfile.train
│   ├── Dockerfile.inference
│   └── docker-compose.yml
├── tests/
│   ├── integration/
│   └── e2e/
├── app.py                        # Streamlit application
├── requirements.txt
├── requirements-train.txt
└── README.md
```

## Epic 1: Dataset & Labeling

**Epic Owner:** Member 1  
**Priority:** Critical  
**Estimated Effort:** 40 story points

### ATS-1: Dataset Audit và Analysis

| Field                   | Value                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-1                                                                                                      |
| **Title**               | Audit dataset hiện tại và phân tích distribution                                                           |
| **Assignee**            | Member 1                                                                                                   |
| **Priority**            | Critical                                                                                                   |
| **Story Points**        | 5                                                                                                          |
| **Description**         | Phân tích dataset gốc `data.json` (220 samples), thống kê label distribution, xác định vấn đề data quality |
| **Acceptance Criteria** | - Report thống kê đầy đủ<br>- Xác định labels cần loại bỏ/thêm<br>- Document data quality issues           |
| **Subtasks**            | - [ ] Viết script analyze_dataset.py<br>- [ ] Chạy phân tích<br>- [ ] Viết report                          |
| **Dependencies**        | None                                                                                                       |
| **Deliverables**        | `reports/dataset_audit.md`, `scripts/analyze_dataset.py`                                                   |

---

### ATS-2: Loại bỏ UNKNOWN Labels

| Field                   | Value                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-2                                                                                           |
| **Title**               | Loại bỏ UNKNOWN labels từ dataset                                                               |
| **Assignee**            | Member 1                                                                                        |
| **Priority**            | Critical                                                                                        |
| **Story Points**        | 3                                                                                               |
| **Description**         | Xóa tất cả annotations có label UNKNOWN khỏi dataset                                            |
| **Acceptance Criteria** | - Không còn UNKNOWN trong dataset<br>- Không ảnh hưởng các labels khác<br>- Log số mẫu affected |
| **Subtasks**            | - [ ] Viết script cleanup<br>- [ ] Validate kết quả<br>- [ ] Backup data gốc                    |
| **Dependencies**        | ATS-1                                                                                           |
| **Deliverables**        | `data/processed/cleaned_data.json`                                                              |

---

### ATS-4: Annotate LANGUAGES Labels (Manual + Regex + Tool để hỗ trợ label)

| Field                   | Value                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-4                                                                                                          |
| **Title**               | Thêm LANGUAGES annotations vào dataset                                                                         |
| **Assignee**            | Member 4 (Primary), Member 2 (QA)                                                                              |
| **Priority**            | High                                                                                                           |
| **Story Points**        | 8                                                                                                              |
| **Description**         | Sử dụng regex weak labeling + manual review để thêm LANGUAGES labels                                           |
| **Acceptance Criteria** | - Tối thiểu 100 samples có LANGUAGES<br>- Đúng theo guideline<br>- QA pass 95%+ accuracy                       |
| **Subtasks**            | - [ ] Viết regex patterns<br>- [ ] Chạy weak labeling<br>- [ ] Manual review 50 samples<br>- [ ] QA validation |
| **Dependencies**        | ATS-3                                                                                                          |
| **Deliverables**        | `data/processed/data_with_languages.json`                                                                      |

---

### ATS-6: Train/Dev/Test Split

| Field                   | Value                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------- |
| **Task ID**             | ATS-6                                                                              |
| **Title**               | Chia dataset thành train/dev/test với fixed seed                                   |
| **Assignee**            | Member 3                                                                           |
| **Priority**            | High                                                                               |
| **Story Points**        | 3                                                                                  |
| **Description**         | Split dataset theo tỉ lệ 80/10/10 với seed=42 cho reproducibility                  |
| **Acceptance Criteria** | - Tỉ lệ chính xác<br>- Seed cố định<br>- Log thông tin chi tiết                    |
| **Subtasks**            | - [ ] Implement split logic<br>- [ ] Validate distribution<br>- [ ] Save splits    |
| **Dependencies**        | ATS-5                                                                              |
| **Deliverables**        | `data/processed/train.json`, `data/processed/dev.json`, `data/processed/test.json` |

---

### ATS-7: Synthetic Data Generation

| Field                   | Value                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-7                                                                                                     |
| **Title**               | Generate 1000+ synthetic resume samples                                                                   |
| **Assignee**            | Member 1 (Primary), Member 3 (Support)                                                                    |
| **Priority**            | High                                                                                                      |
| **Story Points**        | 8                                                                                                         |
| **Description**         | Dựa trên notebook `finetuning-ner-bert-v2.ipynb`, generate synthetic data với diverse patterns            |
| **Acceptance Criteria** | - 1000+ unique samples<br>- Cover all labels<br>- Diverse patterns                                        |
| **Subtasks**            | - [ ] Review notebook code<br>- [ ] Expand data pools<br>- [ ] Generate samples<br>- [ ] Validate quality |
| **Dependencies**        | ATS-6                                                                                                     |
| **Deliverables**        | `data/synthetic/synthetic_resumes.json`                                                                   |

---

## Epic 2: Text Preprocessing

**Epic Owner:** Member 2  
**Priority:** High  
**Estimated Effort:** 25 story points

### ATS-9: Design Preprocessing Pipeline

| Field                   | Value                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| **Task ID**             | ATS-9                                                                                            |
| **Title**               | Thiết kế preprocessing pipeline architecture                                                     |
| **Assignee**            | Member 2                                                                                         |
| **Priority**            | High                                                                                             |
| **Story Points**        | 5                                                                                                |
| **Description**         | Design modular preprocessing pipeline: Unicode fix → Whitespace norm → Section detect → PII mask |
| **Acceptance Criteria** | - Architecture diagram<br>- Interface definitions<br>- Input/output specs                        |
| **Subtasks**            | - [ ] Research best practices<br>- [ ] Define interfaces<br>- [ ] Document specs                 |
| **Dependencies**        | None                                                                                             |
| **Deliverables**        | `docs/architecture/preprocessing_pipeline.md`                                                    |

---

### ATS-10: Implement Unicode Fixer

| Field                   | Value                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-10                                                                                         |
| **Title**               | Implement Unicode character fixer                                                              |
| **Assignee**            | Member 4                                                                                       |
| **Priority**            | High                                                                                           |
| **Story Points**        | 5                                                                                              |
| **Description**         | Dựa trên `app.py`, expand Unicode replacement mappings, handle edge cases                      |
| **Acceptance Criteria** | - 100+ character mappings<br>- Handle edge cases                                               |
| **Subtasks**            | - [ ] Extract mappings từ app.py<br>- [ ] Research thêm patterns<br>- [ ] Implement module<br> |
| **Dependencies**        | ATS-9                                                                                          |
| **Deliverables**        | `src/preprocessing/text_cleaner.py`                                                            |

---

### ATS-11: Implement Whitespace Normalizer

| Field                   | Value                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-11                                                                                             |
| **Title**               | Implement whitespace normalization                                                                 |
| **Assignee**            | Member 2                                                                                           |
| **Priority**            | High                                                                                               |
| **Story Points**        | 3                                                                                                  |
| **Description**         | Normalize multiple spaces, newlines, tabs; handle line breaks properly                             |
| **Acceptance Criteria** | - Multiple space → single space<br>- Multiple newlines → single newline<br>- Trim leading/trailing |
| **Subtasks**            | - [ ] Implement logic<br>- [ ] Handle edge cases                                                   |
| **Dependencies**        | ATS-9                                                                                              |
| **Deliverables**        | `src/preprocessing/text_cleaner.py` (extension)                                                    |

---

### ATS-12: Implement Section Detector

| Field                   | Value                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-12                                                                                                              |
| **Title**               | Implement CV section detection                                                                                      |
| **Assignee**            | Member 3                                                                                                            |
| **Priority**            | Medium                                                                                                              |
| **Story Points**        | 5                                                                                                                   |
| **Description**         | Detect EDUCATION, SKILLS, EXPERIENCE, SUMMARY sections với regex patterns                                           |
| **Acceptance Criteria** | - Detect 4+ section types<br>- Return positions<br>- Handle variations                                              |
| **Subtasks**            | - [ ] Research section patterns<br>- [ ] Implement detection<br>- [ ] Test trên real CVs<br>- [ ] Handle edge cases |
| **Dependencies**        | ATS-9                                                                                                               |
| **Deliverables**        | `src/preprocessing/section_detector.py`                                                                             |

---

### ATS-13: Implement PII Masker

| Field                   | Value                                                                |
| ----------------------- | -------------------------------------------------------------------- |
| **Task ID**             | ATS-13                                                               |
| **Title**               | Implement PII masking for logging                                    |
| **Assignee**            | Member 5                                                             |
| **Priority**            | Medium                                                               |
| **Story Points**        | 3                                                                    |
| **Description**         | Mask emails, phones, addresses trước khi log hoặc gửi đến LLM        |
| **Acceptance Criteria** | - Mask emails → [EMAIL]<br>- Mask phones → [PHONE]<br>- Configurable |
| **Subtasks**            | - [ ] Define PII patterns<br>- [ ] Implement masking                 |
| **Dependencies**        | ATS-9                                                                |
| **Deliverables**        | `src/preprocessing/pii_masker.py`                                    |

---

### ATS-14: Integrate Preprocessing Pipeline

| Field                   | Value                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-14                                                                                   |
| **Title**               | Tích hợp tất cả preprocessing components                                                 |
| **Assignee**            | Member 2                                                                                 |
| **Priority**            | High                                                                                     |
| **Story Points**        | 4                                                                                        |
| **Description**         | Tích hợp các modules thành pipeline hoàn chỉnh với proper error handling                 |
| **Acceptance Criteria** | - Single entry point<br>- Proper error handling<br>- Integration tests pass              |
| **Subtasks**            | - [ ] Create pipeline class<br>- [ ] Add error handling<br>- [ ] Write integration tests |
| **Dependencies**        | ATS-10, ATS-11, ATS-12, ATS-13                                                           |
| **Deliverables**        | `src/preprocessing/__init__.py`                                                          |

---

## Epic 3: Rule-based & Regex Engine

**Epic Owner:** Member 4  
**Priority:** High  
**Estimated Effort:** 30 story points

### ATS-15: Design Regex Patterns

| Field                   | Value                                                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-15                                                                                                                              |
| **Title**               | Define và document tất cả regex patterns                                                                                            |
| **Assignee**            | Member 4                                                                                                                            |
| **Priority**            | High                                                                                                                                |
| **Story Points**        | 5                                                                                                                                   |
| **Description**         | Define patterns cho: Email, Phone, URL/Links, Dates, Duration, Languages                                                            |
| **Acceptance Criteria** | - 10+ pattern categories<br>- Test cases cho mỗi pattern<br>- Documentation                                                         |
| **Subtasks**            | - [ ] Email patterns<br>- [ ] Phone patterns (multi-format)<br>- [ ] URL patterns<br>- [ ] Date patterns<br>- [ ] Duration patterns |
| **Dependencies**        | None                                                                                                                                |
| **Deliverables**        | `src/rules/regex_patterns.py`, `docs/regex_patterns.md`                                                                             |

---

### ATS-16: Implement Email Extractor

| Field                   | Value                                                                         |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Task ID**             | ATS-16                                                                        |
| **Title**               | Implement email extraction với regex                                          |
| **Assignee**            | Member 4                                                                      |
| **Priority**            | High                                                                          |
| **Story Points**        | 3                                                                             |
| **Description**         | Extract emails với high precision, handle edge cases                          |
| **Acceptance Criteria** | - Precision > 98%<br>- Handle variations                                      |
| **Subtasks**            | - [ ] Implement pattern<br>- [ ] Test trên dataset<br>- [ ] Handle edge cases |
| **Dependencies**        | ATS-15                                                                        |
| **Deliverables**        | `src/rules/regex_patterns.py`                                                 |

---

### ATS-17: Implement Phone Extractor

| Field                   | Value                                                                      |
| ----------------------- | -------------------------------------------------------------------------- |
| **Task ID**             | ATS-17                                                                     |
| **Title**               | Implement phone number extraction                                          |
| **Assignee**            | Member 4                                                                   |
| **Priority**            | Medium                                                                     |
| **Story Points**        | 3                                                                          |
| **Description**         | Extract phone numbers với multiple formats: +91-xxx, xxx-xxx-xxxx, etc.    |
| **Acceptance Criteria** | - Support 5+ formats<br>- Precision > 95%                                  |
| **Subtasks**            | - [ ] Define formats<br>- [ ] Implement patterns<br>- [ ] Test và validate |
| **Dependencies**        | ATS-15                                                                     |
| **Deliverables**        | `src/rules/regex_patterns.py`                                              |

---

### ATS-18: Implement URL/Link Extractor

| Field                   | Value                                                                      |
| ----------------------- | -------------------------------------------------------------------------- |
| **Task ID**             | ATS-18                                                                     |
| **Title**               | Implement URL và profile link extraction                                   |
| **Assignee**            | Member 4                                                                   |
| **Priority**            | Medium                                                                     |
| **Story Points**        | 3                                                                          |
| **Description**         | Extract LinkedIn, GitHub, portfolio links                                  |
| **Acceptance Criteria** | - LinkedIn pattern<br>- GitHub pattern<br>- Generic URL pattern            |
| **Subtasks**            | - [ ] LinkedIn pattern<br>- [ ] GitHub pattern<br>- [ ] Portfolio patterns |
| **Dependencies**        | ATS-15                                                                     |
| **Deliverables**        | `src/rules/regex_patterns.py`                                              |

---

### ATS-19: Implement Date/Duration Extractor

| Field                   | Value                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-19                                                                                                   |
| **Title**               | Implement date và duration extraction                                                                    |
| **Assignee**            | Member 4                                                                                                 |
| **Priority**            | Medium                                                                                                   |
| **Story Points**        | 5                                                                                                        |
| **Description**         | Extract years, date ranges, experience duration                                                          |
| **Acceptance Criteria** | - Year extraction<br>- Date range (2018-2020)<br>- Duration (5 years)                                    |
| **Subtasks**            | - [ ] Year patterns<br>- [ ] Month-Year patterns<br>- [ ] Duration patterns<br>- [ ] Experience patterns |
| **Dependencies**        | ATS-15                                                                                                   |
| **Deliverables**        | `src/rules/regex_patterns.py`                                                                            |

---

### ATS-20: Implement Rule Engine

| Field                   | Value                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-20                                                                                 |
| **Title**               | Implement rule-based entity extraction engine                                          |
| **Assignee**            | Member 4                                                                               |
| **Priority**            | High                                                                                   |
| **Story Points**        | 5                                                                                      |
| **Description**         | Orchestrate tất cả regex extractors, return structured entities                        |
| **Acceptance Criteria** | - Single interface<br>- Configurable rules<br>- Proper entity format                   |
| **Subtasks**            | - [ ] Design RuleEngine class<br>- [ ] Integrate extractors<br>- [ ] Add configuration |
| **Dependencies**        | ATS-16, ATS-17, ATS-18, ATS-19                                                         |
| **Deliverables**        | `src/rules/rule_engine.py`                                                             |

---

### ATS-21: Implement Entity Postprocessor

| Field                   | Value                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-21                                                                                                    |
| **Title**               | Implement entity normalization và postprocessing                                                          |
| **Assignee**            | Member 3                                                                                                  |
| **Priority**            | High                                                                                                      |
| **Story Points**        | 5                                                                                                         |
| **Description**         | Normalize entities: lowercase, synonym mapping, deduplication                                             |
| **Acceptance Criteria** | - Synonym mapping (python3 → python)<br>- Deduplication<br>- Configurable                                 |
| **Subtasks**            | - [ ] Define synonym map<br>- [ ] Implement normalization<br>- [ ] Implement deduplication<br>- [ ] Tests |
| **Dependencies**        | ATS-20                                                                                                    |
| **Deliverables**        | `src/rules/postprocessor.py`                                                                              |

---

## Epic 4: NER Model Fine-tuning

**Epic Owner:** Member 3  
**Priority:** Critical  
**Estimated Effort:** 45 story points

### ATS-23: Prepare Training Data for BERT

| Field                   | Value                                                                             |
| ----------------------- | --------------------------------------------------------------------------------- |
| **Task ID**             | ATS-23                                                                            |
| **Title**               | Convert dataset sang BIO format cho BERT training                                 |
| **Assignee**            | Member 1                                                                          |
| **Priority**            | Critical                                                                          |
| **Story Points**        | 5                                                                                 |
| **Description**         | Parse resume data sang BIO tagged format tương thích với transformers             |
| **Acceptance Criteria** | - BIO format correct<br>- Handle all 23 labels<br>- Align với tokenizer           |
| **Subtasks**            | - [ ] Implement parsing logic<br>- [ ] Handle edge cases<br>- [ ] Validate output |
| **Dependencies**        | ATS-8                                                                             |
| **Deliverables**        | `training/prepare_data.py`                                                        |

---

### ATS-24: Setup Training Environment

| Field                   | Value                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-24                                                                                    |
| **Title**               | Setup môi trường training với PyTorch + Transformers                                      |
| **Assignee**            | Member 3                                                                                  |
| **Priority**            | Critical                                                                                  |
| **Story Points**        | 3                                                                                         |
| **Description**         | Install dependencies, verify GPU access, setup experiment tracking                        |
| **Acceptance Criteria** | - All dependencies installed<br>- GPU accessible<br>- Can load base model                 |
| **Subtasks**            | - [ ] Install requirements-train.txt<br>- [ ] Verify GPU<br>- [ ] Test base model loading |
| **Dependencies**        | None                                                                                      |
| **Deliverables**        | `requirements-train.txt`, setup documentation                                             |

---

### ATS-25: Train Baseline BERT NER Model (có thể train trên Kaggle)

| Field                   | Value                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-25                                                                                            |
| **Title**               | Train baseline model và record metrics                                                            |
| **Assignee**            | Member 1                                                                                          |
| **Priority**            | Critical                                                                                          |
| **Story Points**        | 8                                                                                                 |
| **Description**         | Fine-tune dslim/bert-base-NER với default hyperparameters, record baseline metrics                |
| **Acceptance Criteria** | - Training completes<br>- Baseline F1 recorded<br>- Model saved                                   |
| **Subtasks**            | - [ ] Implement training script<br>- [ ] Run training<br>- [ ] Save model<br>- [ ] Record metrics |
| **Dependencies**        | ATS-23, ATS-24                                                                                    |
| **Deliverables**        | `models/bert_ner_v1.0/`, `reports/baseline_metrics.md`                                            |

---

### ATS-26: Implement Evaluation Script

| Field                   | Value                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-26                                                                                    |
| **Title**               | Implement model evaluation với seqeval                                                    |
| **Assignee**            | Member 2                                                                                  |
| **Priority**            | High                                                                                      |
| **Story Points**        | 5                                                                                         |
| **Description**         | Evaluate model với precision, recall, F1 per label và overall                             |
| **Acceptance Criteria** | - Per-label metrics<br>- Overall metrics<br>- Classification report                       |
| **Subtasks**            | - [ ] Implement eval script<br>- [ ] Calculate per-label metrics<br>- [ ] Generate report |
| **Dependencies**        | ATS-25                                                                                    |
| **Deliverables**        | `training/evaluate.py`                                                                    |

---

### ATS-27: Error Analysis by Label

| Field                   | Value                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-27                                                                                                        |
| **Title**               | Phân tích errors theo từng label                                                                              |
| **Assignee**            | Member 2                                                                                                      |
| **Priority**            | High                                                                                                          |
| **Story Points**        | 5                                                                                                             |
| **Description**         | Identify false positives, false negatives, wrong labels cho mỗi entity type                                   |
| **Acceptance Criteria** | - Error breakdown by label<br>- Example errors<br>- Recommendations                                           |
| **Subtasks**            | - [ ] Collect predictions<br>- [ ] Categorize errors<br>- [ ] Analyze patterns<br>- [ ] Write recommendations |
| **Dependencies**        | ATS-26                                                                                                        |
| **Deliverables**        | `reports/error_analysis.md`, `training/error_analysis.py`                                                     |

---

### ATS-28: Hyperparameter Experiment Plan

| Field                   | Value                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-28                                                                                 |
| **Title**               | Lập kế hoạch experiments cho hyperparameter tuning                                     |
| **Assignee**            | Member 3                                                                               |
| **Priority**            | High                                                                                   |
| **Story Points**        | 3                                                                                      |
| **Description**         | Define experiments: learning rate, batch size, epochs, warmup ratio                    |
| **Acceptance Criteria** | - Experiment matrix<br>- Success criteria<br>- Resource estimation                     |
| **Subtasks**            | - [ ] Define param ranges<br>- [ ] Create experiment matrix<br>- [ ] Document criteria |
| **Dependencies**        | ATS-27                                                                                 |
| **Deliverables**        | `docs/experiment_plan.md`                                                              |

---

### ATS-29: Run Hyperparameter Experiments

| Field                   | Value                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| **Task ID**             | ATS-29                                                                                                 |
| **Title**               | Chạy hyperparameter experiments                                                                        |
| **Assignee**            | Member 3                                                                                               |
| **Priority**            | High                                                                                                   |
| **Story Points**        | 8                                                                                                      |
| **Description**         | Run experiments theo plan, track results                                                               |
| **Acceptance Criteria** | - All experiments completed<br>- Results tracked<br>- Best config identified                           |
| **Subtasks**            | - [ ] Setup experiment tracking<br>- [ ] Run experiments<br>- [ ] Analyze results<br>- [ ] Select best |
| **Dependencies**        | ATS-28                                                                                                 |
| **Deliverables**        | `reports/experiment_results.md`                                                                        |

---

### ATS-30: Train Final Model

| Field                   | Value                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **Task ID**             | ATS-30                                                                                     |
| **Title**               | Train final model với best hyperparameters                                                 |
| **Assignee**            | Member 3                                                                                   |
| **Priority**            | Critical                                                                                   |
| **Story Points**        | 5                                                                                          |
| **Description**         | Train model với best config, evaluate on test set                                          |
| **Acceptance Criteria** | - F1 ≥ 0.85<br>- Per-label targets met<br>- Model saved                                    |
| **Subtasks**            | - [ ] Apply best config<br>- [ ] Train model<br>- [ ] Evaluate on test<br>- [ ] Save model |
| **Dependencies**        | ATS-29                                                                                     |
| **Deliverables**        | `models/bert_ner_v2.0/`                                                                    |

---

### ATS-31: Model Versioning Setup (Optional có thể làm hoặc không)

| Field                   | Value                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-31                                                                                         |
| **Title**               | Setup model versioning và registry                                                             |
| **Assignee**            | Member 5                                                                                       |
| **Priority**            | Medium                                                                                         |
| **Story Points**        | 3                                                                                              |
| **Description**         | Create model registry, versioning scheme, metadata tracking                                    |
| **Acceptance Criteria** | - Registry created<br>- Version scheme documented<br>- Metadata tracked                        |
| **Subtasks**            | - [ ] Define versioning scheme<br>- [ ] Create registry file<br>- [ ] Script to version models |
| **Dependencies**        | ATS-30                                                                                         |
| **Deliverables**        | `models/model_registry.json`, `scripts/version_model.py`                                       |

---

## Epic 5: Inference Pipeline

**Epic Owner:** Member 4  
**Priority:** Critical  
**Estimated Effort:** 30 story points

### ATS-32: Design Inference Pipeline Architecture

| Field                   | Value                                                                          |
| ----------------------- | ------------------------------------------------------------------------------ |
| **Task ID**             | ATS-32                                                                         |
| **Title**               | Design end-to-end inference pipeline                                           |
| **Assignee**            | Member 4                                                                       |
| **Priority**            | Critical                                                                       |
| **Story Points**        | 5                                                                              |
| **Description**         | Design pipeline: Preprocess → NER → Rules → Merge → Postprocess                |
| **Acceptance Criteria** | - Architecture diagram<br>- Interface definitions<br>- Error handling strategy |
| **Subtasks**            | - [ ] Draw architecture<br>- [ ] Define interfaces<br>- [ ] Document flow      |
| **Dependencies**        | ATS-14, ATS-20, ATS-30                                                         |
| **Deliverables**        | `docs/architecture/inference_pipeline.md`                                      |

---

### ATS-33: Implement spaCy NER Inference

| Field                   | Value                                                                    |
| ----------------------- | ------------------------------------------------------------------------ |
| **Task ID**             | ATS-33                                                                   |
| **Title**               | Implement spaCy NER inference wrapper                                    |
| **Assignee**            | Member 4                                                                 |
| **Priority**            | High                                                                     |
| **Story Points**        | 5                                                                        |
| **Description**         | Wrap spaCy model for inference, return standardized entity format        |
| **Acceptance Criteria** | - Load model<br>- Extract entities<br>- Standardized output              |
| **Subtasks**            | - [ ] Implement wrapper<br>- [ ] Handle errors<br>- [ ] Test performance |
| **Dependencies**        | ATS-32                                                                   |
| **Deliverables**        | `src/ner/spacy_ner.py`                                                   |

---

### ATS-34: Implement BERT NER Inference

| Field                   | Value                                                                   |
| ----------------------- | ----------------------------------------------------------------------- |
| **Task ID**             | ATS-34                                                                  |
| **Title**               | Implement BERT NER inference với transformers                           |
| **Assignee**            | Member 3                                                                |
| **Priority**            | Medium                                                                  |
| **Story Points**        | 5                                                                       |
| **Description**         | Alternative inference với BERT model sử dụng HuggingFace pipeline       |
| **Acceptance Criteria** | - Load BERT model<br>- Run inference<br>- Standardized output           |
| **Subtasks**            | - [ ] Implement wrapper<br>- [ ] Benchmark speed<br>- [ ] Test accuracy |
| **Dependencies**        | ATS-32, ATS-30                                                          |
| **Deliverables**        | `src/ner/bert_ner.py`                                                   |

---

### ATS-35: Implement Entity Merger

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| **Task ID**             | ATS-35                                                                      |
| **Title**               | Implement NER + Rule entity merger                                          |
| **Assignee**            | Member 3                                                                    |
| **Priority**            | High                                                                        |
| **Story Points**        | 5                                                                           |
| **Description**         | Merge entities từ NER và rule-based với priority handling                   |
| **Acceptance Criteria** | - Priority rules clear<br>- No duplicates<br>- Handle overlaps              |
| **Subtasks**            | - [ ] Define merge rules<br>- [ ] Implement merger<br>- [ ] Test edge cases |
| **Dependencies**        | ATS-33, ATS-20                                                              |
| **Deliverables**        | `src/ner/entity_merger.py`                                                  |

---

### ATS-36: Integrate Full Inference Pipeline

| Field                   | Value                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-36                                                                                            |
| **Title**               | Tích hợp full inference pipeline                                                                  |
| **Assignee**            | Member 4                                                                                          |
| **Priority**            | Critical                                                                                          |
| **Story Points**        | 5                                                                                                 |
| **Description**         | Integrate tất cả components thành pipeline hoàn chỉnh                                             |
| **Acceptance Criteria** | - Single entry point<br>- All components connected<br>- Integration tests pass                    |
| **Subtasks**            | - [ ] Create InferencePipeline class<br>- [ ] Connect components<br>- [ ] Write integration tests |
| **Dependencies**        | ATS-14, ATS-33, ATS-35, ATS-21                                                                    |
| **Deliverables**        | `src/ner/inference_pipeline.py`                                                                   |

---

## Epic 6: ATS Scoring

**Epic Owner:** Member 4  
**Priority:** High  
**Estimated Effort:** 25 story points

### ATS-38: Define Scoring Formula và Weights

| Field                   | Value                                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-38                                                                                                               |
| **Title**               | Define scoring formula và weight configuration                                                                       |
| **Assignee**            | Member 4                                                                                                             |
| **Priority**            | High                                                                                                                 |
| **Story Points**        | 5                                                                                                                    |
| **Description**         | Define weighted scoring: Skills (30%), Experience (25%), Education (20%), Keywords (15%), Languages (5%), Other (5%) |
| **Acceptance Criteria** | - Formula documented<br>- Weights configurable<br>- YAML config                                                      |
| **Subtasks**            | - [ ] Research scoring methods<br>- [ ] Define formula<br>- [ ] Create config file                                   |
| **Dependencies**        | None                                                                                                                 |
| **Deliverables**        | `configs/scoring_weights.yaml`, `docs/scoring_algorithm.md`                                                          |

---

### ATS-39: Implement Skills Score Calculator

| Field                   | Value                                                          |
| ----------------------- | -------------------------------------------------------------- | ------- | --- | --- | --- |
| **Task ID**             | ATS-39                                                         |
| **Title**               | Implement skills matching score                                |
| **Assignee**            | Member 4                                                       |
| **Priority**            | High                                                           |
| **Story Points**        | 5                                                              |
| **Description**         | Calculate skills overlap:                                      | CV ∩ JD | /   | JD  |     |
| **Acceptance Criteria** | - Correct calculation<br>- Handle synonyms                     |
| **Subtasks**            | - [ ] Implement calculation<br>- [ ] Integrate synonym mapping |
| **Dependencies**        | ATS-38                                                         |
| **Deliverables**        | `src/scoring/ats_calculator.py`                                |

---

### ATS-40: Implement Experience Score Calculator

| Field                   | Value                                                                   |
| ----------------------- | ----------------------------------------------------------------------- |
| **Task ID**             | ATS-40                                                                  |
| **Title**               | Implement experience matching score                                     |
| **Assignee**            | Member 4                                                                |
| **Priority**            | High                                                                    |
| **Story Points**        | 5                                                                       |
| **Description**         | Calculate based on years of experience, company relevance               |
| **Acceptance Criteria** | - Years comparison<br>- Handle missing data<br>- Tests                  |
| **Subtasks**            | - [ ] Extract years<br>- [ ] Calculate score<br>- [ ] Handle edge cases |
| **Dependencies**        | ATS-38                                                                  |
| **Deliverables**        | `src/scoring/ats_calculator.py`                                         |

---

### ATS-41: Implement Education Score Calculator

| Field                   | Value                                                                         |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Task ID**             | ATS-41                                                                        |
| **Title**               | Implement education matching score                                            |
| **Assignee**            | Member 3                                                                      |
| **Priority**            | Medium                                                                        |
| **Story Points**        | 3                                                                             |
| **Description**         | Calculate based on degree level matching                                      |
| **Acceptance Criteria** | - Degree level mapping<br>- Comparison logic<br>- Tests                       |
| **Subtasks**            | - [ ] Define degree levels<br>- [ ] Implement comparison<br>- [ ] Write tests |
| **Dependencies**        | ATS-38                                                                        |
| **Deliverables**        | `src/scoring/ats_calculator.py`                                               |

---

### ATS-42: Implement Score Explanation Generator

| Field                   | Value                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------- |
| **Task ID**             | ATS-42                                                                             |
| **Title**               | Generate detailed score explanation                                                |
| **Assignee**            | Member 3                                                                           |
| **Priority**            | High                                                                               |
| **Story Points**        | 5                                                                                  |
| **Description**         | Generate matched/missing skills, strengths, areas to improve                       |
| **Acceptance Criteria** | - Matched skills list<br>- Missing skills list<br>- Strengths/weaknesses           |
| **Subtasks**            | - [ ] Implement explanation logic<br>- [ ] Format output<br>- [ ] Test với samples |
| **Dependencies**        | ATS-39, ATS-40, ATS-41                                                             |
| **Deliverables**        | `src/scoring/explainer.py`                                                         |

---

### ATS-44: Design LLM Integration Architecture

| Field                   | Value                                                                            |
| ----------------------- | -------------------------------------------------------------------------------- |
| **Task ID**             | ATS-44                                                                           |
| **Title**               | Design LLM suggestion integration                                                |
| **Assignee**            | Member 5                                                                         |
| **Priority**            | High                                                                             |
| **Story Points**        | 3                                                                                |
| **Description**         | Design: API selection (OpenAI/Anthropic/Gemini), prompt structure, output schema |
| **Acceptance Criteria** | - Architecture document<br>- API comparison<br>- Schema defined                  |
| **Subtasks**            | - [ ] Compare APIs<br>- [ ] Define schema<br>- [ ] Document                      |
| **Dependencies**        | None                                                                             |
| **Deliverables**        | `docs/architecture/llm_integration.md`                                           |

---

### ATS-45: Implement OpenAI/Claude API Integration

| Field                   | Value                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-45                                                                                       |
| **Title**               | Implement LLM API integration                                                                |
| **Assignee**            | Member 5                                                                                     |
| **Priority**            | High                                                                                         |
| **Story Points**        | 5                                                                                            |
| **Description**         | Implement clients cho OpenAI và Anthropic với proper error handling                          |
| **Acceptance Criteria** | - Both APIs supported<br>- Error handling<br>- Rate limiting                                 |
| **Subtasks**            | - [ ] OpenAI client<br>- [ ] Anthropic client<br>- [ ] Error handling<br>- [ ] Rate limiting |
| **Dependencies**        | ATS-44                                                                                       |
| **Deliverables**        | `src/llm/suggestion_engine.py`                                                               |

---

### ATS-46: Design Prompt Templates

| Field                   | Value                                                                   |
| ----------------------- | ----------------------------------------------------------------------- |
| **Task ID**             | ATS-46                                                                  |
| **Title**               | Design và test prompt templates                                         |
| **Assignee**            | Member 3                                                                |
| **Priority**            | High                                                                    |
| **Story Points**        | 5                                                                       |
| **Description**         | Create effective prompts cho CV improvement suggestions                 |
| **Acceptance Criteria** | - Clear prompt structure<br>- Tested với samples<br>- Consistent output |
| **Subtasks**            | - [ ] Draft prompts<br>- [ ] Test iterations<br>- [ ] Finalize template |
| **Dependencies**        | ATS-44                                                                  |
| **Deliverables**        | `src/llm/prompts.py`                                                    |

---

### ATS-47: Define Output Schema

| Field                   | Value                                                           |
| ----------------------- | --------------------------------------------------------------- |
| **Task ID**             | ATS-47                                                          |
| **Title**               | Define LLM output schema và validation                          |
| **Assignee**            | Member 2                                                        |
| **Priority**            | Medium                                                          |
| **Story Points**        | 3                                                               |
| **Description**         | Define structured output format, validation rules               |
| **Acceptance Criteria** | - JSON schema<br>- Validation logic<br>- Tests                  |
| **Subtasks**            | - [ ] Define schema<br>- [ ] Implement validation<br>- [ ] Test |
| **Dependencies**        | ATS-46                                                          |
| **Deliverables**        | `src/llm/schemas.py`                                            |

---

### ATS-48: Implement Safety Module

| Field                   | Value                                                                  |
| ----------------------- | ---------------------------------------------------------------------- |
| **Task ID**             | ATS-48                                                                 |
| **Title**               | Implement PII filtering và output safety                               |
| **Assignee**            | Member 5                                                               |
| **Priority**            | Critical                                                               |
| **Story Points**        | 5                                                                      |
| **Description**         | Filter PII before sending to LLM, validate output safety               |
| **Acceptance Criteria** | - No PII sent to LLM<br>- Output filtered<br>- Logging safe            |
| **Subtasks**            | - [ ] Input filtering<br>- [ ] Output validation<br>- [ ] Safe logging |
| **Dependencies**        | ATS-45                                                                 |
| **Deliverables**        | `src/llm/safety.py`                                                    |

---

### ATS-49: Implement Fallback Suggestions

| Field                   | Value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| **Task ID**             | ATS-49                                                       |
| **Title**               | Implement rule-based fallback khi LLM unavailable            |
| **Assignee**            | Member 4                                                     |
| **Priority**            | Medium                                                       |
| **Story Points**        | 3                                                            |
| **Description**         | Basic suggestions khi LLM API fails hoặc disabled            |
| **Acceptance Criteria** | - Basic suggestions<br>- Graceful degradation<br>- Tests     |
| **Subtasks**            | - [ ] Define fallback rules<br>- [ ] Implement<br>- [ ] Test |
| **Dependencies**        | ATS-45                                                       |
| **Deliverables**        | `src/llm/suggestion_engine.py`                               |

---

## Epic 8: Containerization & DevOps

**Epic Owner:** Member 5  
**Priority:** High  
**Estimated Effort:** 25 story points

### ATS-50: Create Training Dockerfile

| Field                   | Value                                                             |
| ----------------------- | ----------------------------------------------------------------- |
| **Task ID**             | ATS-50                                                            |
| **Title**               | Create Dockerfile cho training environment                        |
| **Assignee**            | Member 5                                                          |
| **Priority**            | High                                                              |
| **Story Points**        | 5                                                                 |
| **Description**         | Docker image với PyTorch, transformers, GPU support               |
| **Acceptance Criteria** | - Build successful<br>- GPU access<br>- Training runs             |
| **Subtasks**            | - [ ] Write Dockerfile<br>- [ ] Test build<br>- [ ] Test training |
| **Dependencies**        | ATS-24                                                            |
| **Deliverables**        | `docker/Dockerfile.train`                                         |

---

### ATS-51: Create Inference Dockerfile

| Field                   | Value                                                               |
| ----------------------- | ------------------------------------------------------------------- |
| **Task ID**             | ATS-51                                                              |
| **Title**               | Create Dockerfile cho inference                                     |
| **Assignee**            | Member 5                                                            |
| **Priority**            | High                                                                |
| **Story Points**        | 5                                                                   |
| **Description**         | Lightweight image với spaCy, Streamlit                              |
| **Acceptance Criteria** | - Build < 2GB<br>- Startup < 30s<br>- App runs                      |
| **Subtasks**            | - [ ] Write Dockerfile<br>- [ ] Optimize size<br>- [ ] Test startup |
| **Dependencies**        | ATS-36                                                              |
| **Deliverables**        | `docker/Dockerfile.inference`                                       |

---

### ATS-52: Create Docker Compose

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| **Task ID**             | ATS-52                                                                      |
| **Title**               | Create Docker Compose configuration                                         |
| **Assignee**            | Member 5                                                                    |
| **Priority**            | High                                                                        |
| **Story Points**        | 3                                                                           |
| **Description**         | Compose file với training và inference services                             |
| **Acceptance Criteria** | - Both services defined<br>- Volume mounts<br>- Env vars                    |
| **Subtasks**            | - [ ] Define services<br>- [ ] Configure volumes<br>- [ ] Environment setup |
| **Dependencies**        | ATS-50, ATS-51                                                              |
| **Deliverables**        | `docker/docker-compose.yml`                                                 |

---

### ATS-53: Setup Environment Configuration

| Field                   | Value                                                                 |
| ----------------------- | --------------------------------------------------------------------- |
| **Task ID**             | ATS-53                                                                |
| **Title**               | Setup environment variable configuration                              |
| **Assignee**            | Member 5                                                              |
| **Priority**            | Medium                                                                |
| **Story Points**        | 3                                                                     |
| **Description**         | .env template, configuration management                               |
| **Acceptance Criteria** | - .env.example created<br>- All vars documented<br>- Secrets handling |
| **Subtasks**            | - [ ] List all env vars<br>- [ ] Create template<br>- [ ] Document    |
| **Dependencies**        | ATS-52                                                                |
| **Deliverables**        | `.env.example`, `docs/configuration.md`                               |

---

### ATS-54: Test Containers Locally

| Field                   | Value                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| **Task ID**             | ATS-54                                                                                         |
| **Title**               | Test containers trên local environment                                                         |
| **Assignee**            | Member 1                                                                                       |
| **Priority**            | High                                                                                           |
| **Story Points**        | 3                                                                                              |
| **Description**         | Build và test cả training và inference containers                                              |
| **Acceptance Criteria** | - Both containers build<br>- Training runs<br>- Inference serves                               |
| **Subtasks**            | - [ ] Build containers<br>- [ ] Test training<br>- [ ] Test inference<br>- [ ] Document issues |
| **Dependencies**        | ATS-52                                                                                         |
| **Deliverables**        | `reports/container_testing.md`                                                                 |

---
