# Architecture

## System overview
```mermaid
flowchart TB
  B[Browser: HTML/CSS/JS] --> F[Flask server: routes, accounts, CSRF]
  F --> N[NLP layer]
  F --> L[Logic layer]
  N --> N1[preprocess / sections]
  N --> N2[skill extraction]
  N --> N3[JD analyser]
  N --> N4[keyphrases]
  N --> M[(spaCy, MiniLM embeddings, cross-encoder)]
  L --> L1[parser]
  L --> L2[role predictor]
  L --> L3[matcher]
  L --> L4[job APIs]
  L --> L5[fake job detector]
  L --> L6[market]
  L --> L7[reviewer + Gemini]
  L --> L8[advisor]
  L4 --> X[(JSearch / Adzuna / sample data)]
  F --> D[(SQLite)]
```

## Resume pipeline
```mermaid
flowchart LR
  A[PDF/DOCX/TXT] --> B[Text extraction + ATS flags]
  B --> C[Cleaning + sentences]
  C --> D[Section detection<br/>headings, then content classifier]
  D --> E[Skills layer 1: dictionary]
  E --> F[Layer 2: semantic phrase mapping]
  F --> G[Layer 3: possible new skills]
  G --> H[Context: learning / negation]
  H --> I[Skill levels from evidence]
  I --> J[Profile]
```

## Matching pipeline
```mermaid
flowchart LR
  P[Profile] --> S1
  J[Job postings] --> A[JD analysis: must/nice, alternatives, years, degree]
  A --> S1[Stage 1: bi-encoder similarity for all jobs]
  S1 --> K[+ skill coverage + TF-IDF keywords + level fit]
  K --> S2[Stage 2: cross-encoder re-ranks top 20]
  S2 --> R[Ranked jobs + verdicts]
  R --> E[Explanation: requirement to best resume line]
  R --> Q[Quick wins: near-miss skills]
  R --> FJ[Fake job check]
```

## Data flow between pages
Profile (targets) -> Jobs (search + ranking, stored per user) -> Match / Market / Prepare.
Ticking a skill in the plan adds it to the profile, which refreshes matches and readiness.

## Database tables
users, resumes, profiles, job_cache, results, saved_jobs, progress, fundamentals, readiness_history, settings.
