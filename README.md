# LegiFy: AI-Powered Legal Document Summarization

## Overview

**LegiFy** is an AI-driven system designed to **summarize, simplify, and analyze legal documents** efficiently.
It combines:

* **Named Entity Recognition (NER)**
* **Extractive summarization**
* **Abstractive summarization**

to deliver **high-quality legal text understanding** for faster decision-making.

---

## Features

### 1️⃣ Named Entity Recognition (NER) with LegalBERT

* Fine-tuned **LegalBERT** model for **legal-specific NER**.
* Detects entities such as:

  * Legal sections
  * Acts & statutes
  * Case references
  * NGO-related policies
* Useful for **government legal documents** and **NGO policy analysis**.

---

### 2️⃣ Extractive Summarization with BERTSUM

* Uses **BERTSUM** to extract **most relevant sentences**.
* Preserves **original legal meaning**.
* Ideal for **long judgments, contracts, and rulings**.

---

### 3️⃣ Custom Dataset & Fine-Tuning

* Based on **Yashaswat/Indian-Legal-Text-ABS** dataset.
* Covers:

  * **Legal Sections:** Article 21, IPC Section 420
  * **Acts & Statutes:** Environmental Protection Act, 1986
  * **NGO Laws:** Foreign Contribution Regulation Act (FCRA)
  * **Case References:** *Vishaka v. State of Rajasthan (1997)*

---

### 4️⃣ Evaluation Metrics

* **ROUGE** → Text overlap measurement
* **BLEU** → Fluency & accuracy
* **BERTScore** → Semantic similarity

Ensures **reliable legal summarization quality**.

---

### 5️⃣ Real-Time Legal Querying

* Ask **context-aware questions** from uploaded documents.
* AI extracts **precise legal answers** instantly.

---

### 6️⃣ Multilingual Support (Planned)

* Support for **regional Indian languages**.
* Improves accessibility for **non-English legal content**.

---

## Project Structure

```
LEGIFY/
│
├── backend/      → Django API + AI models
├── frontend/     → React + Vite user interface
└── README.md
```

---

## Installation & Usage

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Abhishek8252/DocuVoice.git
cd DocuVoice
```

---

## Backend Setup (Django)

```bash
cd backend

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run backend server
python manage.py runserver
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

## Frontend Setup (React + Vite)

Open **a new terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## How It Works

1. User uploads a **legal document**.
2. Backend performs:

   * **Text extraction**
   * **NER detection**
   * **Summarization**
3. Frontend displays:

   * **Simplified summary**
   * **Highlighted legal entities**
   * **Interactive Q&A**

---

## Future Enhancements

* Advanced **spaCy-based legal NLP pipeline**
* **Contract & compliance summarization**
* **Interactive analytics dashboard**
* **Cloud deployment with scalable AI inference**

---

## Author

**Abhishek**
AI & Full-Stack Developer
Focused on **Legal AI, NLP, and intelligent document processing**.
