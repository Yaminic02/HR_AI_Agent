# 🏢 HR AI Agent  
### Production-Grade HR Intelligence System with Deterministic Logic + Local & API LLMs

---

## 🚀 Project Overview
**HR AI Agent** is a production-oriented AI system designed to handle real HR workflows safely and reliably.

Unlike generic AI chatbots, this system **strictly separates deterministic HR decisions from probabilistic LLM reasoning**.  
It supports both **Local LLMs (Ollama)** and **API-based LLMs (OpenAI)**, making it suitable for enterprise and privacy-sensitive environments.

---

## 💡 Why This Project Stands Out
✔ Rule-based, auditable HR decisions  
✔ LLMs used **only** for explanations and Q&A  
✔ Supports offline, zero-cost Local LLM execution  
✔ Designed for real HR risk & compliance  
✔ Mirrors **how production AI systems are actually built**

---

## 🧠 Design Philosophy
> **“LLMs should assist humans — not replace business rules.”**

All promotions, attrition scores, and HR outcomes are computed using **deterministic logic**.  
The LLM layer is sandboxed and **never allowed to override rules or data**.

---

## ✨ Key Capabilities
- Excel / CSV-based employee ingestion  
- Deterministic promotion & attrition logic  
- Dual LLM architecture (Local + API)  
- Rule-first question routing  
- FAISS-based contextual retrieval  
- One-line, hallucination-free answers  
- PDF employee report generation  
- Interactive Streamlit dashboard  

---

## 🏗️ System Architecture
User Query
↓
Rule-Based Router (Employee / Promotion / Attrition)
↓
If rule not matched → FAISS Context Retrieval
↓
LLM Explanation Layer (Local Ollama or OpenAI API)
↓
Short, Data-Grounded Answer


---

## 🤖 Dual LLM Execution Modes

### 🔹 Local LLM Mode (Ollama – Phi-3)
- Fully offline
- Zero API cost
- Ideal for sensitive HR data

### 🔹 API LLM Mode (OpenAI)
- Cloud-based reasoning
- Controlled temperature & token usage
- Used strictly for explanations

✅ Both modes enforce **one-line, data-only responses**

---

## 🔐 Safety & Guardrails
- No hallucinated HR decisions  
- No speculative answers  
- No policy overrides  
- Numeric data validation  
- Deterministic outputs for audits  
- Complete LLM isolation from business logic  

---

## 🧰 Technology Stack
- **Language:** Python  
- **UI:** Streamlit  
- **Data:** Pandas, NumPy  
- **Retrieval:** FAISS  
- **Local LLM:** Ollama (Phi-3)  
- **API LLM:** OpenAI  
- **Reports:** ReportLab  

---

## 📂 Project Structure
hr-ai-agent/
├── hr_ai_agent_local.py # Local LLM – Ollama
├── hr_ai_agent_api.py # API LLM – OpenAI
├── data/
├── requirements.txt
├── .env.example
└── README.md


---

## 🌱 Future Enhancements
- RAG over HR policy documents  
- Role-based access (HR vs Manager)  
- Audit logs for LLM outputs  
- WhatsApp / Slack integration  
- Evaluation metrics for answer quality  

---

## 👩‍💻 Author
**Yamini Chauhan**

This project demonstrates **production-ready AI thinking**, a strong **reliability mindset**, and hands-on experience building **LLM + rule-based hybrid systems**.
