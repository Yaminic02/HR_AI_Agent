#!/usr/bin/env python3
"""
HR AI Agent — Excel Upload + Rule-Based + Local LLM (Ollama)
SHORT answers | NO refusals | Data-first

==============================
PREREQUISITES (INSTALL FIRST)
==============================

1️⃣ Python 3.9+
   https://www.python.org/downloads/

2️⃣ Create virtual environment
   python -m venv venv
   venv\\Scripts\\activate        (Windows)
   source venv/bin/activate      (Mac/Linux)

3️⃣ Install Python libraries
   pip install streamlit pandas numpy requests faiss-cpu reportlab openpyxl

==============================
OLLAMA SETUP (POINT 4 ONWARDS)
==============================

4️⃣ Download Ollama
   https://ollama.com/download

5️⃣ Install Ollama
   (Run installer and finish setup)

6️⃣ Verify installation
   ollama --version

7️⃣ Download model (run once)
   ollama pull phi3

8️⃣ Start Ollama server
   ollama serve

9️⃣ (Optional test)
   ollama run phi3

🔟 Keep Ollama terminal OPEN
    API runs at: http://localhost:11434

==============================
RUN APPLICATION
==============================

11️⃣ Start HR AI Agent
    streamlit run hr_ai_agent.py

==============================
EXCEL / CSV REQUIRED COLUMNS
==============================
employee_id
name
department
role
salary
performance_score
engagement_score
promotion_eligibility_score
attrition_risk_score
manager_name
"""


# ----------------------------
# IMPORTS
# ----------------------------
import pandas as pd
import numpy as np
import streamlit as st
import json
import requests
import faiss
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(page_title="HR AI Agent", layout="wide")
st.title("🏢 HR AI Agent")

# ----------------------------
# FILE UPLOAD (CSV / EXCEL)
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Employee Data (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.warning("Please upload an employee file to continue")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# ----------------------------
# NUMERIC SAFETY
# ----------------------------
numeric_features = [
    "performance_score",
    "engagement_score",
    "salary",
    "promotion_eligibility_score",
    "attrition_risk_score"
]

for col in numeric_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ----------------------------
# RULE LOGIC
# ----------------------------
def promotion_decision(emp):
    if (
        emp["promotion_eligibility_score"] >= 60 and
        emp["performance_score"] >= 70 and
        emp["engagement_score"] >= 70
    ):
        return "Promote"
    return "No"

def attrition_label(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"

df["promotion_status"] = df.apply(promotion_decision, axis=1)
df["attrition_level"] = df["attrition_risk_score"].apply(attrition_label)

# ----------------------------
# CORE DATA ANSWERS
# ----------------------------
def employee_details(emp):
    return (
        f"Name: {emp['name']} | "
        f"ID: {emp['employee_id']} | "
        f"Dept: {emp['department']} | "
        f"Role: {emp['role']} | "
        f"Salary: {emp['salary']} | "
        f"Perf: {emp['performance_score']} | "
        f"Promotion: {emp['promotion_status']} | "
        f"Attrition: {emp['attrition_level']}"
    )

def best_performer():
    return df.sort_values("performance_score", ascending=False).iloc[0]["name"]

def promoted_list():
    p = df[df["promotion_status"] == "Promote"]
    if p.empty:
        return "No promoted employees"
    return ", ".join(p["name"].tolist())

# ----------------------------
# LOCAL LLM (ONE-LINE ANSWER)
# ----------------------------
def llm_short_answer(question, context):
    prompt = f"""
Answer in ONE line only.
Use ONLY this data.
{json.dumps(context)}
Q: {question}
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        return r.json()["response"].strip()
    except:
        return "LLM unavailable"

# ----------------------------
# HR CHAT ROUTER (RULE FIRST)
# ----------------------------
def hr_chat(question):
    q = question.lower()

    # By Employee ID
    if "employee id" in q or "id" in q:
        digits = "".join(filter(str.isdigit, q))
        if digits:
            emp = df[df["employee_id"] == int(digits)]
            if not emp.empty:
                return employee_details(emp.iloc[0])

    # By Name
    for name in df["name"].values:
        if name.lower() in q:
            emp = df[df["name"] == name].iloc[0]
            return employee_details(emp)

    # Promotions
    if "who is promoted" in q:
        return f"Promoted: {promoted_list()}"

    # Best Performer
    if "best performer" in q:
        return f"Best performer: {best_performer()}"

    # Attrition
    if "attrition" in q:
        high = df[df["attrition_level"] == "High"]
        return f"High attrition employees: {len(high)}"

    return None

# ----------------------------
# UI TABS
# ----------------------------
tabs = st.tabs([
    "Employees",
    "Attrition & Promotion",
    "Team Report",
    "PDF Reports",
    "HR Chat"
])

# TAB 1: Employees
with tabs[0]:
    st.dataframe(df)

# TAB 2: Employee Lookup
with tabs[1]:
    eid = st.number_input("Employee ID", min_value=1, step=1)
    emp = df[df["employee_id"] == eid]
    if not emp.empty:
        emp = emp.iloc[0]
        st.write(employee_details(emp))
        st.metric("Attrition", emp["attrition_level"])
        st.metric("Promotion", emp["promotion_status"])

# TAB 3: Team Report
with tabs[2]:
    manager = st.selectbox("Manager", df["manager_name"].unique())
    team = df[df["manager_name"] == manager]
    st.write(f"Team size: {len(team)}")
    st.dataframe(team)

# TAB 4: PDF Report
with tabs[3]:
    eid = st.number_input("Employee ID for PDF", min_value=1, step=1, key="pdf")
    if st.button("Generate PDF"):
        emp = df[df["employee_id"] == eid]
        if not emp.empty:
            emp = emp.iloc[0]
            doc = SimpleDocTemplate(f"employee_{eid}.pdf", pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph(employee_details(emp), styles["Normal"])]
            doc.build(story)
            st.success("PDF created in project folder")

# TAB 5: HR Chat
with tabs[4]:
    q = st.text_input("Ask HR Question")
    if st.button("Ask"):
        answer = hr_chat(q)
        if answer:
            st.success(answer)
        else:
            index = faiss.IndexFlatL2(len(numeric_features))
            index.add(df[numeric_features].values.astype("float32"))
            _, idx = index.search(
                np.array([df[numeric_features].mean().values], dtype="float32"), 3
            )
            context = df.iloc[idx[0]].to_dict(orient="records")
            st.info(llm_short_answer(q, context))
