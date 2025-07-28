import streamlit as st
import requests
import os
from google import genai
from google.genai import types

# ----------------------------- #
# CONFIGURATION & SECRETS
# ----------------------------- #
AUTORAG_URL = "https://api.cloudflare.com/client/v4/accounts/eb80b92269b2dc4b8ceb558428ebf2f7/autorag/rags/ipodb/search"
API_TOKEN = st.secrets["API_TOKEN"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ----------------------------- #
# Cloudflare AutoRAG Query
# ----------------------------- #
def query_autorag(query: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
    }

    payload = {
        "query": query,
        "queryLength": len(query),
        "maxResults": 20,
        "scoreThreshold": 0.15,
        "autoragDatabase": "ipodb",
        "entityType": "IPO_COMPLIANCE"
    }

    response = requests.post(AUTORAG_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        matches = data.get("result", [])
        return matches
    else:
        st.error(f"AutoRAG API Error: {response.status_code} - {response.text}")
        return []

# ----------------------------- #
# Gemini AI Analysis
# ----------------------------- #
def gemini_analysis(input_text: str):
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(input_text)],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
    )

    output = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=contents,
        config=generate_content_config,
    ):
        output += chunk.text
    return output

# ----------------------------- #
# Streamlit App UI
# ----------------------------- #
st.set_page_config(page_title="DRHP AutoRAG + Gemini Analyzer", layout="wide")
st.title("📘 DRHP Compliance Analyzer (AutoRAG + Gemini AI)")

with st.form("query_form"):
    query_input = st.text_area("Enter your DRHP-related query or topic:", height=150)
    submitted = st.form_submit_button("Search and Analyze")

if submitted and query_input.strip():
    st.info("🔍 Searching AutoRAG...")
    matches = query_autorag(query_input)

    if not matches:
        st.warning("No relevant results found.")
    else:
        st.success(f"✅ Retrieved {len(matches)} relevant documents from AutoRAG.")

        # Display retrieved context (optional preview)
        with st.expander("View AutoRAG Retrieved Context"):
            for i, match in enumerate(matches, 1):
                st.markdown(f"**Result {i}**\n\n`Score: {match.get('score', 0)}`\n\n{match.get('text', '')[:1000]}...\n\n---")

        # Concatenate relevant chunks for Gemini
        context_text = "\n\n---\n\n".join(match.get("text", "") for match in matches)

        prompt = f"""
Given the following IPO/DRHP excerpts retrieved via semantic search, analyze them for any red flags, regulatory compliance gaps, and other important takeaways for merchant bankers or legal teams.

### Excerpts:
{context_text}

### Instructions:
- Highlight key compliance issues (SEBI ICDR, LODR, Companies Act)
- Point out any missing or vague disclosures
- Suggest additional checks if necessary
- Be clear and actionable in tone
"""

        st.info("🧠 Analyzing with Gemini AI...")
        output = gemini_analysis(prompt)

        st.subheader("📊 Gemini AI Output")
        st.markdown(output)

else:
    st.warning("Please enter a query to begin.")
