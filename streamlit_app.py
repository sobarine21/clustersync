import streamlit as st
import requests
import os
from google import genai
from google.genai import types
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
ACCOUNT_ID = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
AUTORAG_NAME = st.secrets["CLOUDFLARE_AUTORAG_NAME"]
API_TOKEN = st.secrets["CLOUDFLARE_API_TOKEN"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

AUTORAG_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/autorag/rags/{AUTORAG_NAME}/search"

# Enable debug sidebar
debug_mode = st.sidebar.checkbox("🔍 Enable Debug Logs")

# -----------------------------
# FUNCTION: AutoRAG Search
# -----------------------------
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
        "autoragDatabase": "ipodb"
        # "entityType" REMOVED as per request
    }

    if debug_mode:
        st.sidebar.subheader("🔧 AutoRAG Request Payload")
        st.sidebar.code(json.dumps(payload, indent=2))
        st.sidebar.subheader("🔧 API Endpoint")
        st.sidebar.code(AUTORAG_URL)

    try:
        response = requests.post(AUTORAG_URL, headers=headers, json=payload)
        if debug_mode:
            st.sidebar.subheader("🧾 Raw AutoRAG Response")
            st.sidebar.code(response.text)

        if response.status_code == 200:
            data = response.json()
            return data.get("result", [])
        else:
            error_data = response.json()
            error_message = error_data.get("errors", [{}])[0].get("message", "Unknown error")
            st.error(f"AutoRAG API Error {response.status_code}: {error_message}")
            return []
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return []

# -----------------------------
# FUNCTION: Gemini AI Analysis
# -----------------------------
def gemini_analysis(input_text: str):
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(input_text)],
        )
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
    )

    output = ""
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        ):
            output += chunk.text
    except Exception as e:
        st.error(f"Gemini AI Error: {str(e)}")
        return ""
    
    return output

# -----------------------------
# STREAMLIT APP UI
# -----------------------------
st.set_page_config(page_title="DRHP Compliance Analyzer", layout="wide")
st.title("📘 DRHP Compliance Analyzer (AutoRAG + Gemini AI)")

with st.form("query_form"):
    query_input = st.text_area("Enter your DRHP query or keyword(s):", height=150)
    submitted = st.form_submit_button("Search & Analyze")

if submitted and query_input.strip():
    st.info("🔍 Performing semantic search via Cloudflare AutoRAG...")
    matches = query_autorag(query_input)

    if not matches:
        st.warning("No relevant results found.")
    else:
        st.success(f"✅ Found {len(matches)} relevant context chunks.")

        # Show matched contexts
        with st.expander("🔎 View Retrieved Contexts"):
            for i, match in enumerate(matches, start=1):
                st.markdown(f"**Result {i} — Score: {match.get('score', 0):.2f}**")
                st.code(match.get("text", "")[:1500] + "..." if len(match.get("text", "")) > 1500 else match.get("text", ""))
                st.markdown("---")

        # Concatenate results into context
        context_text = "\n\n---\n\n".join([m.get("text", "") for m in matches])

        # Prompt for Gemini
        prompt = f"""
You are a regulatory compliance analyst. Review the following excerpts from a DRHP document and:

1. Identify red flags or inconsistencies
2. Point out missing or vague disclosures as per SEBI ICDR, LODR, Companies Act
3. Suggest any additional due diligence checks

### DRHP Excerpts:
{context_text}
        """

        st.info("🧠 Running Gemini AI analysis...")
        result = gemini_analysis(prompt)

        st.subheader("📊 Gemini AI Summary")
        st.markdown(result)

else:
    st.warning("Please enter a query to proceed.")
