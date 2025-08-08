import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import initialize_agent, AgentType
import time
import os
import pandas as pd
from dotenv import load_dotenv
from duckduckgo_search.exceptions import DuckDuckGoSearchException

# =========================
# Load environment variables
# =========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in .env file. Please add it before running.")
    st.stop()

# =========================
# Arxiv, Wikipedia, and Search Tools
# =========================
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# =========================
# Feedback Storage Setup
# =========================
FEEDBACK_FILE = "feedback.csv"
if os.path.exists(FEEDBACK_FILE):
    feedback_df = pd.read_csv(FEEDBACK_FILE)
else:
    feedback_df = pd.DataFrame(columns=["query", "feedback"])

# =========================
# Streamlit UI
# =========================
st.title("🔎 LangChain - Search & Evaluation Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a search-enabled chatbot. What can I help you find today?"}
    ]

if "metrics" not in st.session_state:
    st.session_state["metrics"] = {
        "total_queries": 0,
        "total_time": 0.0,
        "precision_scores": [],
        "recall_scores": [],
        "mrr_scores": []
    }

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# =========================
# Evaluation Functions
# =========================
def evaluate_relevance(retrieved_docs, user_query):
    query_terms = set(user_query.lower().split())
    relevant_docs = [doc for doc in retrieved_docs if query_terms & set(doc.lower().split())]
    precision = len(relevant_docs) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_docs) / len(query_terms) if query_terms else 0
    return precision, recall

def reciprocal_rank(retrieved_docs, user_query):
    query_terms = set(user_query.lower().split())
    for rank, doc in enumerate(retrieved_docs, start=1):
        if query_terms & set(doc.lower().split()):
            return 1 / rank
    return 0

def safe_ddg_search(query):
    """Prevent crash if DuckDuckGo rate-limits"""
    try:
        return search.run(query)
    except DuckDuckGoSearchException:
        return "DuckDuckGo rate limit hit. Please wait or try another query."

# =========================
# Chat Input Handling
# =========================
if prompt := st.chat_input(placeholder="Ask me anything!"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # LLM and Tools
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [arxiv, wiki, search]

    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        start_time = time.time()
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except DuckDuckGoSearchException:
            response = safe_ddg_search(prompt)
        except Exception as e:
            response = f"Search failed: {e}"
        elapsed_time = time.time() - start_time

        # Simulated retrieved docs
        retrieved_docs = [response]
        precision, recall = evaluate_relevance(retrieved_docs, prompt)
        mrr = reciprocal_rank(retrieved_docs, prompt)

        # Store metrics
        st.session_state["metrics"]["total_queries"] += 1
        st.session_state["metrics"]["total_time"] += elapsed_time
        st.session_state["metrics"]["precision_scores"].append(precision)
        st.session_state["metrics"]["recall_scores"].append(recall)
        st.session_state["metrics"]["mrr_scores"].append(mrr)

        # Store assistant message
        st.session_state.messages.append({'role': 'assistant', "content": response})

        # Show response and metrics
        st.write(response)
        st.markdown(
            f"**Precision@k:** {precision:.2f} | **Recall@k:** {recall:.2f} | "
            f"**MRR:** {mrr:.2f} | **Latency:** {elapsed_time:.2f} sec"
        )

        # Feedback buttons
        col1, col2 = st.columns(2)
        if col1.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
            feedback_df.loc[len(feedback_df)] = [prompt, "positive"]
            feedback_df.to_csv(FEEDBACK_FILE, index=False)
        if col2.button("👎 Not Helpful", key=f"down_{len(st.session_state.messages)}"):
            feedback_df.loc[len(feedback_df)] = [prompt, "negative"]
            feedback_df.to_csv(FEEDBACK_FILE, index=False)

# =========================
# Sidebar Metrics Summary
# =========================
st.sidebar.title("Metrics Summary")
if st.sidebar.button("Show Summary"):
    total_queries = st.session_state["metrics"]["total_queries"]
    avg_latency = (st.session_state["metrics"]["total_time"] / total_queries) if total_queries > 0 else 0
    avg_precision = sum(st.session_state["metrics"]["precision_scores"]) / total_queries if total_queries > 0 else 0
    avg_recall = sum(st.session_state["metrics"]["recall_scores"]) / total_queries if total_queries > 0 else 0
    avg_mrr = sum(st.session_state["metrics"]["mrr_scores"]) / total_queries if total_queries > 0 else 0

    st.sidebar.write(f"Total Queries: {total_queries}")
    st.sidebar.write(f"Average Latency: {avg_latency:.2f} sec")
    st.sidebar.write(f"Average Precision@k: {avg_precision:.2f}")
    st.sidebar.write(f"Average Recall@k: {avg_recall:.2f}")
    st.sidebar.write(f"Average MRR: {avg_mrr:.2f}")
    st.sidebar.write(f"Total Feedback: {len(feedback_df)}")

    if not feedback_df.empty:
        positive_ratio = len(feedback_df[feedback_df["feedback"] == "positive"]) / len(feedback_df)
        st.sidebar.write(f"Positive Feedback Ratio: {positive_ratio:.2f}")
    else:
        st.sidebar.write("No feedback yet")
