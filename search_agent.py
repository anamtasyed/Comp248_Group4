import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Automatically loads the .env file inside your exercise2 folder
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class BeautySearchAgent:
    """
    Search Agent for COMP 248 Group 4 (Beauty Products Inquiries).
    Responsible for query optimization, external web search, and document processing.
    """

    def __init__(self, max_results: int = 3):
        if not GROQ_API_KEY:
            raise RuntimeError(f"GROQ_API_KEY was not found in {env_path}. Check your .env file.")

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            max_retries=2,
            api_key=GROQ_API_KEY,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        self.max_results = max_results
        self.search_tool = (
            TavilySearchResults(max_results=max_results)
            if TAVILY_API_KEY
            else None
        )

    def optimize_query(self, user_query: str) -> str:
        """Rewrites a user's beauty question into a focused search engine query."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a beauty research search specialist. Rewrite the user's "
                    "question into a concise, high-precision web search query focusing on "
                    "cosmetic ingredients (INCI), product formulation, safety, or reviews. "
                    "Return ONLY the search query text, nothing else.",
                ),
                ("human", "{query}"),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke({"query": user_query})
        return response.content.strip()

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return "unknown"

    def process_url_content(self, url: str) -> List[str]:
        """Loads and splits full webpage content using WebBaseLoader."""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if not docs:
                return []
            chunks = self.splitter.split_documents(docs)
            return [chunk.page_content.strip() for chunk in chunks[:3]]
        except Exception as e:
            print(f"[Warning] Could not scrape {url}: {e}")
            return []

    def run(self, task_id: str, user_query: str, deep_scrape: bool = False) -> List[Dict[str, Any]]:
        optimized_query = self.optimize_query(user_query)
        print(f"[Search Agent] Optimized Query: {optimized_query}")

        structured_results = []

        if self.search_tool:
            raw_results = self.search_tool.invoke({"query": optimized_query})
            for idx, item in enumerate(raw_results):
                url = item.get("url", "")
                snippet = item.get("content", "")

                extra_chunks = self.process_url_content(url) if deep_scrape and url else []
                combined_text = "\n".join(extra_chunks) if extra_chunks else snippet

                result_hash = hashlib.sha256(f"{task_id}:{url}:{idx}".encode("utf-8")).hexdigest()[:12]

                structured_results.append(
                    {
                        "result_id": f"res_{result_hash}",
                        "task_id": task_id,
                        "title": item.get("title", f"Beauty Search Result {idx + 1}"),
                        "url": url,
                        "content_snippet": combined_text,
                        "source_domain": self._extract_domain(url),
                        "retrieved_at": datetime.utcnow().isoformat(),
                        "relevance_score": item.get("score", 0.85),
                    }
                )
        else:
            print("[Search Agent] TAVILY_API_KEY not found. Running fallback WebBaseLoader demo...")
            fallback_url = "https://wikipedia.org"
            chunks = self.process_url_content(fallback_url)
            for idx, chunk_text in enumerate(chunks):
                structured_results.append(
                    {
                        "result_id": f"res_fallback_{idx}",
                        "task_id": task_id,
                        "title": "FDA Cosmetic Ingredients Reference",
                        "url": fallback_url,
                        "content_snippet": chunk_text,
                        "source_domain": "fda.gov",
                        "retrieved_at": datetime.utcnow().isoformat(),
                        "relevance_score": 0.80,
                    }
                )

        return structured_results


if __name__ == "__main__":
    agent = BeautySearchAgent(max_results=3)
    sample_query = "Is niacinamide safe to combine with vitamin C in facial serums?"
    results = agent.run(task_id="task_001", user_query=sample_query, deep_scrape=False)

    print("\n--- Structured Search Agent Output ---")
    for res in results:
        print(f"ID: {res['result_id']} | Domain: {res['source_domain']}")
        print(f"URL: {res['url']}")
        print(f"Snippet: {res['content_snippet'][:200]}...\n")