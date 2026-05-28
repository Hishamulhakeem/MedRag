import json
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from backend.config import settings
from backend.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

MEDICAL_SYSTEM_PROMPT = """You are MedRAG, an expert AI medical report analyst.
Rules:
1. Answer ONLY from provided context chunks. Never hallucinate values.
2. Always cite source: [Source: chunk_N] after each factual claim.
3. Explain medical terms in plain language after technical explanation.
4. Flag abnormal values with severity: [CRITICAL] [HIGH] [LOW] [BORDERLINE].
5. Always recommend consulting a licensed physician for medical decisions.
6. If asked about a value not in context, say: "This value was not found in the uploaded report."
"""

class RAGChainService:
    def __init__(self):
        self.vector_store = VectorStoreService()

    def _get_llm(self, streaming: bool = True):
        """Retrieve primary LLM (OpenAI) with fallbacks (Anthropic, Mock)."""
        # Case 1: Keys are not set -> Return Mock LLM
        if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY:
            logger.warning("No API keys found for OpenAI or Anthropic. Using Mock LLM.")
            return None

        llms = []
        
        # Add OpenAI GPT-4o as primary if key is set
        if settings.OPENAI_API_KEY:
            try:
                llms.append(ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1,
                    streaming=streaming,
                    openai_api_key=settings.OPENAI_API_KEY
                ))
            except Exception as e:
                logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")

        # Add Anthropic Claude 3.5 Sonnet as fallback if key is set
        if settings.ANTHROPIC_API_KEY:
            try:
                llms.append(ChatAnthropic(
                    model_name="claude-3-5-sonnet-20241022",
                    temperature=0.1,
                    streaming=streaming,
                    anthropic_api_key=settings.ANTHROPIC_API_KEY
                ))
            except Exception as e:
                logger.error(f"Failed to initialize ChatAnthropic: {str(e)}")

        if not llms:
            return None

        # Return the primary LLM with fallbacks
        primary = llms[0]
        if len(llms) > 1:
            return primary.with_fallbacks(llms[1:])
        return primary

    async def ask_question_stream(
        self, 
        report_id: str, 
        question: str, 
        chat_history: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Retrieves relevant context, queries the LLM chain, and yields SSE tokens.
        Yields format:
        - First chunk: data: {"sources": [...]}
        - Mid chunks: data: {"token": "..."}
        - Final chunk: data: {"done": true, "sources": [...]}
        """
        # 1. Retrieve matching chunks from vector store
        retrieved_chunks = self.vector_store.query_report(report_id, question, top_k=5)
        
        sources_list = []
        context_str = ""
        for i, chk in enumerate(retrieved_chunks):
            chunk_idx = chk["metadata"].get("chunk_index", i)
            page_num = chk["metadata"].get("page_number", 1)
            chunk_index_to_int = int(chunk_idx)
            sources_list.append({
                "chunk_text": chk["text"],
                "chunk_index": chunk_index_to_int,
                "page_number": int(page_num),
                "relevance_score": float(chk["score"])
            })
            context_str += f"[Source: chunk_{chunk_index_to_int}] (Page {page_num})\n{chk['text']}\n\n"

        if not context_str.strip():
            context_str = "No medical report data is available in context."

        # Yield retrieved sources first
        sources_payload = json.dumps({"sources": sources_list})
        yield f"data: {sources_payload}\n\n"

        # 2. Build message prompt
        messages = [SystemMessage(content=MEDICAL_SYSTEM_PROMPT)]
        
        # Load chat history into memory
        for msg in chat_history[-10:]:  # Limit to last 10 messages
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Add the human prompt with loaded context
        human_content = f"Context:\n{context_str}\n\nQuestion: {question}"
        messages.append(HumanMessage(content=human_content))

        # 3. Stream Response
        llm = self._get_llm(streaming=True)
        
        if llm is None:
            # --- OFFLINE / MOCK RAG MODE ---
            # Generate a mock response based on keyword presence in the retrieved chunks
            logger.info("Executing Mock RAG generation.")
            token_payload = json.dumps({"token": "[Offline Demo Mode: No API keys configured]\n\n"})
            yield f"data: {token_payload}\n\n"
            
            # Find the best chunk that matches terms
            best_chunk_text = ""
            best_chunk_idx = 0
            if sources_list:
                best_source = sources_list[0]
                best_chunk_text = best_source["chunk_text"]
                best_chunk_idx = best_source["chunk_index"]

            if "abnormal" in question.lower() or "flag" in question.lower() or "range" in question.lower():
                mock_answer = f"Based on the analysis of the report, we detected several values. Referring to the document, we see values listed like: '{best_chunk_text[:180]}...' [Source: chunk_{best_chunk_idx}]. Please consult a physician for full diagnostic reviews."
            elif "summary" in question.lower() or "summarize" in question.lower():
                mock_answer = f"This report contains medical results. The key contents state: '{best_chunk_text[:200]}' [Source: chunk_{best_chunk_idx}]. Remember to consult your licensed physician for medical decisions."
            else:
                mock_answer = f"Regarding your question, the report context mentions: '{best_chunk_text[:150]}...' [Source: chunk_{best_chunk_idx}]. If you have any further symptoms, please contact your healthcare provider immediately."
            
            # Yield tokens in chunks of 5 characters to simulate streaming speed
            for chunk in [mock_answer[i:i+5] for i in range(0, len(mock_answer), 5)]:
                chunk_payload = json.dumps({"token": chunk})
                yield f"data: {chunk_payload}\n\n"
                import asyncio
                await asyncio.sleep(0.02)
        else:
            try:
                # Execute the streaming chain
                async for chunk in llm.astream(messages):
                    token = chunk.content
                    chunk_payload = json.dumps({"token": token})
                    yield f"data: {chunk_payload}\n\n"
            except Exception as e:
                logger.error(f"Error during LLM stream: {str(e)}")
                err_payload = json.dumps({"token": f"\n\n[LLM Error: {str(e)}]"})
                yield f"data: {err_payload}\n\n"

        # Finalize connection
        final_payload = json.dumps({"done": True, "sources": sources_list})
        yield f"data: {final_payload}\n\n"
