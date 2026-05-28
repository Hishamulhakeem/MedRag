import pytest
import json
from unittest.mock import patch, MagicMock
from backend.services.rag_chain import RAGChainService

@pytest.mark.anyio
async def test_rag_chain_stream_mock():
    rag_service = RAGChainService()
    
    # Mock ChromaDB query return value
    mock_retrieved = [
        {
            "id": "chk_0",
            "text": "The patient fasting blood glucose level was 98 mg/dL.",
            "metadata": {"chunk_index": 0, "page_number": 1},
            "score": 0.85
        }
    ]
    
    with patch.object(rag_service.vector_store, 'query_report', return_value=mock_retrieved):
        # We ensure API keys are empty to force Mock Mode
        with patch('backend.services.rag_chain.settings.OPENAI_API_KEY', ""), \
             patch('backend.services.rag_chain.settings.ANTHROPIC_API_KEY', ""):
            
            stream = rag_service.ask_question_stream(
                report_id="fake-id",
                question="What is the glucose value?",
                chat_history=[]
            )
            
            events = []
            async for chunk in stream:
                events.append(chunk)
                
            # Verify we received multiple events
            assert len(events) >= 3
            
            # First event should yield sources
            first_event = events[0]
            assert "sources" in first_event
            parsed_first = json.loads(first_event.replace("data: ", "").strip())
            assert parsed_first["sources"][0]["chunk_index"] == 0
            assert parsed_first["sources"][0]["page_number"] == 1
            assert "glucose" in parsed_first["sources"][0]["chunk_text"].lower()

            # The final event should signal completion
            last_event = events[-1]
            assert "done" in last_event
            parsed_last = json.loads(last_event.replace("data: ", "").strip())
            assert parsed_last["done"] is True
