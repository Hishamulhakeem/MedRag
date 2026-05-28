import json
import logging
from typing import AsyncGenerator, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from backend.config import settings

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are an expert AI medical report summarizer.
Your goal is to provide a clear, concise, and structured patient-friendly summary of the medical report text and any detected abnormalities.

Format your response in Markdown using the following structure:
1. **Report Details**: Date, patient name (if mentioned in text, otherwise 'Not specified'), page count.
2. **Overall Clinical Status**: A brief summary assessment (e.g., 'Normal', 'Attention Recommended', or 'Critical').
3. **Significant Findings**: Highlight any values that are outside normal reference ranges with their levels (e.g., HIGH, LOW).
4. **Patient-Friendly Explanations**: Explain what these findings mean in simple, plain language.
5. **Suggested Next Steps**: General actions the patient can take (e.g., lifestyle, diet, follow-up tests).
6. **Clinical Disclaimer**: A standard reminder that this is an AI analysis and they must consult their primary physician.
"""

class SummarizerService:
    def _get_llm(self):
        """Retrieve primary/fallback LLM, or return None for Mock mode."""
        if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY:
            return None
        
        llms = []
        if settings.OPENAI_API_KEY:
            try:
                llms.append(ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.2,
                    openai_api_key=settings.OPENAI_API_KEY
                ))
            except Exception as e:
                logger.error(f"Failed to load OpenAI for summarizer: {str(e)}")

        if settings.ANTHROPIC_API_KEY:
            try:
                llms.append(ChatAnthropic(
                    model_name="claude-3-5-sonnet-20241022",
                    temperature=0.2,
                    anthropic_api_key=settings.ANTHROPIC_API_KEY
                ))
            except Exception as e:
                logger.error(f"Failed to load Anthropic for summarizer: {str(e)}")

        if not llms:
            return None
        
        primary = llms[0]
        if len(llms) > 1:
            return primary.with_fallbacks(llms[1:])
        return primary

    async def generate_summary_stream(
        self, 
        extracted_text: str, 
        abnormalities: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Streams the medical report summary as SSE chunks."""
        llm = self._get_llm()

        if llm is None:
            # --- OFFLINE / MOCK SUMMARY MODE ---
            logger.info("Executing Mock summary generation.")
            token_payload = json.dumps({"token": "[Offline Demo Mode: No API keys configured]\n\n"})
            yield f"data: {token_payload}\n\n"
            
            # Format abnormalities for the mock output
            abnormal_items = [a for a in abnormalities if a["status"] != "NORMAL"]
            
            mock_text = "# Medical Report Summary\n\n"
            mock_text += "## 1. Report Details\n"
            mock_text += "- **Date**: Not specified in context\n"
            mock_text += "- **Patient Name**: Patient Profile\n"
            mock_text += f"- **Abnormalities Detected**: {len(abnormal_items)} parameter(s)\n\n"
            
            mock_text += "## 2. Overall Clinical Status\n"
            if any(a["status"] == "CRITICAL" for a in abnormalities):
                mock_text += "**CRITICAL ALERT**: Immediate medical attention is recommended. Multiple clinical parameters are significantly outside safe operating bounds.\n\n"
            elif abnormal_items:
                mock_text += "**ATTENTION RECOMMENDED**: Some test values are outside the normal reference ranges. Discussion with your healthcare provider is advised.\n\n"
            else:
                mock_text += "**NORMAL**: All tested biomarkers are within standard reference ranges.\n\n"
            
            mock_text += "## 3. Significant Findings\n"
            if not abnormal_items:
                mock_text += "- All parsed values are normal.\n"
            else:
                for item in abnormal_items:
                    status_badge = f"**[{item['status']}]**"
                    mock_text += f"- **{item['test_name']}**: {item['value']} {item['unit']} (Normal: {item['normal_range']}) {status_badge}\n"
            mock_text += "\n"
            
            mock_text += "## 4. Patient-Friendly Explanations\n"
            if not abnormal_items:
                mock_text += "- Your laboratory results indicate healthy biological functions in the tested categories.\n"
            else:
                for item in abnormal_items:
                    mock_text += f"- **{item['test_name']}**: {item['explanation']}\n"
            mock_text += "\n"
            
            mock_text += "## 5. Suggested Next Steps\n"
            if abnormal_items:
                mock_text += "1. **Consult a Doctor**: Schedule an appointment with your primary care provider to review these findings.\n"
                mock_text += "2. **Monitor Levels**: Retest abnormal parameters in 3-6 months as recommended by your physician.\n"
                mock_text += "3. **Hydration & Diet**: Maintain appropriate lifestyle measures tailored to the specific findings.\n"
            else:
                mock_text += "1. **Routine Checkups**: Maintain your annual physical schedules.\n"
                mock_text += "2. **Healthy Habits**: Continue your balanced lifestyle and dietary routines.\n"
            mock_text += "\n"
            
            mock_text += "## 6. Clinical Disclaimer\n"
            mock_text += "> **IMPORTANT NOTICE**: This summary is AI-generated for informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always discuss these findings with a licensed physician.\n"

            # Stream tokens to UI
            for chunk in [mock_text[i:i+6] for i in range(0, len(mock_text), 6)]:
                chunk_payload = json.dumps({"token": chunk})
                yield f"data: {chunk_payload}\n\n"
                import asyncio
                await asyncio.sleep(0.01)
        else:
            try:
                abnormalities_str = json.dumps(abnormalities, indent=2)
                messages = [
                    SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(content=f"Report Raw Text:\n{extracted_text}\n\nAbnormal Values JSON:\n{abnormalities_str}")
                ]
                
                async for chunk in llm.astream(messages):
                    chunk_payload = json.dumps({"token": chunk.content})
                    yield f"data: {chunk_payload}\n\n"
            except Exception as e:
                logger.error(f"Error during summarization LLM run: {str(e)}")
                err_payload = json.dumps({"token": f"\n\n[Summarizer Error: {str(e)}]"})
                yield f"data: {err_payload}\n\n"

        done_payload = json.dumps({"done": True})
        yield f"data: {done_payload}\n\n"
