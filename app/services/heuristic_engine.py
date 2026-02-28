import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from google import genai
from google.genai import types

from app.core.constants import NIELSEN_HEURISTICS, HeuristicId, SeverityLevel
from app.core.config import settings
from app.services.omniparser_client import UIElementDetectionResult, UIElement
from app.services.rag_knowledge_base import RAGKnowledgeBase
from app.services.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)

class HeuristicViolation:
    def __init__(
        self,
        heuristic_id: str,
        criterion_id: str,
        severity: SeverityLevel,
        description: str,
        affected_elements: List[str],
        recommendation: str
    ):
        self.heuristic_id = heuristic_id
        self.criterion_id = criterion_id
        self.severity = severity
        self.description = description
        self.affected_elements = affected_elements
        self.recommendation = recommendation

    def to_dict(self):
        return {
            "heuristic_id": self.heuristic_id,
            "criterion_id": self.criterion_id,
            "severity": self.severity.value,
            "description": self.description,
            "affected_elements": self.affected_elements,
            "recommendation": self.recommendation
        }

class HeuristicLLMAnalysis:
    def __init__(
        self,
        violations: List["HeuristicViolation"],
        llm_explanation: Optional[str] = None
    ):
        self.violations = violations
        self.llm_explanation = llm_explanation

class HeuristicScore:
    def __init__(
        self,
        heuristic_id: str,
        score: int,
        max_score: int = 100,
        violations: Optional[List[HeuristicViolation]] = None,
        explanation: Optional[str] = None,
        llm_explanation: Optional[str] = None
    ):
        self.heuristic_id = heuristic_id
        self.score = score
        self.max_score = max_score
        self.violations = violations or []
        self.explanation = explanation
        self.llm_explanation = llm_explanation

    def to_dict(self):
        return {
            "heuristic_id": self.heuristic_id,
            "score": self.score,
            "max_score": self.max_score,
            "percentage": round((self.score / self.max_score) * 100, 2),
            "violations": [v.to_dict() for v in self.violations],
            "explanation": self.explanation,
            "llm_explanation": self.llm_explanation
        }

class HeuristicEvaluationResult:
    def __init__(
        self,
        overall_score: float,
        heuristic_scores: List[HeuristicScore],
        total_violations: int,
        critical_issues: int,
        evaluation_metadata: Optional[Dict[str, Any]] = None
    ):
        self.overall_score = overall_score
        self.heuristic_scores = heuristic_scores
        self.total_violations = total_violations
        self.critical_issues = critical_issues
        self.evaluation_metadata = evaluation_metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "overall_score": self.overall_score,
            "heuristic_scores": [hs.to_dict() for hs in self.heuristic_scores],
            "total_violations": self.total_violations,
            "critical_issues": self.critical_issues,
            "evaluation_metadata": self.evaluation_metadata,
            "timestamp": self.timestamp
        }

class HeuristicEvaluationEngine:
    """LLM-powered heuristic evaluation engine.
    
    This engine evaluates user interfaces against Nielsen's usability heuristics
    using Large Language Model reasoning instead of brittle rule-based logic.
    
    Key Features:
    - Uses real OmniParser output format (type, bbox, interactivity, content)
    - No reliance on hallucinated attributes (hover_state, confirmation, etc.)
    - LLM-based violation detection with structured JSON responses
    - RAG-enhanced evaluation with knowledge base context
    - Supports all 10 Nielsen heuristics (H1-H10)
    
    Version: 2.0.0-llm
    Evaluation Method: LLM-based with prompt engineering
    
    Example Usage:
        engine = request.app.state.heuristic_engine
        result = await engine.evaluate_interface(detection_result)
    """
    
    def __init__(self, rag_kb: Optional[RAGKnowledgeBase] = None):
        self.logger = logging.getLogger(__name__)
        self.llm_client = None  # Backward-compatible alias for the OpenAI client path
        self.llm_provider = "openai"
        self.openai_client = None
        self.gemini_client = None
        self.gemini_aio = None
        self.rag_kb = rag_kb
        self.initialized = False

    async def initialize(self):
        if self.initialized:
            return
        self.logger.info("Initializing Heuristic Evaluation Engine...")
        requested_provider = (settings.LLM_PROVIDER or "openai").strip().lower()
        provider = requested_provider

        # Keep `auto` deterministic for debugging/comparisons.
        if provider == "auto":
            if settings.OPENAI_API_KEY:
                provider = "openai"
            elif settings.GEMINI_API_KEY:
                provider = "gemini"
            else:
                raise RuntimeError(
                    "No LLM API key configured for auto mode. "
                    "Set OPENAI_API_KEY or GEMINI_API_KEY."
                )

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

            self.openai_client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            self.llm_client = self.openai_client
            self.gemini_client = None
            self.gemini_aio = None
            self.llm_provider = "openai"

        elif provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.gemini_aio = self.gemini_client.aio
            self.openai_client = None
            self.llm_client = None
            self.llm_provider = "gemini"
        else:
            raise RuntimeError(
                f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
                "Use one of: openai, gemini, auto"
            )

        if self.rag_kb is None:
            raise RuntimeError("RAGKnowledgeBase dependency missing. Inject via startup.")
        if not self.rag_kb.index_initialized:
            await self.rag_kb.initialize()
        self.initialized = True
        self.logger.info(
            "Heuristic Evaluation Engine initialized (provider=%s, model=%s)",
            self.llm_provider,
            self._active_model_name()
        )

    def _has_active_llm(self) -> bool:
        return (
            (self.llm_provider == "openai" and self.openai_client is not None)
            or (self.llm_provider == "gemini" and self.gemini_aio is not None)
        )

    def _active_model_name(self) -> str:
        if self.llm_provider == "gemini":
            return settings.GEMINI_MODEL
        return settings.OPENAI_MODEL

    def _extract_json_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned

    async def _llm_generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3
    ) -> str:
        if self.llm_provider == "openai":
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return (response.choices[0].message.content or "").strip() if response.choices else ""

        if self.llm_provider == "gemini":
            response = await self.gemini_aio.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    response_mime_type="application/json"
                )
            )
            return (response.text or "").strip()

        raise RuntimeError(f"Unsupported active LLM provider: {self.llm_provider}")

    async def _llm_generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 200
    ) -> str:
        if self.llm_provider == "openai":
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return (response.choices[0].message.content or "").strip() if response.choices else ""

        if self.llm_provider == "gemini":
            system_parts = []
            user_parts = []
            for msg in messages:
                role = (msg.get("role") or "user").strip().lower()
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "system":
                    system_parts.append(content)
                else:
                    user_parts.append(f"{role.upper()}:\n{content}")

            response = await self.gemini_aio.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents="\n\n".join(user_parts) if user_parts else "",
                config=types.GenerateContentConfig(
                    system_instruction="\n\n".join(system_parts) if system_parts else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return (response.text or "").strip()

        raise RuntimeError(f"Unsupported active LLM provider: {self.llm_provider}")

    def _serialize_elements_for_llm(self, elements: List[UIElement]) -> str:
        """Serialize UI elements to JSON format for LLM consumption.
        
        Only includes real OmniParser output fields:
        - type: element type
        - bbox: [x1, y1, x2, y2] bounding box
        - interactivity: boolean
        - content: text content
        """
        serialized = []
        for elem in elements:
            serialized.append({
                "type": elem.element_type,
                "bbox": [elem.bbox[0], elem.bbox[1], elem.bbox[2], elem.bbox[3]],
                "interactivity": elem.interactivity,
                "content": elem.text
            })
        return json.dumps(serialized, indent=2)

    async def _evaluate_with_llm(
        self,
        heuristic_id: HeuristicId,
        elements: List[UIElement],
        detection_result: UIElementDetectionResult
    ) -> HeuristicLLMAnalysis:
        """Use LLM to evaluate heuristic violations.
        
        This method:
        1. Serializes UI elements to JSON
        2. Retrieves heuristic definition and criteria
        3. Constructs a prompt for the LLM
        4. Parses LLM response into HeuristicViolation objects and a short summary
        """
        # Get heuristic definition
        heuristic_def = NIELSEN_HEURISTICS.get(heuristic_id)
        if not heuristic_def:
            self.logger.error(f"No definition found for {heuristic_id.value}")
            return HeuristicLLMAnalysis(violations=[], llm_explanation=None)

        # Serialize elements
        elements_json = self._serialize_elements_for_llm(elements)

        # Get RAG context if available
        rag_context = ""
        if self.rag_kb:
            try:
                rag_examples = await self.rag_kb.retrieve_relevant_context(
                    query=f"{heuristic_def['name']} violations examples",
                    heuristic_id=heuristic_id.value,
                    top_k=3
                )
                if rag_examples:
                    rag_context = "\n\nRelevant examples and best practices:\n"
                    for ex in rag_examples:
                        rag_context += f"- {ex['content']}\n"
            except Exception as e:
                self.logger.warning(f"RAG search failed: {e}")

        # Construct prompt
        criteria_text = "\n".join([
            f"- {c['id']}: {c['description']} - {c['evaluation']}"
            for c in heuristic_def['measurable_criteria']
        ])

        prompt = f"""You are a UX evaluation expert analyzing a user interface for usability violations.

**Heuristic**: {heuristic_def['name']}
**Description**: {heuristic_def['description']}

**Measurable Criteria**:
{criteria_text}

**UI Elements** (from OmniParser detection):
{elements_json}
{rag_context}

**Task**: Analyze the UI elements and identify violations of the above heuristic criteria.

For each violation found, provide:
- criterion_id: The specific criterion violated (e.g., "H1.1", "H2.2")
- severity: One of ["critical", "major", "minor", "cosmetic"]
- description: Clear description of the violation
- affected_elements: List of element content/text affected
- recommendation: Specific actionable recommendation to fix

Also provide:
- llm_explanation: A concise 2-3 sentence summary for this heuristic's key issues.
  Prioritize violations accuracy first; summary quality is secondary.

Respond with a JSON object in this shape:
{{
  "violations": [ ... ],
  "llm_explanation": "..."
}}

If no violations are found, return:
{{
  "violations": [],
  "llm_explanation": ""
}}

Example response format:
{{
  "violations": [
    {{
      "criterion_id": "H1.2",
      "severity": "major",
      "description": "Submit button lacks visible feedback state",
      "affected_elements": ["Submit"],
      "recommendation": "Add hover and active states to provide visual feedback"
    }}
  ],
  "llm_explanation": "Users may not perceive clear system feedback after interacting with controls. Add visible interaction and status cues for primary actions."
}}

Response:"""

        # Enforce RAG usage if context exists
        if rag_context:
            prompt += """

IMPORTANT INSTRUCTION:
You have access to "Relevant examples and best practices" above (from the RAG Knowledge Base).
1. If a violation matches a provided example validation, you MUST cite the example source in your 'recommendation'.
2. Format citations as: "Recommendation text... (Ref: [Source Name/Pattern])"
"""

        try:
            # Call LLM
            content = await self._llm_generate_json(
                system_prompt="You are a UX evaluation expert. Respond only with valid JSON.",
                user_prompt=prompt,
                temperature=0.3
            )

            # Parse response
            self.logger.debug(f"LLM response for {heuristic_id.value}: {content}")
            
            # Accept both the new object shape and older array-only responses.
            parsed = json.loads(self._extract_json_text(content))
            
            llm_explanation_text = None
            violations_data = parsed
            if isinstance(parsed, dict):
                llm_explanation_text = (
                    parsed.get("llm_explanation")
                    or parsed.get("summary")
                    or parsed.get("explanation")
                    or ""
                ).strip() or None
                # If wrapped in object, try common keys
                for key in ['violations', 'results', 'findings', 'issues']:
                    if key in parsed:
                        violations_data = parsed[key]
                        break
            
            if not isinstance(violations_data, list):
                self.logger.warning(f"LLM response not a list: {violations_data}")
                return HeuristicLLMAnalysis(
                    violations=[],
                    llm_explanation=llm_explanation_text
                )

            # Convert to HeuristicViolation objects
            violations = []
            for v_data in violations_data:
                try:
                    # Map severity string to enum
                    severity_map = {
                        "critical": SeverityLevel.CRITICAL,
                        "major": SeverityLevel.MAJOR,
                        "minor": SeverityLevel.MINOR,
                        "cosmetic": SeverityLevel.COSMETIC
                    }
                    severity = severity_map.get(v_data.get("severity", "minor").lower(), SeverityLevel.MINOR)

                    violation = HeuristicViolation(
                        heuristic_id=heuristic_id.value,
                        criterion_id=v_data.get("criterion_id", f"{heuristic_id.value}.0"),
                        severity=severity,
                        description=v_data.get("description", "Unspecified violation"),
                        affected_elements=v_data.get("affected_elements", []),
                        recommendation=v_data.get("recommendation", "Review and address this issue")
                    )
                    violations.append(violation)
                except Exception as e:
                    self.logger.error(f"Error parsing violation: {e}, data: {v_data}")
                    continue

            return HeuristicLLMAnalysis(
                violations=violations,
                llm_explanation=llm_explanation_text
            )

        except Exception as e:
            self.logger.error(f"LLM evaluation failed for {heuristic_id.value}: {e}")
            raise ValueError(f"AI Service Unavailable: {str(e)}")

    def calculate_score(self, violations: List[HeuristicViolation], heuristic_id: str) -> tuple[int, str]:
        heuristic_def = NIELSEN_HEURISTICS.get(HeuristicId(heuristic_id))
        if not heuristic_def:
            return 100, "No criteria defined"

        total_deduction = 0
        explanation_parts = []

        for violation in violations:
            criterion = next(
                (c for c in heuristic_def["measurable_criteria"] if c["id"] == violation.criterion_id),
                None
            )
            if criterion:
                weights = criterion["severity_weights"]
                deduction = weights.get(violation.severity.value, 1)
                total_deduction += deduction
                explanation_parts.append(
                    f"{violation.severity.value}: {violation.description}"
                )

        score = max(0, 100 - total_deduction)
        explanation = f"Score {score}/100 - {len(violations)} violations: " + "; ".join(explanation_parts)

        return score, explanation

    async def evaluate_heuristic(
        self,
        heuristic_id: HeuristicId,
        elements: List[UIElement],
        detection_result: UIElementDetectionResult
    ) -> HeuristicScore:
        """Evaluate a specific heuristic using LLM-based analysis.
        
        This method uses LLM to analyze UI elements and detect violations,
        eliminating reliance on hallucinated OmniParser attributes.
        """
        self.logger.info(f"Evaluating heuristic {heuristic_id.value} with LLM")

        # One LLM call returns both violations (structured) and an optional summary.
        analysis = await self._evaluate_with_llm(heuristic_id, elements, detection_result)
        violations = analysis.violations

        score, explanation = self.calculate_score(violations, heuristic_id.value)

        llm_explanation = analysis.llm_explanation

        return HeuristicScore(
            heuristic_id=heuristic_id.value,
            score=score,
            violations=violations,
            explanation=explanation,
            llm_explanation=llm_explanation
        )

    async def _evaluate_h4_consistency_legacy(
        self,
        elements: List[UIElement],
        detection_result: UIElementDetectionResult
    ) -> List[HeuristicViolation]:
        """Legacy rule-based H4 evaluation (deprecated).
        
        This method is kept for reference but no longer used.
        LLM-based evaluation is now preferred.
        """
        violations = []
        
        # H4.1: Check button dimension consistency
        buttons = [e for e in elements if e.element_type == "button"]
        if len(buttons) >= 2:
            # Group buttons and check dimension consistency using bbox
            button_dimensions = []
            for btn in buttons:
                # Calculate width and height from bbox
                width = btn.width
                height = btn.height
                if width > 0 and height > 0:
                    button_dimensions.append((btn.text, width, height))
            
            if len(button_dimensions) >= 2:
                # Check if button heights are consistent (allow 10% variance)
                heights = [d[2] for d in button_dimensions]
                avg_height = sum(heights) / len(heights)
                inconsistent_buttons = [
                    d[0] for d in button_dimensions 
                    if abs(d[2] - avg_height) > avg_height * 0.1
                ]
                
                if inconsistent_buttons:
                    violations.append(HeuristicViolation(
                        heuristic_id="H4",
                        criterion_id="H4.1",
                        severity=SeverityLevel.MINOR,
                        description=f"Inconsistent button heights detected",
                        affected_elements=inconsistent_buttons,
                        recommendation="Standardize button heights for visual consistency"
                    ))
        
        # H4.2: Check typography consistency for similar elements
        # Infer heading levels from text elements based on bbox height
        from app.services.omniparser_client import infer_heading_level
        
        text_elements = [e for e in elements if e.element_type in ["text", "heading"]]
        if len(text_elements) >= 2:
            # Group by inferred heading level
            headings_by_level = {}
            for elem in text_elements:
                level = infer_heading_level(elem, text_elements)
                if level:  # Only consider actual headings
                    if level not in headings_by_level:
                        headings_by_level[level] = []
                    headings_by_level[level].append((elem.text, elem.height))
            
            # Check consistency within each heading level
            for level, headings in headings_by_level.items():
                if len(headings) >= 2:
                    heights = [h[1] for h in headings]
                    avg_height = sum(heights) / len(heights)
                    # Check for variance > 15% within same level
                    inconsistent = [
                        h[0] for h in headings 
                        if abs(h[1] - avg_height) > avg_height * 0.15
                    ]
                    
                    if inconsistent:
                        violations.append(HeuristicViolation(
                            heuristic_id="H4",
                            criterion_id="H4.2",
                            severity=SeverityLevel.MINOR,
                            description=f"Inconsistent heights for level {level} headings",
                            affected_elements=inconsistent,
                            recommendation="Ensure consistent sizing for headings at the same level"
                        ))
        
        # H4.3: Color consistency check removed
        # Real Omniparser output doesn't include color information
        # Color analysis would require separate vision model analysis
        
        # H4.4: Check terminology consistency
        button_texts = [btn.text.lower() for btn in buttons]
        
        # Define conflicting term pairs
        conflicting_terms = [
            (["delete", "remove"], "delete/remove"),
            (["save", "submit"], "save/submit"),
            (["cancel", "close", "exit"], "cancel/close/exit"),
            (["edit", "modify", "update"], "edit/modify/update"),
            (["add", "create", "new"], "add/create/new")
        ]
        
        for terms, term_group in conflicting_terms:
            found_terms = [t for t in terms if any(t in bt for bt in button_texts)]
            if len(found_terms) > 1:
                violations.append(HeuristicViolation(
                    heuristic_id="H4",
                    criterion_id="H4.4",
                    severity=SeverityLevel.MAJOR,
                    description=f"Inconsistent terminology: both {' and '.join(found_terms)} used for similar actions",
                    affected_elements=found_terms,
                    recommendation=f"Choose one term from {term_group} and use it consistently"
                ))
        
        return violations

    async def evaluate_interface(
        self,
        detection_result: UIElementDetectionResult
    ) -> HeuristicEvaluationResult:
        self.logger.info("Starting comprehensive heuristic evaluation...")

        if not self.initialized:
            await self.initialize()

        heuristic_ids = [
            HeuristicId.H1_VISIBILITY_OF_SYSTEM_STATUS,
            HeuristicId.H2_MATCH_BETWEEN_SYSTEM_AND_REAL_WORLD,
            HeuristicId.H3_USER_CONTROL_AND_FREEDOM,
            HeuristicId.H4_CONSISTENCY_AND_STANDARDS,
            HeuristicId.H5_ERROR_PREVENTION,
            HeuristicId.H6_RECOGNITION_RATHER_THAN_RECALL,
            HeuristicId.H7_FLEXIBILITY_AND_EFFICIENCY_OF_USE,
            HeuristicId.H8_AESTHETIC_AND_MINIMALIST_DESIGN,
            HeuristicId.H9_HELP_USERS_RECOGNIZE_RECOVER_FROM_ERRORS,
            HeuristicId.H10_HELP_AND_DOCUMENTATION
        ]

        heuristic_scores = []
        total_violations = 0
        critical_issues = 0

        for heuristic_id in heuristic_ids:
            score = await self.evaluate_heuristic(
                heuristic_id,
                detection_result.elements,
                detection_result
            )
            heuristic_scores.append(score)
            total_violations += len(score.violations)
            critical_issues += sum(1 for v in score.violations if v.severity == SeverityLevel.CRITICAL)

        overall_score = sum(hs.score for hs in heuristic_scores) / len(heuristic_scores)

        result = HeuristicEvaluationResult(
            overall_score=round(overall_score, 2),
            heuristic_scores=heuristic_scores,
            total_violations=total_violations,
            critical_issues=critical_issues,
            evaluation_metadata={
                "total_elements": len(detection_result.elements),
                "evaluation_version": "2.0.0-llm",
                "evaluation_method": "llm-based",
                "llm_provider": self.llm_provider,
                "llm_model": self._active_model_name()
            }
        )

        self.logger.info(
            f"Evaluation complete: Overall score {overall_score:.2f}, "
            f"{total_violations} violations, {critical_issues} critical"
        )

        return result

    async def _llm_explain_heuristic(
        self,
        heuristic_id: HeuristicId,
        detection_result: UIElementDetectionResult,
        score: int,
        violations: List[HeuristicViolation]
    ) -> Optional[str]:
        """Generate LLM-based explanation for heuristic evaluation results.
        
        Uses only real OmniParser output fields (type, bbox, interactivity, content).
        Gracefully handles missing/optional fields.
        """
        if not self._has_active_llm():
            return None

        try:
            # Serialize elements using only real OmniParser fields
            elements_brief = [
                {
                    "type": e.element_type,
                    "text": e.text,
                    "interactivity": e.interactivity,  # Use real field name
                    "bbox": e.bbox  # Include bounding box for context
                }
                for e in detection_result.elements[:20]
            ]

            prompt = (
                "You are a UX heuristic expert. Summarize the key usability issues for "
                f"{heuristic_id.value} using the detected UI elements and violations. "
                "Keep it concise (2-3 sentences) and actionable."
            )

            messages = [
                {
                    "role": "system",
                    "content": "Provide clear, concise UX heuristic explanations."
                },
                {
                    "role": "user",
                    "content": (
                        f"Heuristic: {heuristic_id.value}\n"
                        f"Score: {score}/100\n"
                        f"Violations: {[v.description for v in violations]}\n"
                        f"Elements: {elements_brief}"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            content = await self._llm_generate_text(
                messages=messages,
                temperature=0.2,
                max_tokens=200
            )

            return content or None

        except Exception as exc:
            self.logger.error(f"LLM explanation failed for {heuristic_id.value}: {exc}")
            raise ModelInferenceError(
                message=f"Failed to generate explanation for heuristic {heuristic_id.value}",
                details={"heuristic_id": heuristic_id.value, "error": str(exc)}
            )
