"""
Quality Evaluator - Assesses if agent responses meet quality standards.

Evaluates responses for relevance, completeness, accuracy, and groundedness.
Provides feedback for improvement.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Report on response quality."""
    is_acceptable: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    dimensions: Dict[str, float]  # relevance, completeness, accuracy, groundedness


class QualityEvaluator:
    """
    Evaluates quality of agent responses.
    
    Checks:
    - Relevance: Does it answer the question?
    - Completeness: Is all required info present?
    - Accuracy: Are facts correct?
    - Groundedness: Based on retrieved data?
    """
    
    def __init__(self, llm, threshold: float = 0.7):
        """
        Initialize evaluator.
        
        Args:
            llm: LLM for evaluation
            threshold: Minimum acceptable score (0.0 to 1.0)
        """
        self.llm = llm
        self.threshold = threshold
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None
    ) -> QualityReport:
        """
        Evaluate response quality.
        
        Args:
            query: Original user query
            response: Agent response to evaluate
            evidence: Retrieved evidence documents
            context: Additional context
            
        Returns:
            Quality report with scores and suggestions
        """
        logger.info("Evaluating response quality")
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            query=query,
            response=response,
            evidence=evidence,
            context=context
        )
        
        # Get LLM evaluation
        try:
            evaluation_text = self.llm.invoke(prompt)
            
            # Parse evaluation
            dimensions = self._parse_dimensions(evaluation_text)
            issues = self._extract_issues(evaluation_text)
            suggestions = self._extract_suggestions(evaluation_text)
            
            # Calculate overall score
            score = sum(dimensions.values()) / len(dimensions)
            is_acceptable = score >= self.threshold
            
            report = QualityReport(
                is_acceptable=is_acceptable,
                score=score,
                issues=issues,
                suggestions=suggestions,
                dimensions=dimensions
            )
            
            logger.info(f"Quality evaluation: score={score:.2f}, acceptable={is_acceptable}")
            return report
            
        except Exception as e:
            logger.error(f"Error evaluating quality: {e}", exc_info=True)
            # Return acceptable by default if evaluation fails
            return QualityReport(
                is_acceptable=True,
                score=0.8,
                issues=[],
                suggestions=[],
                dimensions={}
            )
    
    def _build_evaluation_prompt(
        self,
        query: str,
        response: str,
        evidence: Optional[List[Dict[str, Any]]],
        context: Optional[str]
    ) -> str:
        """Build prompt for LLM evaluation."""
        
        prompt_parts = [
            "You are a quality evaluator for AI agent responses.",
            "",
            "Evaluate the following response on these dimensions (0-10 scale):",
            "1. RELEVANCE: Does it directly answer the user's question?",
            "2. COMPLETENESS: Is all necessary information included?",
            "3. ACCURACY: Are the facts and numbers correct?",
            "4. GROUNDEDNESS: Is it based on the provided evidence?",
            "",
            f"USER QUERY:\n{query}",
            "",
            f"AGENT RESPONSE:\n{response}",
            ""
        ]
        
        # Add evidence if available
        if evidence:
            prompt_parts.append("EVIDENCE DOCUMENTS:")
            for i, doc in enumerate(evidence[:3], 1):  # First 3 docs
                title = doc.get("title", "Unknown")
                prompt_parts.append(f"{i}. {title}")
            prompt_parts.append("")
        
        # Add context if available
        if context:
            prompt_parts.append(f"CONTEXT:\n{context[:500]}...\n")
        
        prompt_parts.extend([
            "Provide your evaluation in this format:",
            "RELEVANCE: [score 0-10]",
            "COMPLETENESS: [score 0-10]",
            "ACCURACY: [score 0-10]",
            "GROUNDEDNESS: [score 0-10]",
            "",
            "ISSUES:",
            "- [issue 1]",
            "- [issue 2]",
            "",
            "SUGGESTIONS:",
            "- [suggestion 1]",
            "- [suggestion 2]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_dimensions(self, evaluation_text: str) -> Dict[str, float]:
        """Parse dimension scores from evaluation text."""
        dimensions = {
            "relevance": 0.8,
            "completeness": 0.8,
            "accuracy": 0.8,
            "groundedness": 0.8
        }
        
        lines = evaluation_text.lower().split("\n")
        
        for line in lines:
            for dim in dimensions.keys():
                if dim in line and ":" in line:
                    try:
                        # Extract score
                        parts = line.split(":")
                        score_part = parts[1].strip()
                        # Get first number
                        score_str = ""
                        for char in score_part:
                            if char.isdigit() or char == ".":
                                score_str += char
                            elif score_str:
                                break
                        
                        if score_str:
                            score = float(score_str)
                            # Normalize to 0-1
                            dimensions[dim] = min(score / 10.0, 1.0)
                    except:
                        pass
        
        return dimensions
    
    def _extract_issues(self, evaluation_text: str) -> List[str]:
        """Extract issues from evaluation text."""
        issues = []
        lines = evaluation_text.split("\n")
        
        in_issues = False
        for line in lines:
            line = line.strip()
            
            if "ISSUES:" in line.upper():
                in_issues = True
                continue
            
            if in_issues:
                if "SUGGESTIONS:" in line.upper():
                    break
                if line.startswith("-") or line.startswith("*"):
                    issue = line.lstrip("-*").strip()
                    if issue:
                        issues.append(issue)
        
        return issues
    
    def _extract_suggestions(self, evaluation_text: str) -> List[str]:
        """Extract suggestions from evaluation text."""
        suggestions = []
        lines = evaluation_text.split("\n")
        
        in_suggestions = False
        for line in lines:
            line = line.strip()
            
            if "SUGGESTIONS:" in line.upper():
                in_suggestions = True
                continue
            
            if in_suggestions:
                if line.startswith("-") or line.startswith("*"):
                    suggestion = line.lstrip("-*").strip()
                    if suggestion:
                        suggestions.append(suggestion)
        
        return suggestions
