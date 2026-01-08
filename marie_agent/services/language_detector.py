"""
Language Detector Service - Detects language from text.

Provides fast, accurate language detection for queries and content.
Used across the system for language-aware processing.
"""

from typing import Dict, List, Literal
import logging

logger = logging.getLogger(__name__)


LanguageCode = Literal["es", "en", "pt", "fr", "de", "it", "unknown"]


class LanguageDetector:
    """
    Lightweight language detection service.
    
    Uses keyword-based detection for speed.
    Falls back to pattern-based detection if needed.
    """
    
    # Common words per language
    LANGUAGE_MARKERS = {
        "es": [
            # Questions
            "qué", "cómo", "cuándo", "dónde", "cuál", "cuáles", "quién", "quiénes",
            "por qué", "para qué", "cuánto", "cuántos", "cuánta", "cuántas",
            # Common verbs
            "es", "son", "está", "están", "hay", "tiene", "tienen", "puede", "pueden",
            "hacer", "haz", "dame", "muestra", "busca", "encuentra",
            # Greetings/common
            "hola", "gracias", "por favor", "buenos", "buenas", "días", "tardes", "noches",
            # Academic specific
            "universidad", "investigador", "investigadores", "artículos", "publicaciones",
            "papers", "revista", "congreso", "tesis", "autor", "autores"
        ],
        "en": [
            # Questions
            "what", "how", "when", "where", "which", "who", "whose",
            "why", "how many", "how much",
            # Common verbs
            "is", "are", "was", "were", "have", "has", "can", "could",
            "do", "does", "did", "get", "show", "find", "search",
            # Greetings/common
            "hello", "hi", "hey", "thanks", "thank you", "please", "good",
            # Academic specific
            "university", "researcher", "researchers", "articles", "publications",
            "papers", "journal", "conference", "thesis", "author", "authors"
        ],
        "pt": [
            # Questions
            "o que", "como", "quando", "onde", "qual", "quais", "quem",
            "por que", "quanto", "quantos", "quanta", "quantas",
            # Common
            "é", "são", "está", "estão", "tem", "têm", "pode", "podem",
            "olá", "obrigado", "obrigada", "por favor", "bom", "boa",
            # Academic
            "universidade", "pesquisador", "pesquisadores", "artigos", "publicações"
        ],
        "fr": [
            # Questions
            "quoi", "comment", "quand", "où", "quel", "quelle", "qui",
            "pourquoi", "combien",
            # Common
            "est", "sont", "a", "ont", "peut", "peuvent",
            "bonjour", "merci", "s'il vous plaît",
            # Academic
            "université", "chercheur", "chercheurs", "articles", "publications"
        ]
    }
    
    def detect(self, text: str) -> LanguageCode:
        """
        Detect language from text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (es, en, pt, fr, de, it, unknown)
        """
        if not text or len(text.strip()) < 2:
            return "unknown"
        
        text_lower = text.lower()
        
        # Count markers for each language
        scores = {}
        for lang, markers in self.LANGUAGE_MARKERS.items():
            score = sum(1 for marker in markers if marker in text_lower)
            if score > 0:
                scores[lang] = score
        
        # No markers found
        if not scores:
            # Check character patterns
            return self._detect_by_patterns(text)
        
        # Return language with highest score
        detected = max(scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"Detected language: {detected} (scores: {scores})")
        return detected
    
    def _detect_by_patterns(self, text: str) -> LanguageCode:
        """Fallback: detect by character patterns."""
        # Spanish: has ñ, inverted punctuation
        if any(char in text for char in ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']):
            return "es"
        
        # Portuguese: has ã, õ, ç
        if any(char in text for char in ['ã', 'õ', 'ç']):
            return "pt"
        
        # French: has accent grave, circumflex
        if any(char in text for char in ['à', 'è', 'ù', 'ê', 'î', 'ô', 'û', 'ë', 'ï']):
            return "fr"
        
        # Default to English for Latin chars
        if any(char.isalpha() for char in text):
            return "en"
        
        return "unknown"
    
    def get_language_name(self, code: LanguageCode) -> str:
        """Get full language name from code."""
        names = {
            "es": "Spanish",
            "en": "English",
            "pt": "Portuguese",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "unknown": "Unknown"
        }
        return names.get(code, "Unknown")
    
    def is_spanish(self, text: str) -> bool:
        """Quick check if text is Spanish."""
        return self.detect(text) == "es"
    
    def is_english(self, text: str) -> bool:
        """Quick check if text is English."""
        return self.detect(text) == "en"


# Global singleton
_language_detector = None


def get_language_detector() -> LanguageDetector:
    """Get global Language Detector instance."""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
        logger.info("Language Detector initialized")
    return _language_detector
