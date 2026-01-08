"""
Colombian Entity Detector - Uses ONLY OpenSearch.

Detects Colombian entities (institutions, authors, research groups) to determine
if RAG should be activated. Uses specialized Colombian indices created by RAG indexer.

The RAG indexer creates separate indices for Colombian data:
- impactu_marie_agent_affiliations_colombia
- impactu_marie_agent_person_colombia
- impactu_marie_agent_works_colombia

These indices contain ONLY Colombian entities, making searches faster and more accurate.
"""

from typing import Dict, Any, List, Optional
import logging
from opensearchpy import OpenSearch

from marie_agent.config import config

logger = logging.getLogger(__name__)


class ColombianDetector:
    """
    Detects Colombian entities using OpenSearch indices.
    
    Uses specialized Colombian indices (*_colombia) if available,
    otherwise falls back to main indices with country filters.
    """
    
    def __init__(self):
        """Initialize OpenSearch connection."""
        self.client = OpenSearch(
            hosts=[config.opensearch.url],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        self.index_prefix = config.opensearch.index_prefix
        
        # Check if Colombian indices exist
        self.has_colombia_indices = self._check_colombia_indices()
        
        if self.has_colombia_indices:
            logger.info("✅ Using specialized Colombian indices (*_colombia)")
        else:
            logger.info("⚠️  Colombian indices not found, will use main indices with filters")
    
    def _check_colombia_indices(self) -> bool:
        """Check if specialized Colombian indices exist."""
        try:
            colombia_indices = [
                f"{self.index_prefix}_affiliations_colombia",
                f"{self.index_prefix}_person_colombia",
                f"{self.index_prefix}_works_colombia"
            ]
            
            for index_name in colombia_indices:
                if self.client.indices.exists(index=index_name):
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking Colombian indices: {e}")
            return False
    
    def _get_index_name(self, collection: str) -> str:
        """
        Get index name for collection.
        
        Returns Colombian index if available, otherwise main index.
        """
        if self.has_colombia_indices:
            return f"{self.index_prefix}_{collection}_colombia"
        else:
            return f"{self.index_prefix}_{collection}"
    
    def detect_colombian_context(self, query: str) -> Dict[str, Any]:
        """
        Detect if query mentions Colombian entities.
        
        Searches OpenSearch Colombian indices for:
        - Colombian institutions (affiliations)
        - Colombian authors (persons)
        - Colombian works (publications)
        
        Args:
            query: User query text
            
        Returns:
            {
                "has_colombian_entities": bool,
                "detected_entities": List[Dict],
                "confidence": float,
                "using_colombia_indices": bool
            }
        """
        logger.info(f"Detecting Colombian entities in: {query[:100]}...")
        
        detected = []
        
        try:
            # Search in all three Colombian indices/collections
            # Order: affiliations -> persons -> works
            
            # 1. Search affiliations
            affiliation_result = self._search_affiliation(query)
            if affiliation_result:
                detected.append(affiliation_result)
                logger.info(f"✅ Found Colombian affiliation: {affiliation_result['name']}")
            
            # 2. Search persons
            person_result = self._search_person(query)
            if person_result:
                detected.append(person_result)
                logger.info(f"✅ Found Colombian person: {person_result['name']}")
            
            # 3. Search works
            work_result = self._search_work(query)
            if work_result:
                detected.append(work_result)
                logger.info(f"✅ Found Colombian work: {work_result.get('title', '')[:50]}")
            
        except Exception as e:
            logger.error(f"Error detecting Colombian entities: {e}")
        
        has_colombian = len(detected) > 0
        
        if has_colombian:
            logger.info(f"🇨🇴 Found {len(detected)} Colombian entities")
        else:
            logger.info("No Colombian entities found")
        
        return {
            "has_colombian_entities": has_colombian,
            "detected_entities": detected,
            "confidence": 0.9 if has_colombian else 0.0,
            "using_colombia_indices": self.has_colombia_indices
        }
    
    def _search_affiliation(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Search for Colombian affiliations (institutions).
        
        Uses Colombian index if available, otherwise filters by country.
        """
        try:
            index_name = self._get_index_name("affiliations")
            
            # Base query
            search_query = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["name^3", "abbreviations^2", "aliases"],
                        "fuzziness": "AUTO"
                    }
                },
                "size": 5
            }
            
            # If using main index, add country filter
            if not self.has_colombia_indices:
                search_query["query"] = {
                    "bool": {
                        "must": search_query["query"],
                        "filter": {
                            "nested": {
                                "path": "addresses",
                                "query": {
                                    "term": {"addresses.country_code": "CO"}
                                }
                            }
                        }
                    }
                }
            
            response = self.client.search(
                index=index_name,
                body=search_query
            )
            
            if response['hits']['total']['value'] > 0:
                hit = response['hits']['hits'][0]
                source = hit['_source']
                
                return {
                    "type": "affiliation",
                    "id": hit['_id'],
                    "name": source.get('name', ''),
                    "score": hit['_score'],
                    "country": "Colombia"
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching affiliations (index may not exist): {e}")
            return None
    
    def _search_person(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Search for Colombian persons (authors/researchers).
        
        Uses Colombian index if available, otherwise filters by country.
        """
        try:
            index_name = self._get_index_name("person")
            
            # Base query
            search_query = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["full_name^3", "first_names^2", "last_names^2"],
                        "fuzziness": "AUTO"
                    }
                },
                "size": 5
            }
            
            # If using main index, add country filter
            if not self.has_colombia_indices:
                search_query["query"] = {
                    "bool": {
                        "must": search_query["query"],
                        "filter": {
                            "nested": {
                                "path": "affiliations",
                                "query": {
                                    "nested": {
                                        "path": "affiliations.addresses",
                                        "query": {
                                            "term": {"affiliations.addresses.country_code": "CO"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            
            response = self.client.search(
                index=index_name,
                body=search_query
            )
            
            if response['hits']['total']['value'] > 0:
                hit = response['hits']['hits'][0]
                source = hit['_source']
                
                return {
                    "type": "person",
                    "id": hit['_id'],
                    "name": source.get('full_name', ''),
                    "score": hit['_score'],
                    "country": "Colombia"
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching persons (index may not exist): {e}")
            return None
    
    def _search_work(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Search for Colombian works (publications).
        
        Uses Colombian index if available, otherwise filters by country.
        """
        try:
            index_name = self._get_index_name("works")
            
            # Base query
            search_query = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title^3", "abstract^2", "keywords"],
                        "fuzziness": "AUTO"
                    }
                },
                "size": 5
            }
            
            # If using main index, add country filter
            if not self.has_colombia_indices:
                search_query["query"] = {
                    "bool": {
                        "must": search_query["query"],
                        "filter": {
                            "nested": {
                                "path": "authors",
                                "query": {
                                    "nested": {
                                        "path": "authors.affiliations",
                                        "query": {
                                            "nested": {
                                                "path": "authors.affiliations.addresses",
                                                "query": {
                                                    "term": {"authors.affiliations.addresses.country_code": "CO"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            
            response = self.client.search(
                index=index_name,
                body=search_query
            )
            
            if response['hits']['total']['value'] > 0:
                hit = response['hits']['hits'][0]
                source = hit['_source']
                
                return {
                    "type": "work",
                    "id": hit['_id'],
                    "title": source.get('title', ''),
                    "score": hit['_score'],
                    "country": "Colombia"
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching works (index may not exist): {e}")
            return None


# Singleton instance
_detector_instance = None


def get_colombian_detector() -> ColombianDetector:
    """Get singleton instance of Colombian detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ColombianDetector()
    return _detector_instance
