"""
MARIE Tools - LangChain tool definitions.

Following LangGraph best practices for tool calling pattern.
"""

from typing import List, Dict, Any, Tuple
import logging
from langchain.tools import tool

from marie_agent.config import config
from pymongo import MongoClient
from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
def search_publications(query: str, limit: int = 10) -> Tuple[str, List[Dict]]:
    """
    Search for academic publications using semantic search.
    
    Args:
        query: Search query for publications
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (formatted_string, raw_documents)
    """
    try:
        # Connect to OpenSearch
        client = OpenSearch(
            hosts=[{"host": config.opensearch.host, "port": config.opensearch.port}],
            http_auth=(config.opensearch.username, config.opensearch.password),
            use_ssl=config.opensearch.use_ssl,
            verify_certs=config.opensearch.verify_certs
        )
        
        # Search across all indices
        index_pattern = f"{config.opensearch.index_prefix}_*"
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "title", "authors"],
                    "type": "best_fields"
                }
            },
            "size": limit
        }
        
        response = client.search(index=index_pattern, body=search_body)
        hits = response["hits"]["hits"]
        
        # Format for LLM
        if not hits:
            formatted = "No publications found matching the query."
            return formatted, []
        
        documents = []
        formatted_parts = []
        
        for i, hit in enumerate(hits, 1):
            source = hit["_source"]
            doc = {
                "id": hit["_id"],
                "score": hit["_score"],
                "title": source.get("title", "N/A"),
                "authors": source.get("authors", []),
                "year": source.get("year"),
                "source": source.get("source", "Unknown")
            }
            documents.append(doc)
            
            formatted_parts.append(
                f"{i}. {doc['title']}\n"
                f"   Authors: {', '.join(doc['authors'][:3])}...\n"
                f"   Year: {doc['year']}\n"
                f"   Relevance: {doc['score']:.2f}"
            )
        
        formatted = f"Found {len(hits)} publications:\n\n" + "\n\n".join(formatted_parts)
        
        logger.info(f"search_publications returned {len(documents)} results")
        return formatted, documents
        
    except Exception as e:
        logger.error(f"Error in search_publications: {e}")
        return f"Error searching publications: {str(e)}", []


@tool(response_format="content_and_artifact")
def resolve_entity(name: str, entity_type: str = "institution") -> Tuple[str, Dict]:
    """
    Resolve entity name to canonical form and ID.
    
    Args:
        name: Entity name to resolve
        entity_type: Type of entity (institution, author, group)
        
    Returns:
        Tuple of (formatted_string, entity_metadata)
    """
    try:
        # Connect to MongoDB
        client = MongoClient(config.mongodb.uri)
        db = client[config.mongodb.database]
        
        if entity_type == "institution":
            collection = db["affiliations"]
            query = {"names.name": {"$regex": name, "$options": "i"}}
        elif entity_type == "author":
            collection = db["person"]
            query = {"full_name": {"$regex": name, "$options": "i"}}
        else:
            return f"Unknown entity type: {entity_type}", {}
        
        # Find matching entity
        entity = collection.find_one(query)
        
        if not entity:
            return f"Entity '{name}' not found in database.", {}
        
        # Extract metadata
        metadata = {
            "id": str(entity["_id"]),
            "canonical_name": entity.get("names", [{}])[0].get("name", name),
            "type": entity_type,
            "aliases": [n.get("name") for n in entity.get("names", [])]
        }
        
        formatted = (
            f"Entity resolved:\n"
            f"Name: {metadata['canonical_name']}\n"
            f"Type: {metadata['type']}\n"
            f"ID: {metadata['id']}\n"
            f"Known aliases: {len(metadata['aliases'])}"
        )
        
        logger.info(f"resolve_entity: {name} -> {metadata['id']}")
        return formatted, metadata
        
    except Exception as e:
        logger.error(f"Error in resolve_entity: {e}")
        return f"Error resolving entity: {str(e)}", {}


@tool(response_format="content_and_artifact")
def calculate_metrics(entity_id: str, years: int = 5) -> Tuple[str, Dict]:
    """
    Calculate publication metrics for an entity.
    
    Args:
        entity_id: Entity ID to calculate metrics for
        years: Number of years to include in metrics
        
    Returns:
        Tuple of (formatted_string, metrics_dict)
    """
    try:
        # Connect to MongoDB
        client = MongoClient(config.mongodb.uri)
        db = client[config.mongodb.database]
        
        # Calculate metrics via aggregation
        pipeline = [
            {"$match": {"authors.id": entity_id}},
            {"$group": {
                "_id": None,
                "total_papers": {"$sum": 1},
                "total_citations": {"$sum": "$citations_count"},
                "years": {"$addToSet": "$year"}
            }}
        ]
        
        result = list(db["works"].aggregate(pipeline))
        
        if not result:
            return "No publications found for this entity.", {}
        
        metrics = result[0]
        metrics_dict = {
            "entity_id": entity_id,
            "total_papers": metrics.get("total_papers", 0),
            "total_citations": metrics.get("total_citations", 0),
            "years_active": len(metrics.get("years", [])),
            "h_index": 0  # TODO: Calculate actual h-index
        }
        
        formatted = (
            f"Publication Metrics:\n"
            f"Total Papers: {metrics_dict['total_papers']}\n"
            f"Total Citations: {metrics_dict['total_citations']}\n"
            f"Years Active: {metrics_dict['years_active']}\n"
            f"H-Index: {metrics_dict['h_index']}"
        )
        
        logger.info(f"calculate_metrics: {entity_id} -> {metrics_dict['total_papers']} papers")
        return formatted, metrics_dict
        
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        return f"Error calculating metrics: {str(e)}", {}


@tool
def rank_researchers(entity_id: str, limit: int = 3, metric: str = "papers") -> str:
    """
    Rank top researchers for an institution.
    
    Args:
        entity_id: Institution ID
        limit: Number of top researchers to return
        metric: Ranking metric (papers, citations, h_index)
        
    Returns:
        Formatted string with top researchers
    """
    try:
        # Connect to MongoDB
        client = MongoClient(config.mongodb.uri)
        db = client[config.mongodb.database]
        
        # Aggregate top researchers
        pipeline = [
            {"$match": {"authors.affiliations.id": entity_id}},
            {"$unwind": "$authors"},
            {"$match": {"authors.affiliations.id": entity_id}},
            {"$group": {
                "_id": "$authors.id",
                "name": {"$first": "$authors.full_name"},
                "papers": {"$sum": 1},
                "citations": {"$sum": "$citations_count"}
            }},
            {"$sort": {metric: -1}},
            {"$limit": limit}
        ]
        
        results = list(db["works"].aggregate(pipeline))
        
        if not results:
            return "No researchers found for this institution."
        
        lines = [f"Top {limit} Researchers (by {metric}):\n"]
        for i, researcher in enumerate(results, 1):
            lines.append(
                f"{i}. {researcher['name']}\n"
                f"   Papers: {researcher['papers']}\n"
                f"   Citations: {researcher['citations']}"
            )
        
        formatted = "\n\n".join(lines)
        logger.info(f"rank_researchers: {entity_id} -> {len(results)} researchers")
        return formatted
        
    except Exception as e:
        logger.error(f"Error in rank_researchers: {e}")
        return f"Error ranking researchers: {str(e)}"


# Tool registry for easy access
MARIE_TOOLS = [
    search_publications,
    resolve_entity,
    calculate_metrics,
    rank_researchers
]


def get_tools() -> List:
    """Get all available MARIE tools."""
    return MARIE_TOOLS
