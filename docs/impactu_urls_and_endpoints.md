# ImpactU URLs and Endpoints Reference

## Overview

ImpactU is the public research information platform for Colombian scientific output. This document provides URL patterns for linking MARIE agent responses to the ImpactU web interface.

**Base URL:** `https://impactu.colav.co`

---

## 1. URL Patterns

### 1.1 Work/Publication URLs

**Pattern:** `https://impactu.colav.co/work/{work_id}`

**Description:** Direct link to a specific research output (article, thesis, book, etc.)

**Example:**
```
https://impactu.colav.co/work/507f1f77bcf86cd799439011
```

**Usage in RAG:**
- Include in metadata for every work document
- Provide users with direct access to full publication details
- Enable citation and export features

---

### 1.2 Person/Author URLs

**Pattern:** `https://impactu.colav.co/person/{author_id}/research/products?max=10&page=1&sort=citations_desc`

**Description:** Direct link to an author's research profile showing their publications

**Parameters:**
- `max`: Number of results per page (default: 10)
- `page`: Page number (default: 1)
- `sort`: Sorting criteria
  - `citations_desc` - Most cited first
  - `year_desc` - Most recent first
  - `year_asc` - Oldest first

**Example:**
```
https://impactu.colav.co/person/507f1f77bcf86cd799439011/research/products?max=10&page=1&sort=citations_desc
```

**Alternative Sections:**
- `/research/products` - Publications
- `/research/projects` - Research projects
- `/research/groups` - Research groups
- `/coauthors` - Collaboration network
- `/affiliations` - Institutional affiliations

**Usage in RAG:**
- Link to author profiles when mentioning researchers
- Provide access to complete publication lists
- Enable exploration of author collaborations

---

### 1.3 Affiliation/Institution URLs

**Pattern:** `https://impactu.colav.co/affiliation/{affiliation_id}`

**Description:** Direct link to an institution's research profile

**Example:**
```
https://impactu.colav.co/affiliation/https://ror.org/01cmy8y83
```

**Note:** Affiliation IDs are often ROR IDs (Research Organization Registry)

**Alternative Sections:**
- `/affiliations/{id}/research/products` - Institution publications
- `/affiliations/{id}/research/groups` - Research groups
- `/affiliations/{id}/researchers` - Researchers affiliated

**Usage in RAG:**
- Link to institutional profiles
- Provide access to institution-level metrics
- Enable exploration of institutional research output

---

### 1.4 Source/Journal URLs

**Pattern:** `https://impactu.colav.co/source/{source_id}`

**Description:** Direct link to a journal/conference/venue profile

**Example:**
```
https://impactu.colav.co/source/507f1f77bcf86cd799439011
```

**Alternative Sections:**
- `/source/{id}/works` - Publications in this venue
- `/source/{id}/metrics` - Citation metrics

**Usage in RAG:**
- Link to journal/conference profiles
- Provide venue quality indicators
- Enable exploration of venue publications

---

### 1.5 Project URLs

**Pattern:** `https://impactu.colav.co/project/{project_id}`

**Description:** Direct link to a research project profile

**Example:**
```
https://impactu.colav.co/project/507f1f77bcf86cd799439011
```

**Usage in RAG:**
- Link to project descriptions
- Show project participants and outputs
- Connect projects to publications

---

## 2. Search and Filter URLs

### 2.1 Search Results

**Pattern:** `https://impactu.colav.co/search/{entity_type}?keywords={query}&page=1`

**Entity Types:**
- `works` - Publications
- `person` - Researchers
- `affiliations` - Institutions
- `sources` - Journals/Conferences

**Example:**
```
https://impactu.colav.co/search/works?keywords=machine+learning&page=1
https://impactu.colav.co/search/person?keywords=maria+garcia&page=1
```

---

### 2.2 Advanced Filters

**Works Filter Parameters:**
- `keywords` - Search terms
- `year_start` - Starting year
- `year_end` - Ending year
- `type` - Publication type
- `subject` - Research area
- `institution` - Affiliation filter
- `sort` - Sorting criteria

**Example:**
```
https://impactu.colav.co/search/works?keywords=neural+networks&year_start=2020&year_end=2023&sort=citations_desc
```

---

## 3. Metadata Structure for RAG

### 3.1 Work Metadata with URLs

```python
metadata = {
    # Basic Info
    'work_id': 'ObjectId',
    'title': 'Publication Title',
    'doi': '10.xxxx/yyyy',
    'year': 2023,
    'type': 'article',
    'source': 'Journal Name',
    
    # ImpactU URL
    'url': f'https://impactu.colav.co/work/{work_id}',
    
    # Authors with URLs
    'authors': 'Author 1, Author 2, Author 3',
    'author_urls': [
        {
            'name': 'Author 1',
            'id': 'author_id_1',
            'url': 'https://impactu.colav.co/person/author_id_1/research/products?max=10&page=1&sort=citations_desc'
        },
        {
            'name': 'Author 2',
            'id': 'author_id_2',
            'url': 'https://impactu.colav.co/person/author_id_2/research/products?max=10&page=1&sort=citations_desc'
        }
    ],
    
    # Additional Context
    'citations_count': 42,
    'open_access': True
}
```

### 3.2 Person Metadata with URLs

```python
metadata = {
    # Basic Info
    'person_id': 'ObjectId',
    'full_name': 'María García',
    'orcid': '0000-0001-2345-6789',
    
    # ImpactU URLs
    'profile_url': f'https://impactu.colav.co/person/{person_id}/research/products?max=10&page=1&sort=citations_desc',
    'coauthors_url': f'https://impactu.colav.co/person/{person_id}/coauthors',
    'projects_url': f'https://impactu.colav.co/person/{person_id}/research/projects',
    
    # Metrics
    'products_count': 150,
    'citations_count': 2500,
    
    # Current Affiliation
    'current_affiliation': 'Universidad Nacional de Colombia',
    'affiliation_id': 'https://ror.org/01cmy8y83',
    'affiliation_url': 'https://impactu.colav.co/affiliation/https://ror.org/01cmy8y83'
}
```

---

## 4. RAG Response Template with URLs

### 4.1 Publication Response

```markdown
**Title:** [Publication Title](https://impactu.colav.co/work/507f1f77bcf86cd799439011)

**Authors:**
- [Author 1](https://impactu.colav.co/person/id1/research/products?max=10&page=1&sort=citations_desc)
- [Author 2](https://impactu.colav.co/person/id2/research/products?max=10&page=1&sort=citations_desc)

**Year:** 2023  
**Source:** Journal Name  
**Citations:** 42  
**DOI:** [10.xxxx/yyyy](https://doi.org/10.xxxx/yyyy)

**Abstract:** [Summary text...]

📊 [View in ImpactU](https://impactu.colav.co/work/507f1f77bcf86cd799439011)
```

### 4.2 Author Response

```markdown
**Name:** [María García](https://impactu.colav.co/person/507f1f77bcf86cd799439011/research/products?max=10&page=1&sort=citations_desc)

**Affiliation:** [Universidad Nacional de Colombia](https://impactu.colav.co/affiliation/https://ror.org/01cmy8y83)

**Research Areas:** Machine Learning, Computer Vision

**Metrics:**
- Publications: 150
- Citations: 2,500
- H-index: 28

**Quick Links:**
- 📄 [Publications](https://impactu.colav.co/person/507f1f77bcf86cd799439011/research/products?max=10&page=1&sort=citations_desc)
- 👥 [Collaborators](https://impactu.colav.co/person/507f1f77bcf86cd799439011/coauthors)
- 🔬 [Research Projects](https://impactu.colav.co/person/507f1f77bcf86cd799439011/research/projects)
```

---

## 5. Implementation Guidelines

### 5.1 URL Generation

```python
def generate_work_url(work_id: str) -> str:
    """Generate ImpactU URL for a work."""
    return f"https://impactu.colav.co/work/{work_id}"

def generate_author_url(author_id: str, max_results: int = 10, sort: str = "citations_desc") -> str:
    """Generate ImpactU URL for an author's publications."""
    return f"https://impactu.colav.co/person/{author_id}/research/products?max={max_results}&page=1&sort={sort}"

def generate_affiliation_url(affiliation_id: str) -> str:
    """Generate ImpactU URL for an affiliation."""
    # Note: ROR IDs need to be URL-encoded if they contain special chars
    return f"https://impactu.colav.co/affiliation/{affiliation_id}"

def generate_search_url(entity_type: str, keywords: str, **filters) -> str:
    """Generate ImpactU search URL."""
    base_url = f"https://impactu.colav.co/search/{entity_type}"
    params = {"keywords": keywords.replace(" ", "+")}
    params.update(filters)
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}?{query_string}"
```

### 5.2 Markdown Link Generation

```python
def create_markdown_link(text: str, url: str) -> str:
    """Create a markdown link."""
    return f"[{text}]({url})"

def create_author_link(author_name: str, author_id: str) -> str:
    """Create a markdown link for an author."""
    url = generate_author_url(author_id)
    return create_markdown_link(author_name, url)

def create_work_link(title: str, work_id: str) -> str:
    """Create a markdown link for a work."""
    url = generate_work_url(work_id)
    return create_markdown_link(title, url)
```

---

## 6. Important Notes

### 6.1 ID Formats

- **Work IDs:** MongoDB ObjectId (24-char hex string)
- **Person IDs:** MongoDB ObjectId or ORCID
- **Affiliation IDs:** Often ROR IDs (format: `https://ror.org/xxxxxxx`)
- **Source IDs:** MongoDB ObjectId or ISSN

### 6.2 URL Encoding

- Use URL encoding for special characters in IDs
- ROR IDs already include `https://` - don't encode the entire string
- Space → `+` or `%20` in search queries

### 6.3 Performance Considerations

- ImpactU URLs load dynamically - consider caching
- Deep links (with filters) may take longer to load
- Default pagination to 10-20 items for best performance

---

## 7. Testing URLs

### 7.1 Valid Test URLs

```bash
# Work example (if valid ID)
https://impactu.colav.co/work/507f1f77bcf86cd799439011

# Person example
https://impactu.colav.co/person/507f1f77bcf86cd799439011/research/products

# Affiliation example (Universidad Nacional de Colombia)
https://impactu.colav.co/affiliation/https://ror.org/01cmy8y83

# Search example
https://impactu.colav.co/search/works?keywords=machine+learning
```

---

## 8. Integration with RAG Pipeline

### 8.1 During Indexing

```python
# When creating metadata during indexing
metadata = {
    'work_id': str(work['_id']),
    'url': f"https://impactu.colav.co/work/{str(work['_id'])}",
    'author_urls': [
        {
            'name': author['full_name'],
            'id': str(author['id']),
            'url': f"https://impactu.colav.co/person/{str(author['id'])}/research/products?max=10&page=1&sort=citations_desc"
        }
        for author in work.get('authors', [])[:5]  # Limit to top 5
    ]
}
```

### 8.2 During Response Generation

```python
# When generating responses
def format_response_with_links(results: List[Dict]) -> str:
    """Format RAG results with ImpactU links."""
    response_parts = []
    
    for result in results:
        work_id = result['metadata']['work_id']
        title = result['metadata']['title']
        authors = result['metadata'].get('author_urls', [])
        
        # Create markdown with links
        response = f"**[{title}](https://impactu.colav.co/work/{work_id})**\n\n"
        response += "**Authors:** "
        response += ", ".join([
            f"[{author['name']}]({author['url']})"
            for author in authors
        ])
        response += f"\n\n{result['text']}\n\n"
        response += f"📊 [View in ImpactU](https://impactu.colav.co/work/{work_id})\n"
        
        response_parts.append(response)
    
    return "\n---\n\n".join(response_parts)
```

---

## 9. Future Enhancements

### 9.1 Deep Linking Features

- Direct links to specific sections (e.g., citations, references)
- Filtered views (by year, type, institution)
- Comparison views (multiple authors, institutions)

### 9.2 API Integration

- If ImpactU provides an API, integrate for real-time data
- Embed widgets/iframes for rich previews
- Dynamic thumbnail generation

---

**Document Version:** 1.0  
**Last Updated:** January 7, 2026  
**Status:** Production Reference

**Source:** Extracted from `marie_impactu/rag_indexer/indexer.py`
