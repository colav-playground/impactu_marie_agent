"""
Text processing utilities for KAHI documents
"""
from typing import Dict, List, Any, Optional


def inverted_index_to_text(inverted_index: Dict[str, List[int]]) -> str:
    """
    Convert an inverted index to text.
    
    Args:
        inverted_index: Dictionary mapping words to their positions
        
    Returns:
        Reconstructed text string
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    
    # Create list of tuples (position, word)
    word_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list) and positions:
            for pos in positions:
                word_positions.append((pos, word))
    
    # Sort by position and reconstruct text
    word_positions.sort(key=lambda x: x[0])
    text = ' '.join([word for _, word in word_positions])
    
    return text


def extract_titles(work: Dict[str, Any]) -> str:
    """
    Extract titles from work document.
    
    Args:
        work: Work document from MongoDB
        
    Returns:
        Combined title text
    """
    titles = []
    if work.get('titles'):
        for title_obj in work['titles']:
            if title_obj.get('title'):
                titles.append(title_obj['title'])
    
    return " | ".join(titles)


def extract_abstract(work: Dict[str, Any]) -> str:
    """
    Extract abstract from work document.
    Handles both string and inverted index formats.
    Combines all abstracts (different languages) into one text.
    
    Args:
        work: Work document from MongoDB
        
    Returns:
        Combined abstract text from all languages
    """
    if not work.get('abstracts'):
        return ""
    
    abstracts = []
    
    for abstract_obj in work['abstracts']:
        abstract_data = abstract_obj.get('abstract')
        lang = abstract_obj.get('lang', 'unknown')
        
        # If it's a string directly
        if isinstance(abstract_data, str) and len(abstract_data) > 50:
            abstracts.append(f"[{lang}] {abstract_data}")
        
        # If it's an inverted index (dictionary)
        elif isinstance(abstract_data, dict):
            abstract_text = inverted_index_to_text(abstract_data)
            if abstract_text and len(abstract_text) > 50:
                abstracts.append(f"[{lang}] {abstract_text}")
    
    return "\n\n".join(abstracts)


def extract_keywords(work: Dict[str, Any]) -> List[str]:
    """
    Extract keywords from work document.
    
    Args:
        work: Work document from MongoDB
        
    Returns:
        List of keywords
    """
    keywords = []
    
    # Direct keywords
    if work.get('keywords') and isinstance(work['keywords'], list):
        for kw in work['keywords']:
            if isinstance(kw, dict) and kw.get('keyword'):
                keywords.append(kw['keyword'])
            elif isinstance(kw, str):
                keywords.append(kw)
    
    # Subjects
    if work.get('subjects') and isinstance(work['subjects'], list):
        for subj in work['subjects']:
            if isinstance(subj, dict) and subj.get('subject'):
                keywords.append(subj['subject'])
            elif isinstance(subj, str):
                keywords.append(subj)
    
    return list(set(keywords))  # Remove duplicates


def extract_authors(work: Dict[str, Any], limit: int = 5) -> List[str]:
    """
    Extract author names from work document.
    
    Args:
        work: Work document from MongoDB
        limit: Maximum number of authors to return
        
    Returns:
        List of author names
    """
    if not work.get('authors'):
        return []
    
    authors = []
    for author in work['authors'][:limit]:
        if author.get('full_name'):
            authors.append(author['full_name'])
    
    return authors


def extract_source_name(work: Dict[str, Any]) -> str:
    """
    Extract source/journal name from work document.
    
    Args:
        work: Work document from MongoDB
        
    Returns:
        Source name or empty string
    """
    if work.get('source') and isinstance(work['source'], dict):
        return work['source'].get('name', '')
    return ""


def extract_work_text(work: Dict[str, Any]) -> str:
    """
    Extract all relevant text from a work document for indexing.
    Includes enriched information about authors and their affiliations.
    
    Args:
        work: Work document from MongoDB
        
    Returns:
        Combined text for embedding
    """
    text_parts = []
    
    # Title
    titles = extract_titles(work)
    if titles:
        text_parts.append(f"Title: {titles}")
    
    # Abstract
    abstract = extract_abstract(work)
    if abstract:
        text_parts.append(f"Abstract: {abstract}")
    
    # Authors with affiliations (ENRICHED)
    if work.get('authors') and isinstance(work['authors'], list):
        author_texts = []
        for author in work['authors'][:10]:  # Limit to 10 authors
            author_name = author.get('full_name', '')
            if author_name:
                # Add author affiliations
                if author.get('affiliations') and isinstance(author['affiliations'], list):
                    aff_names = []
                    for aff in author['affiliations'][:2]:  # Top 2 affiliations per author
                        aff_name = aff.get('name', '')
                        if aff_name:
                            aff_names.append(aff_name)
                    if aff_names:
                        author_texts.append(f"{author_name} ({', '.join(aff_names)})")
                    else:
                        author_texts.append(author_name)
                else:
                    author_texts.append(author_name)
        
        if author_texts:
            text_parts.append(f"Authors: {'; '.join(author_texts)}")
    
    # Year
    if work.get('year_published'):
        text_parts.append(f"Year: {work['year_published']}")
    
    # Keywords
    keywords = extract_keywords(work)
    if keywords:
        text_parts.append(f"Keywords: {', '.join(keywords[:10])}")  # Limit to 10
    
    # Source with type (ENRICHED)
    if work.get('source') and isinstance(work['source'], dict):
        source_name = work['source'].get('name', '')
        if source_name:
            # Add source type if available
            source_type = ''
            if work.get('bibliographic_info') and isinstance(work['bibliographic_info'], dict):
                source_type = work['bibliographic_info'].get('type', '')
            
            if source_type:
                text_parts.append(f"Published in: {source_name} ({source_type})")
            else:
                text_parts.append(f"Published in: {source_name}")
    
    # Primary topic
    if work.get('primary_topic') and isinstance(work['primary_topic'], dict):
        topic_name = work['primary_topic'].get('name', '')
        if topic_name:
            text_parts.append(f"Main topic: {topic_name}")
    
    # Citation count (if significant)
    if work.get('citations_count') and isinstance(work['citations_count'], list):
        for citation in work['citations_count']:
            if citation.get('source') and citation.get('count'):
                count = citation['count']
                if count > 0:
                    text_parts.append(f"Citations: {count}")
                    break
    
    return "\n".join(text_parts)


def create_work_metadata(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create enriched metadata for a work document with relational IDs.
    Includes ImpactU URLs and cross-references.
    
    Args:
        work: Work document from MongoDB
        
    Returns:
        Metadata dictionary with relational information
    """
    work_id = str(work.get('_id', ''))
    metadata = {
        'work_id': work_id,
        'doi': work.get('doi', ''),
        'year': work.get('year_published', 0),
        'url': f'https://impactu.colav.co/work/{work_id}' if work_id else '',
    }
    
    # Add updated information with sources
    if work.get('updated'):
        sources_info = []
        for update in work['updated']:
            if update.get('source') and update.get('time'):
                sources_info.append({
                    'source': update['source'],
                    'time': update['time']
                })
        if sources_info:
            metadata['updated'] = sources_info
    
    # Add main title
    titles = extract_titles(work)
    if titles:
        metadata['title'] = titles.split(' | ')[0]  # First title
    
    # ENRICHED: Author IDs and names for relational queries
    if work.get('authors') and isinstance(work['authors'], list):
        author_ids = []
        author_names = []
        author_affiliations_ids = []
        
        for author in work['authors'][:10]:  # Limit to 10
            # Collect author IDs
            if author.get('id'):
                author_ids.append(author['id'])
            
            # Collect author names
            if author.get('full_name'):
                author_names.append(author['full_name'])
            
            # Collect affiliation IDs from authors
            if author.get('affiliations') and isinstance(author['affiliations'], list):
                for aff in author['affiliations']:
                    if aff.get('id'):
                        author_affiliations_ids.append(aff['id'])
        
        if author_ids:
            metadata['author_ids'] = author_ids
        if author_names:
            metadata['authors'] = ', '.join(author_names[:5])  # First 5 for display
        if author_affiliations_ids:
            metadata['author_affiliation_ids'] = list(set(author_affiliations_ids))  # Unique
    
    # ENRICHED: Source/Journal ID and name
    if work.get('source') and isinstance(work['source'], dict):
        if work['source'].get('id'):
            metadata['source_id'] = str(work['source']['id'])
        if work['source'].get('name'):
            metadata['source_name'] = work['source']['name']
    
    # ENRICHED: Work type
    if work.get('types') and isinstance(work['types'], list):
        for type_obj in work['types']:
            if type_obj.get('type'):
                metadata['work_type'] = type_obj['type']
                break
    
    # ENRICHED: Topics (multiple)
    if work.get('topics') and isinstance(work['topics'], list):
        topic_names = []
        for topic in work['topics'][:5]:  # Limit to top 5
            if topic.get('display_name'):
                topic_names.append(topic['display_name'])
        if topic_names:
            metadata['topics'] = topic_names
    
    # Primary topic (single)
    if work.get('primary_topic') and isinstance(work['primary_topic'], dict):
        if work['primary_topic'].get('display_name'):
            metadata['primary_topic'] = work['primary_topic']['display_name']
    
    # Add citations count
    if work.get('citations_count') and isinstance(work['citations_count'], list):
        for citation in work['citations_count']:
            if citation.get('source') and citation.get('count'):
                metadata['citations_count'] = citation['count']
                break
    
    # Bibliographic info
    if work.get('bibliographic_info') and isinstance(work['bibliographic_info'], dict):
        bib_info = work['bibliographic_info']
        if bib_info.get('volume'):
            metadata['volume'] = bib_info['volume']
        if bib_info.get('issue'):
            metadata['issue'] = bib_info['issue']
        if bib_info.get('start_page'):
            metadata['start_page'] = bib_info['start_page']
        if bib_info.get('end_page'):
            metadata['end_page'] = bib_info['end_page']
    
    # Open access status
    if work.get('open_access') and isinstance(work['open_access'], dict):
        if work['open_access'].get('open_access_status'):
            metadata['open_access_status'] = work['open_access']['open_access_status']
        if work['open_access'].get('is_open_access') is not None:
            metadata['is_open_access'] = work['open_access']['is_open_access']
    
    # Author count
    if work.get('author_count'):
        metadata['author_count'] = work['author_count']
    
    # References count
    if work.get('references_count'):
        metadata['references_count'] = work['references_count']
    
    # Date published (timestamp)
    if work.get('date_published'):
        metadata['date_published'] = work['date_published']
    
    return metadata


def should_index_work(work: Dict[str, Any], min_text_length: int = 50) -> bool:
    """
    Determine if a work should be indexed.
    
    Args:
        work: Work document from MongoDB
        min_text_length: Minimum text length required
        
    Returns:
        True if work should be indexed
    """
    text = extract_work_text(work)
    return len(text.strip()) >= min_text_length


# ============================================================================
# PERSON PROCESSING
# ============================================================================

def extract_person_text(person: Dict[str, Any]) -> str:
    """
    Extract all relevant text from a person document for indexing.
    
    Args:
        person: Person document from MongoDB
        
    Returns:
        Combined text for embedding
    """
    text_parts = []
    
    # Full name
    if person.get('full_name'):
        text_parts.append(f"Name: {person['full_name']}")
    
    # Aliases
    if person.get('aliases') and isinstance(person['aliases'], list):
        aliases = [a for a in person['aliases'] if isinstance(a, str)]
        if aliases:
            text_parts.append(f"Also known as: {', '.join(aliases)}")
    
    # Affiliations
    if person.get('affiliations') and isinstance(person['affiliations'], list):
        affiliations_text = []
        for aff in person['affiliations'][:10]:  # Limit to 10
            name = aff.get('name', '')
            types = aff.get('types', [])
            type_str = types[0].get('type', '') if types else ''
            if name:
                if type_str:
                    affiliations_text.append(f"{name} ({type_str})")
                else:
                    affiliations_text.append(name)
        if affiliations_text:
            text_parts.append(f"Affiliations: {', '.join(affiliations_text)}")
    
    # Keywords/subjects
    if person.get('keywords') and isinstance(person['keywords'], list):
        keywords = []
        for kw in person['keywords'][:10]:
            if isinstance(kw, dict):
                keyword_val = kw.get('keyword', '')
                if keyword_val and isinstance(keyword_val, str):
                    keywords.append(keyword_val)
            elif isinstance(kw, str):
                keywords.append(kw)
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")
    
    if person.get('subjects') and isinstance(person['subjects'], list):
        subjects = []
        for subj in person['subjects'][:10]:
            if isinstance(subj, dict):
                subject_val = subj.get('subject', '')
                if subject_val and isinstance(subject_val, str):
                    subjects.append(subject_val)
            elif isinstance(subj, str):
                subjects.append(subj)
        if subjects:
            text_parts.append(f"Research areas: {', '.join(subjects)}")
    
    # Ranking
    if person.get('ranking') and isinstance(person['ranking'], list):
        for rank in person['ranking'][:3]:
            source = rank.get('source', '')
            rank_val = rank.get('rank', '')
            if source and rank_val:
                text_parts.append(f"Ranking {source}: {rank_val}")
    
    # Sex/Gender (allowed)
    if person.get('sex'):
        text_parts.append(f"Gender: {person['sex']}")
    
    # NOTE: Sensitive personal data excluded for privacy:
    # - birthplace (lugar de nacimiento)
    # - birthdate (fecha de nacimiento)
    # - marital_status (estado civil)
    # - identification documents (cédulas, pasaportes) - handled in external_ids
    
    return "\n".join(text_parts)


def create_person_metadata(person: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create enriched metadata for a person document with relational IDs.
    
    Args:
        person: Person document from MongoDB
        
    Returns:
        Metadata dictionary with relational information
    """
    person_id = str(person.get('_id', ''))
    metadata = {
        'person_id': person_id,
        'url': f'https://impactu.colav.co/person/{person_id}/research/products?max=10&page=1&sort=citations_desc' if person_id else '',
    }
    
    # Add updated information with sources
    if person.get('updated'):
        sources_info = []
        for update in person['updated']:
            if update.get('source') and update.get('time'):
                sources_info.append({
                    'source': update['source'],
                    'time': update['time']
                })
        if sources_info:
            metadata['updated'] = sources_info
    
    # Full name
    if person.get('full_name'):
        metadata['full_name'] = person['full_name']
    
    # Initials
    if person.get('initials'):
        metadata['initials'] = person['initials']
    
    # ENRICHED: Affiliation IDs for relational queries
    if person.get('affiliations') and isinstance(person['affiliations'], list):
        affiliation_ids = []
        affiliation_names = []
        
        for aff in person['affiliations'][:5]:  # Top 5 affiliations
            if aff.get('id'):
                affiliation_ids.append(aff['id'])
            if aff.get('name'):
                affiliation_names.append(aff['name'])
        
        if affiliation_ids:
            metadata['affiliation_ids'] = affiliation_ids
        if affiliation_names:
            metadata['affiliations'] = ', '.join(affiliation_names[:3])  # Top 3 for display
    
    # Main affiliation
    if person.get('affiliations') and isinstance(person['affiliations'], list) and len(person['affiliations']) > 0:
        main_aff = person['affiliations'][0]
        if main_aff.get('name'):
            metadata['affiliation'] = main_aff['name']
            metadata['affiliation_id'] = main_aff.get('id', '')
    
    # External IDs (excluding personal identification documents)
    excluded_sources = {
        'cédula de ciudadanía', 'cedula de ciudadania', 'cedula',
        'pasaporte', 'passport',
        'dni', 'nit', 'rut',
        'tarjeta de identidad', 'documento de identidad'
    }
    
    # Academic/research ID sources we want to keep
    academic_sources = {
        'orcid', 'scopus author id', 'google scholar', 'researcherid',
        'scienti', 'scholar', 'openalex', 'lens', 'semantic scholar'
    }
    
    if person.get('external_ids'):
        # Collect ORCID specifically
        for ext_id in person['external_ids']:
            source = ext_id.get('source', '').lower()
            if source == 'orcid' and ext_id.get('id'):
                ext_id_value = ext_id['id']
                if isinstance(ext_id_value, str):
                    metadata['orcid'] = ext_id_value
                    break
        
        # Collect all academic/research IDs (excluding personal docs)
        external_ids_list = []
        for ext_id in person['external_ids']:
            source = ext_id.get('source', '').lower()
            source_original = ext_id.get('source', '')
            
            # Include if: academic source OR (not excluded AND has string ID)
            if source in academic_sources or source not in excluded_sources:
                ext_id_value = ext_id.get('id')
                # Only include if it's a string (not dict like COD_RH)
                if isinstance(ext_id_value, str) and ext_id_value:
                    external_ids_list.append({
                        'source': source_original,
                        'id': ext_id_value
                    })
        
        if external_ids_list:
            metadata['external_ids'] = external_ids_list[:5]  # Limit to 5
    
    return metadata


def should_index_person(person: Dict[str, Any], min_text_length: int = 50) -> bool:
    """
    Determine if a person should be indexed.
    
    Args:
        person: Person document from MongoDB
        min_text_length: Minimum text length required
        
    Returns:
        True if person should be indexed
    """
    text = extract_person_text(person)
    return len(text.strip()) >= min_text_length


# ============================================================================
# AFFILIATIONS PROCESSING
# ============================================================================

def extract_affiliation_text(affiliation: Dict[str, Any]) -> str:
    """
    Extract all relevant text from an affiliation document for indexing.
    
    Args:
        affiliation: Affiliation document from MongoDB
        
    Returns:
        Combined text for embedding
    """
    text_parts = []
    
    # Names in different languages
    if affiliation.get('names') and isinstance(affiliation['names'], list):
        # Group by language
        names_by_lang = {}
        for name_obj in affiliation['names']:
            lang = name_obj.get('lang', 'unknown')
            name = name_obj.get('name', '')
            if name:
                if lang not in names_by_lang:
                    names_by_lang[lang] = []
                names_by_lang[lang].append(name)
        
        # Add primary name (English or first available)
        if 'en' in names_by_lang:
            text_parts.append(f"Name: {names_by_lang['en'][0]}")
        elif names_by_lang:
            first_lang = list(names_by_lang.keys())[0]
            text_parts.append(f"Name: {names_by_lang[first_lang][0]}")
        
        # Add other languages
        for lang, names in list(names_by_lang.items())[:5]:  # Limit to 5 languages
            if lang != 'en' and names:
                text_parts.append(f"Name [{lang}]: {names[0]}")
    
    # Types
    if affiliation.get('types') and isinstance(affiliation['types'], list):
        types = [t.get('type', '') for t in affiliation['types'] if t.get('type')]
        if types:
            text_parts.append(f"Type: {', '.join(set(types[:5]))}")
    
    # Addresses
    if affiliation.get('addresses') and isinstance(affiliation['addresses'], list):
        for addr in affiliation['addresses'][:3]:  # Limit to 3
            addr_parts = []
            if addr.get('city'):
                addr_parts.append(addr['city'])
            if addr.get('state'):
                addr_parts.append(addr['state'])
            if addr.get('country'):
                addr_parts.append(addr['country'])
            if addr_parts:
                text_parts.append(f"Location: {', '.join(addr_parts)}")
    
    # Abbreviations/acronyms
    if affiliation.get('abbreviations') and isinstance(affiliation['abbreviations'], list):
        abbrevs = [a for a in affiliation['abbreviations'] if isinstance(a, str)]
        if abbrevs:
            text_parts.append(f"Abbreviations: {', '.join(abbrevs[:5])}")
    
    # External URLs/websites
    if affiliation.get('external_urls') and isinstance(affiliation['external_urls'], list):
        for url_obj in affiliation['external_urls'][:2]:
            if isinstance(url_obj, dict) and url_obj.get('url'):
                text_parts.append(f"Website: {url_obj['url']}")
    
    return "\n".join(text_parts)


def create_affiliation_metadata(affiliation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata for an affiliation document.
    
    Args:
        affiliation: Affiliation document from MongoDB
        
    Returns:
        Metadata dictionary
    """
    affiliation_id = str(affiliation.get('_id', ''))
    metadata = {
        'affiliation_id': affiliation_id,
        'url': f'https://impactu.colav.co/affiliation/{affiliation_id}/research/products?max=10&page=1&sort=citations_desc' if affiliation_id else '',
    }
    
    # Add updated information with sources
    if affiliation.get('updated'):
        sources_info = []
        for update in affiliation['updated']:
            if update.get('source') and update.get('time'):
                sources_info.append({
                    'source': update['source'],
                    'time': update['time']
                })
        if sources_info:
            metadata['updated'] = sources_info
    
    # Primary name (English preferred)
    if affiliation.get('names') and isinstance(affiliation['names'], list):
        for name_obj in affiliation['names']:
            if name_obj.get('lang') == 'en' and name_obj.get('name'):
                metadata['name'] = name_obj['name']
                break
        # Fallback to first name
        if 'name' not in metadata and affiliation['names']:
            first_name = affiliation['names'][0].get('name')
            if first_name:
                metadata['name'] = first_name
    
    # Types
    if affiliation.get('types') and isinstance(affiliation['types'], list):
        types = [t.get('type', '') for t in affiliation['types'] if t.get('type')]
        if types:
            metadata['types'] = list(set(types[:3]))
    
    # Country
    if affiliation.get('addresses') and isinstance(affiliation['addresses'], list):
        for addr in affiliation['addresses']:
            if addr.get('country'):
                metadata['country'] = addr['country']
                if addr.get('city'):
                    metadata['city'] = addr['city']
                break
    
    # External IDs
    if affiliation.get('external_ids'):
        for ext_id in affiliation['external_ids']:
            source = ext_id.get('source', '').lower()
            if source in ['ror', 'grid', 'wikidata'] and ext_id.get('id'):
                metadata[f'{source}_id'] = ext_id['id']
    
    return metadata


def should_index_affiliation(affiliation: Dict[str, Any], min_text_length: int = 30) -> bool:
    """
    Determine if an affiliation should be indexed.
    
    Args:
        affiliation: Affiliation document from MongoDB
        min_text_length: Minimum text length required
        
    Returns:
        True if affiliation should be indexed
    """
    text = extract_affiliation_text(affiliation)
    return len(text.strip()) >= min_text_length
