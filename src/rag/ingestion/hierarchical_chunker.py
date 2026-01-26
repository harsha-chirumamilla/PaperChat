import re
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HierarchicalChunker:
    """Creates hierarchical chunks preserving paper structure."""
    
    def __init__(self):
        # Common section patterns in academic papers
        self.section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS sections
        ]
    
    def identify_sections(self, text: str) -> List[Dict]:
        """
        Identify paper sections (Abstract, Introduction, Methods, etc.)
        
        Returns:
            List of {title, start_line, end_line, level} for each section
        """
        lines = text.split('\n')
        sections = []
        
        for i, line in enumerate(lines):
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    title = match.group(1).strip()
                    
                    # Common academic sections
                    if any(keyword in title.lower() for keyword in [
                        'abstract', 'introduction', 'background', 'method',
                        'result', 'discussion', 'conclusion', 'reference',
                        'related work', 'experiment', 'evaluation', 'future'
                    ]):
                        sections.append({
                            'title': title,
                            'start_line': i,
                            'level': 1 if len(title) > 20 else 2
                        })
        
        # Calculate end positions
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                section['end_line'] = sections[i + 1]['start_line']
            else:
                section['end_line'] = len(lines)
        
        return sections
    
    def create_hierarchical_chunks(
        self,
        text: str,
        metadata: Dict,
        base_chunk_size: int = 800
    ) -> List[Document]:
        """
        Create chunks that respect section boundaries.
        
        Args:
            text: Full paper text
            metadata: Paper metadata
            base_chunk_size: Target size for sub-section chunks
            
        Returns:
            List of Document chunks with hierarchical metadata
        """
        sections = self.identify_sections(text)
        chunks = []
        lines = text.split('\n')
        
        if not sections:
            # Fallback to regular chunking
            return self._fallback_chunking(text, metadata, base_chunk_size)
        
        for section in sections:
            section_text = '\n'.join(
                lines[section['start_line']:section['end_line']]
            )
            
            # Create section-level chunk
            section_meta = {
                **metadata,
                'section': section['title'],
                'section_level': section['level'],
                'chunk_type': 'section'
            }
            
            # If section is large, split into sub-chunks
            if len(section_text) > base_chunk_size * 1.5:
                sub_chunks = self._split_section(
                    section_text,
                    section_meta,
                    base_chunk_size
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(
                    page_content=section_text,
                    metadata=section_meta
                ))
        
        return chunks
    
    def _split_section(
        self,
        section_text: str,
        section_meta: Dict,
        chunk_size: int
    ) -> List[Document]:
        """Split large section into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=['\n\n', '\n', '. ', ' ']
        )
        
        texts = splitter.split_text(section_text)
        chunks = []
        
        for i, text in enumerate(texts):
            chunk_meta = {
                **section_meta,
                'sub_chunk': i,
                'chunk_type': 'sub_section'
            }
            chunks.append(Document(
                page_content=text,
                metadata=chunk_meta
            ))
        
        return chunks
    
    def _fallback_chunking(
        self,
        text: str,
        metadata: Dict,
        chunk_size: int
    ) -> List[Document]:
        """Standard chunking when no sections detected."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        
        texts = splitter.split_text(text)
        return [
            Document(
                page_content=t,
                metadata={**metadata, 'chunk_id': i, 'chunk_type': 'standard'}
            )
            for i, t in enumerate(texts)
        ]
