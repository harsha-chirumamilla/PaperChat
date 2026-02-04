import os
import logging
import arxiv
from typing import List, Dict
from src.common.config import Config

logger = logging.getLogger(__name__)


class ArxivLoader:
    def __init__(self, download_dir: str = Config.PAPERS_DIR):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.client = arxiv.Client()

    def fetch_papers(self, query: str = "cat:cs.AI", max_results: int = 5, skip_papers: int = 0) -> List[Dict]:
        """
        Fetches papers from arXiv matching the query.
        Returns a list of metadata dictionaries.
        Downloads the PDF if not already present.
        
        Args:
            query: arXiv query string
            max_results: Maximum number of papers to fetch
            skip_papers: Number of papers to skip (for pagination)
        """
        # Fetch more results to account for skipping
        total_to_fetch = max_results + skip_papers
        
        search = arxiv.Search(
            query=query,
            max_results=total_to_fetch,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers_metadata = []
        skipped = 0
        
        for result in self.client.results(search):
            # Skip first N papers
            if skipped < skip_papers:
                skipped += 1
                logger.info("Skipping paper %d: %s", skipped, result.title)
                continue
            
            # Stop if we have enough papers
            if len(papers_metadata) >= max_results:
                break
            
            paper_id = result.entry_id.split('/')[-1]
            filename = f"{paper_id}.pdf"
            filepath = os.path.join(self.download_dir, filename)
            
            # Download if not exists
            if not os.path.exists(filepath):
                logger.info("Downloading %s", result.title)
                try:
                    result.download_pdf(dirpath=self.download_dir, filename=filename)
                except Exception as e:
                    logger.error("Failed to download %s: %s", result.title, e)
                    continue
            else:
                logger.info("File %s already exists. Skipping download", filename)
                
            papers_metadata.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "id": paper_id,
                "filepath": filepath,
                "published": str(result.published),
                "url": result.entry_id
            })
        logger.info("Total papers fetched: %d", len(papers_metadata))
        logger.debug("Papers metadata: %s", papers_metadata)
        return papers_metadata
