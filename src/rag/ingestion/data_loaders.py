import os
import arxiv
from typing import List, Dict
from src.common.config import Config


class ArxivLoader:
    def __init__(self, download_dir: str = Config.PAPERS_DIR):
        self.download_dir = download_dir
        self.client = arxiv.Client()

    def fetch_papers(self, query: str = "cat:cs.AI", max_results: int = 5) -> List[Dict]:
        """
        Fetches papers from arXiv matching the query.
        Returns a list of metadata dictionaries.
        Downloads the PDF if not already present.
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers_metadata = []
        
        for result in self.client.results(search):
            paper_id = result.entry_id.split('/')[-1]
            filename = f"{paper_id}.pdf"
            filepath = os.path.join(self.download_dir, filename)
            
            # Download if not exists
            if not os.path.exists(filepath):
                print(f"Downloading {result.title}...")
                try:
                    result.download_pdf(dirpath=self.download_dir, filename=filename)
                except Exception as e:
                    print(f"Failed to download {result.title}: {e}")
                    continue
            else:
                print(f"File {filename} already exists. Skipping download.")
                
            papers_metadata.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "id": paper_id,
                "filepath": filepath,
                "published": str(result.published),
                "url": result.entry_id
            })
            
        return papers_metadata
