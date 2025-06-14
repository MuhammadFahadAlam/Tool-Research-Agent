import os
from firecrawl import FirecrawlApp, ScrapeOptions
from dotenv import load_dotenv

load_dotenv()

class FirecrawlService:

    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY is not set in the environment variables")
        self.app = FirecrawlApp(api_key=api_key)

    def search_companies(self, query: str, num_results: int = 5):
        try:
            results = self.app.search(
                query=f"{query} company pricing", 
                limit=num_results,
                scrape_options=ScrapeOptions(
                    formats=["markdown"],
                ))
            return results
        except Exception as e:
            print(f"Error searching companies: {e}")
            return []
        
    def scrape_company_pages(self, url: str):
        try:
            results = self.app.scrape_url(
                url=url,
                formats=["markdown"],
            )
            return results
        except Exception as e:
            print(f"Error scraping company page: {e}")
            return None