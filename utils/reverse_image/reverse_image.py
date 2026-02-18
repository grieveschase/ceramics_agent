import time
import logging
import requests
from bs4 import BeautifulSoup
from typing import Optional
from urllib.parse import quote, urlparse
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class GoogleReverseImageSearch:
    def __init__(self):
                
        self.base_url = "https://lens.google.com/uploadbyurl"
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        self.retry_count = 3
        self.retry_delay = 1

    def response(self, query: str, image_url: str, max_results: int = 10) -> Optional[list]:
        self._validate_input(query, image_url)        
        encoded_query = quote(query)
        encoded_image_url = quote(image_url)
        
        url = f"{self.base_url}?q={encoded_query}&url={encoded_image_url}&sbisrc=cr_1_5_2"
        
        logger.info(f"GoogleReverseImageSearch url: [{url}]")
        
        
        response = self._make_request(url)
        if response is None:
            logger.warning("No response received from the server.")
            return None

        matching_links, valid_content = self._parse_search_results_2(response.text)
        if not valid_content:
            logger.warning("Unexpected HTML structure encountered.")
            return None
        
        logger.info(f"GoogleReverseImageSearch total found matching links: [{len(matching_links)}]")

        return matching_links[:min(max_results, len(matching_links))], url
    
    def _validate_input(self, query: str, image_url: str):
        if not query:
            raise ValueError("Query not found. Please enter a query and try again.")
        if not image_url:
            raise ValueError("Image URL not found. Please enter an image URL and try again.")
        if not self._validate_image_url(image_url):
            raise ValueError("Invalid image URL. Please enter a valid image URL and try again.")
    
    def _validate_image_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
        return any(path.endswith(ext) for ext in valid_extensions)
    
    def _make_request(self, url: str):
        attempts = 0
        while attempts < self.retry_count:
            try:
                response = requests.get(url, headers=self.headers, verify=False)
                if response.headers.get('Content-Type', '').startswith('text/html'):
                    response.raise_for_status()
                    return response
                else:
                    logger.warning("Non-HTML content received.")
                    return None
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}")
                attempts += 1
                time.sleep(self.retry_delay)
            except Exception as err:
                logger.error(f"An error occurred: {err}")
                return None
        return None

    def _parse_search_results(self, html_content: str) -> (Optional[list], bool):
        # html_content = response.text
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            search_list = soup.find_all(string = "Search Results")
            result = search_list[0].previous.previous
            links = result.find_all('a', href=True)
            matching_links = [link['href'] for link in links if link.get('href',None)]
            return matching_links, True
        except Exception as e:
            print(f"Error parsing HTML content: {e}")
            logger.error(f"Error parsing HTML content: {e}")
            return None, False
    
    def _parse_search_results_2(self, html_content: str) -> (Optional[list[list[str, str]]], bool):
        # html_content = response.text
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            search_el = soup.find_all('div', id = "search")
                       
            if not search_el or len(search_el) == 0:
                logger.warning("No search element found.")
                return None, False
            
            links = search_el[0].find_all('a', href=True)
            if not links or len(links) == 0:
                logger.warning("No href links found.")
                return None, False
            matching_links = [[link.get_text(strip=True), link['href']] for link in links if link.get('href',None)]
            
            if not matching_links or len(matching_links) == 0:
                logger.warning("No matching links found.")
                return None, False
            return matching_links, True
        except Exception as e:
            print(f"Error parsing HTML content: {e}")
            logger.error(f"Error parsing HTML content: {e}")
            return None, False


import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl_website(url: str):
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_conf
        )
        
        return result, result.markdown

async def crawl_website_simple(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        return result, result.markdown

if __name__ == "__main__":

    request = GoogleReverseImageSearch()

    image_url = r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32727997.jpg"
    query="Vase"
    site_filter_query = r"(site: toomeyco.com)"
    query = f"{query} {site_filter_query}"
    matching_links, url = request.response(
        query=query,
        image_url=image_url,
        max_results=20
        )
    print(url)
    print(len(matching_links))
    
    matching_links_urls = [link[1] for link in matching_links]
    test_url = matching_links_urls[0]

    print(test_url)
    #result, markdown = asyncio.run(crawl_website(test_url))
    result, markdown = asyncio.run(crawl_website_simple(test_url))
    print('break')
    
    #print(result)
    print(markdown)




    



    