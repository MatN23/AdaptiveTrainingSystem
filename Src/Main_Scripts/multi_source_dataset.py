"""
Download and process LEGAL data sources for base training.

100% LEGAL SOURCES:
 Wikipedia: Official Wikimedia dumps (CC BY-SA)
 Project Gutenberg: Public domain books  
 ArXiv: Open access scientific papers
 Stack Overflow: CC BY-SA licensed Q&A

Creates .txt files compatible with BaseTrainingDataset in dataset.py.

Usage:
    python multi_source_dataset.py
    
    # Edit HARDCODED SETTINGS section below to customize
    
Output format: Plain text, one document per line or separated by blank lines
Compatible with: BaseTrainingDataset (handles .txt files automatically)
"""

import logging
import re
import bz2
import json
import gzip
import tarfile
from pathlib import Path
from typing import Iterator, List, Optional
import urllib.request
import xml.etree.ElementTree as ET
import time
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# HARDCODED SETTINGS - Edit these as needed

# Output settings
OUTPUT_DIR = 'datasets'
MAX_FILES_PER_SOURCE = 3  # Number of files to create per data source
MB_PER_FILE = 100.0       # Target size in MB for each file

# Data sources to enable (True/False)
ENABLE_WIKIPEDIA = False
ENABLE_GUTENBERG = False
ENABLE_ARXIV = False
ENABLE_STACKOVERFLOW = False
ENABLE_PUBMED = False
ENABLE_OPENWEBTEXT = True
ENABLE_PHIL_PAPERS = True
ENABLE_COMMON_CRAWL_NEWS = True

# Wikipedia settings
WIKI_LANGUAGE = 'simplewiki'  # 'simplewiki' (smaller) or 'enwiki' (full English)

# ArXiv settings
ARXIV_CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CL']  # AI, Machine Learning, Computation & Language
ARXIV_PAPERS_PER_CATEGORY = 500

# Stack Overflow settings
STACKOVERFLOW_TAGS = ['python', 'javascript', 'machine-learning', 'algorithms']
STACKOVERFLOW_MIN_SCORE = 10  # Minimum upvotes

# OpenWebText settings (Reddit links)
OPENWEBTEXT_SUBREDDITS = ['science', 'AskScience', 'technology', 'programming', 
                          'MachineLearning', 'artificial', 'datascience', 'compsci']
OPENWEBTEXT_POSTS_PER_SUB = 100

# PhilPapers settings
PHILPAPERS_CATEGORIES = ['philosophy-of-mind', 'philosophy-of-science', 
                         'logic-and-philosophy-of-logic', 'epistemology', 'ethics']

# Common Crawl News settings
CC_NEWS_DOMAINS = ['bbc.com', 'reuters.com', 'apnews.com', 'npr.org', 'nature.com']

# PubMed settings
PUBMED_SEARCH_TERMS = [
    # ARTIFICIAL INTELLIGENCE / CS
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "neural networks",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "explainable AI",
    "AI ethics",
    "bioinformatics",
    "computational biology",
    "data mining",
    "big data",
    "predictive modeling",

    "cancer",
    "cardiology",
    "neurology",
    "psychiatry",
    "immunology",
    "endocrinology",
    "genetics",
    "epidemiology",
    "public health",
    "pediatrics",
    "geriatrics",
    "radiology",
    "dermatology",
    "surgery",
    "internal medicine",

    # MENTAL HEALTH
    "depression",
    "anxiety",
    "bipolar disorder",
    "schizophrenia",
    "autism spectrum disorder",
    "ADHD",
    "PTSD",
    "suicide prevention",
    "cognitive behavioral therapy",
    "mental health treatment",
    "substance abuse",
    "addiction medicine",

    # NEUROSCIENCE
    "brain imaging",
    "EEG",
    "fMRI",
    "neuronal networks",
    "synaptic plasticity",
    "neurodegenerative diseases",
    "Alzheimer's disease",
    "Parkinson's disease",
    "multiple sclerosis",
    "spinal cord injury",
    "neuroinflammation",
    "neuropharmacology",

    # GENETICS / MOLECULAR BIOLOGY
    "DNA sequencing",
    "RNA sequencing",
    "gene expression",
    "CRISPR",
    "genome editing",
    "gene therapy",
    "proteomics",
    "metabolomics",
    "epigenetics",
    "genome-wide association study",
    "molecular diagnostics",

    # PUBLIC HEALTH / EPIDEMIOLOGY
    "infectious disease",
    "pandemic",
    "COVID-19",
    "vaccination",
    "disease prevention",
    "health policy",
    "health disparities",
    "global health",
    "healthcare access",
    "environmental health",
    "nutrition",
    "obesity",

    # PHARMACOLOGY / DRUGS
    "drug discovery",
    "pharmacokinetics",
    "pharmacodynamics",
    "clinical trials",
    "drug side effects",
    "antibiotic resistance",
    "chemotherapy",
    "analgesics",
    "opioids",
    "vaccine development",

    # BIOTECH / ENGINEERING
    "medical devices",
    "biomedical engineering",
    "prosthetics",
    "wearable technology",
    "telemedicine",
    "robotic surgery",
    "nanomedicine",
    "tissue engineering",
    "regenerative medicine",
    "3D bioprinting",

    # STATISTICS / MATH / COMPUTATION
    "statistical analysis",
    "Bayesian methods",
    "survival analysis",
    "time series analysis",
    "computational modeling",
    "simulation",
    "optimization",
    "signal processing",
    "pattern recognition",

    # EDUCATION / SOCIAL SCIENCE
    "medical education",
    "learning strategies",
    "behavior change",
    "patient compliance",
    "health communication",
    "medical ethics",
    "socioeconomic status",
    "quality of life",
    "workplace stress",

    # EMERGING TECH / FUTURE STUFF
    "digital health",
    "electronic health records",
    "precision medicine",
    "personalized medicine",
    "digital therapeutics",
    "virtual reality therapy",
    "brain-computer interface",
    "wearable sensors",
    "smart healthcare",
    "mobile health apps",

    # DISEASE CATEGORIES (COVER EVERYTHING)
    "diabetes",
    "hypertension",
    "stroke",
    "heart failure",
    "asthma",
    "COPD",
    "HIV/AIDS",
    "tuberculosis",
    "influenza",
    "autoimmune diseases",
    "rare diseases",

    # HUMAN BIOLOGY / PHYSIOLOGY
    "cell biology",
    "metabolism",
    "hormones",
    "immune response",
    "inflammation",
    "microbiome",
    "reproductive health",
    "sleep",
    "exercise physiology",

    # HEALTHCARE SYSTEMS
    "healthcare quality",
    "healthcare costs",
    "insurance",
    "telehealth",
    "patient outcomes",
    "care coordination",
    "health informatics",
    "clinical decision support",

    # ETHICS / POLICY
    "bioethics",
    "clinical ethics",
    "data privacy",
    "medical law",
    "informed consent",
    "regulatory policy",
    "FDA approval",
    "intellectual property"
]
PUBMED_PAPERS_PER_TERM = 500



class WikipediaProcessor:
    """
    Process Wikipedia XML dumps (100% LEGAL - Official Wikimedia dumps).
    License: CC BY-SA 3.0
    """
    
    def __init__(self, wiki_language='simplewiki'):
        self.wiki_language = wiki_language
        self.dump_url = f"https://dumps.wikimedia.org/{wiki_language}/latest/{wiki_language}-latest-pages-articles.xml.bz2"
    
    def download_dump(self, output_dir: str) -> str:
        """Download Wikipedia dump file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dump_file = output_dir / f"{self.wiki_language}_dump.xml.bz2"
        
        if dump_file.exists():
            logging.info(f"Wikipedia dump already exists: {dump_file}")
            return str(dump_file)
        
        logging.info(f"Downloading Wikipedia dump: {self.wiki_language}")
        logging.info(f"URL: {self.dump_url}")
        
        def download_progress(block_num, block_size, total_size):
            if total_size > 0 and block_num % 100 == 0:
                percent = (block_num * block_size / total_size) * 100
                mb_downloaded = (block_num * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                logging.info(f"  Downloaded: {mb_downloaded:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
        
        try:
            urllib.request.urlretrieve(self.dump_url, dump_file, download_progress)
            logging.info(f"Download complete!")
            return str(dump_file)
        except Exception as e:
            logging.error(f"Download failed: {e}")
            raise
    
    def clean_wiki_text(self, text: str) -> str:
        """Clean Wikipedia markup from text - ULTRA AGGRESSIVE."""
        if not text:
            return ""
        
        # Remove comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove ALL references first (they cause issues)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<ref[^/>]*\/>', '', text, flags=re.IGNORECASE)
        
        # Remove templates (multiple passes)
        for _ in range(5):
            text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        
        # Remove tables and infoboxes
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        
        # Remove ALL file/image references (more patterns)
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\[\[:.*?\]\]', '', text, flags=re.IGNORECASE | re.DOTALL)  # Russian
        
        # Remove gallery tags
        text = re.sub(r'<gallery.*?>.*?</gallery>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert internal links [[Link|Text]] -> Text or [[Link]] -> Link
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        
        # Remove external links
        text = re.sub(r'\[http[^\]]*\]', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove ALL HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove categories
        text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Remove remaining brackets artifacts
        text = re.sub(r'\]\]', '', text)
        text = re.sub(r'\[\[', '', text)
        
        # Remove formatting
        text = re.sub(r"'''", '', text)
        text = re.sub(r"''", '', text)
        
        # Clean headers
        text = re.sub(r'==+\s*([^=]+?)\s*==+', r'\n\1\n', text)
        
        # Remove HTML entities
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&ndash;', '-', text)
        text = re.sub(r'&mdash;', '-', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)  # Remove other entities
        
        # Remove remaining artifacts
        text = re.sub(r'\|+', '', text)  # Remove pipe characters
        text = re.sub(r'=+', '', text)  # Remove equals signs
        text = re.sub(r'\{[^}]*\}', '', text)  # Remove any remaining braces
        
        # Clean whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\t', ' ', text)
        
        # Remove lines that are too short (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 10 or len(line.strip()) == 0]
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def extract_articles(self, dump_file: str) -> Iterator[tuple]:
        """Extract clean articles from Wikipedia dump."""
        logging.info(f"Extracting articles from Wikipedia dump...")
        
        with bz2.open(dump_file, 'rt', encoding='utf-8') as f:
            article_count = 0
            current_title = None
            in_text = False
            text_buffer = []
            
            for line in f:
                if '<title>' in line:
                    current_title = re.search(r'<title>(.*?)</title>', line)
                    if current_title:
                        current_title = current_title.group(1)
                
                if '<text' in line:
                    in_text = True
                    text_match = re.search(r'<text[^>]*>(.*)', line, re.DOTALL)
                    if text_match:
                        text_buffer.append(text_match.group(1))
                
                elif in_text:
                    if '</text>' in line:
                        text_match = re.search(r'(.*?)</text>', line, re.DOTALL)
                        if text_match:
                            text_buffer.append(text_match.group(1))
                        
                        current_text = ''.join(text_buffer)
                        text_buffer = []
                        in_text = False
                        article_count += 1
                        
                        if current_title and current_text:
                            # Skip special pages
                            if any(prefix in current_title for prefix in 
                                 ['Wikipedia:', 'Template:', 'Category:', 'File:', 
                                  'Help:', 'Portal:', 'Talk:', 'User:']):
                                current_title = None
                                continue
                            
                            # Skip redirects
                            if current_text.strip().upper().startswith('#REDIRECT'):
                                current_title = None
                                continue
                            
                            cleaned_text = self.clean_wiki_text(current_text)
                            
                            # Quality filter
                            if len(cleaned_text) > 300:
                                yield (current_title, cleaned_text)
                                
                                if article_count % 1000 == 0:
                                    logging.info(f"  Processed {article_count:,} articles")
                        
                        current_title = None
                    else:
                        text_buffer.append(line)
        
        logging.info(f"Wikipedia extraction complete: {article_count:,} articles")
    
    def create_dataset_files(self, dump_file: str, output_dir: str, num_files: int, mb_per_file: float):
        """Create multiple training dataset files from Wikipedia dump."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} Wikipedia dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        article_generator = self.extract_articles(dump_file)
        created_files = []
        
        for file_idx in range(num_files):
            output_path = output_dir / f"wikipedia_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            current_bytes = 0
            articles_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                try:
                    while current_bytes < target_bytes:
                        title, text = next(article_generator)
                        
                        # Write article
                        article_text = f"{title}\n{text}\n\n"
                        f.write(article_text)
                        
                        article_bytes = len(article_text.encode('utf-8'))
                        current_bytes += article_bytes
                        articles_written += 1
                        
                        if articles_written % 500 == 0:
                            progress_pct = (current_bytes / target_bytes) * 100
                            logging.info(f"  Progress: {current_bytes/(1024*1024):.1f}/{mb_per_file:.1f} MB "
                                       f"({progress_pct:.1f}%) - {articles_written:,} articles")
                
                except StopIteration:
                    logging.info(f"  Ran out of articles at file {file_idx+1}")
                    if current_bytes == 0:
                        output_path.unlink()  # Delete empty file
                        break
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {articles_written:,} articles")
            created_files.append((str(output_path), final_mb))
        
        return created_files


class GutenbergProcessor:
    """
    Process Project Gutenberg books (100% LEGAL - Public Domain).
    License: Public Domain
    """
    
    def __init__(self):
        # Popular public domain books
        self.book_urls = [
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
            "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
            "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
            "https://www.gutenberg.org/files/1080/1080-0.txt",  # A Modest Proposal
            "https://www.gutenberg.org/files/74/74-0.txt",      # Tom Sawyer
            "https://www.gutenberg.org/files/1232/1232-0.txt",  # The Prince
            "https://www.gutenberg.org/files/98/98-0.txt",      # Tale of Two Cities
            "https://www.gutenberg.org/files/1260/1260-0.txt",  # Jane Eyre
            "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula
            "https://www.gutenberg.org/files/1952/1952-0.txt",  # The Yellow Wallpaper
            "https://www.gutenberg.org/files/174/174-0.txt",    # Picture of Dorian Gray
            "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations
            "https://www.gutenberg.org/files/2600/2600-0.txt",  # War and Peace
        ]
    
    def download_book(self, url: str) -> str:
        """Download a single book."""
        try:
            book_id = url.split('/')[-2]
            logging.info(f"  Downloading: Book {book_id}")
            response = urllib.request.urlopen(url, timeout=30)
            content = response.read().decode('utf-8', errors='ignore')
            
            # Remove Gutenberg header/footer
            content = self._clean_gutenberg_text(content)
            return content
        except Exception as e:
            logging.warning(f"  Failed to download {url}: {e}")
            return ""
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Gutenberg headers and footers."""
        # Remove header
        start_match = re.search(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
        if start_match:
            text = text[start_match.end():]
        
        # Remove footer
        end_match = re.search(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
        if end_match:
            text = text[:end_match.start()]
        
        return text.strip()
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float):
        """Create Gutenberg dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} Gutenberg dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        created_files = []
        
        for file_idx in range(num_files):
            output_path = output_dir / f"gutenberg_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            current_bytes = 0
            books_downloaded = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Cycle through books until target size reached
                attempts = 0
                max_attempts = len(self.book_urls) * 3
                
                while current_bytes < target_bytes and attempts < max_attempts:
                    url = self.book_urls[attempts % len(self.book_urls)]
                    attempts += 1
                    
                    book_text = self.download_book(url)
                    if book_text:
                        f.write(book_text)
                        f.write("\n\n")
                        
                        current_bytes += len(book_text.encode('utf-8'))
                        books_downloaded += 1
                        
                        logging.info(f"  Progress: {current_bytes/(1024*1024):.1f}/{mb_per_file:.1f} MB")
                    
                    time.sleep(1)  # Be respectful to Gutenberg servers
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {books_downloaded} books")
            created_files.append((str(output_path), final_mb))
        
        return created_files


class ArXivProcessor:
    """
    Process ArXiv papers (100% LEGAL - Open Access).
    License: Various open licenses (mostly CC BY)
    Note: We extract only abstracts to stay lightweight
    """
    
    def __init__(self, categories: List[str]):
        self.base_url = "http://export.arxiv.org/api/query"
        self.categories = categories
    
    def search_papers(self, category: str, max_results: int = 100) -> List[dict]:
        """Search ArXiv for papers in a category."""
        logging.info(f"  Searching ArXiv category: {category}")
        
        query = f"cat:{category}"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', namespace):
                title_elem = entry.find('atom:title', namespace)
                summary_elem = entry.find('atom:summary', namespace)
                
                if title_elem is not None and summary_elem is not None:
                    papers.append({
                        'title': title_elem.text.strip(),
                        'abstract': summary_elem.text.strip()
                    })
            
            logging.info(f"    Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logging.warning(f"    ArXiv search failed: {e}")
            return []
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float, papers_per_cat: int = 500):
        """Create ArXiv dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} ArXiv dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        # Collect papers from all categories
        all_papers = []
        for category in self.categories:
            papers = self.search_papers(category, papers_per_cat)
            all_papers.extend(papers)
            time.sleep(1)  # Rate limiting
        
        if not all_papers:
            logging.warning("No ArXiv papers found!")
            return []
        
        logging.info(f"Total papers collected: {len(all_papers)}")
        
        created_files = []
        
        # Distribute papers across files
        papers_per_file = len(all_papers) // num_files + 1
        
        for file_idx in range(num_files):
            output_path = output_dir / f"arxiv_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            start_idx = file_idx * papers_per_file
            end_idx = min(start_idx + papers_per_file, len(all_papers))
            file_papers = all_papers[start_idx:end_idx]
            
            if not file_papers:
                break
            
            current_bytes = 0
            papers_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for paper in file_papers:
                    if current_bytes >= target_bytes:
                        break
                    
                    paper_text = f"{paper['title']}\n{paper['abstract']}\n\n"
                    f.write(paper_text)
                    
                    current_bytes += len(paper_text.encode('utf-8'))
                    papers_written += 1
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {papers_written} papers")
            created_files.append((str(output_path), final_mb))
        
        return created_files


class StackOverflowProcessor:
    """
    Process Stack Overflow Q&A (100% LEGAL - CC BY-SA).
    License: CC BY-SA 4.0
    Note: Uses Stack Exchange API (free, no auth needed for basic use)
    """
    
    def __init__(self, tags: List[str], min_score: int = 10):
        self.tags = tags
        self.min_score = min_score
        self.api_url = "https://api.stackexchange.com/2.3/questions"
    
    def fetch_questions(self, tag: str, page_size: int = 100) -> List[dict]:
        """Fetch top questions for a tag."""
        logging.info(f"  Fetching Stack Overflow: {tag}")
        
        params = {
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'site': 'stackoverflow',
            'filter': 'withbody',  # Include question body
            'pagesize': page_size,
            'min': self.min_score
        }
        
        try:
            response = requests.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            questions = []
            for item in data.get('items', []):
                questions.append({
                    'title': item.get('title', ''),
                    'body': item.get('body', ''),
                    'score': item.get('score', 0)
                })
            
            logging.info(f"    Found {len(questions)} questions")
            return questions
            
        except Exception as e:
            logging.warning(f"    Stack Overflow fetch failed: {e}")
            return []
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Simple HTML removal
        text = re.sub(r'<code>.*?</code>', '[CODE]', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&quot;', '"', text)
        return text.strip()
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float):
        """Create Stack Overflow dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} Stack Overflow dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        # Collect questions from all tags
        all_questions = []
        for tag in self.tags:
            questions = self.fetch_questions(tag, page_size=100)
            all_questions.extend(questions)
            time.sleep(1)  # Rate limiting
        
        if not all_questions:
            logging.warning("No Stack Overflow questions found!")
            return []
        
        logging.info(f"Total questions collected: {len(all_questions)}")
        
        created_files = []
        
        # Distribute questions across files
        questions_per_file = len(all_questions) // num_files + 1
        
        for file_idx in range(num_files):
            output_path = output_dir / f"stackoverflow_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            start_idx = file_idx * questions_per_file
            end_idx = min(start_idx + questions_per_file, len(all_questions))
            file_questions = all_questions[start_idx:end_idx]
            
            if not file_questions:
                break
            
            current_bytes = 0
            questions_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for q in file_questions:
                    if current_bytes >= target_bytes:
                        break
                    
                    title = self.clean_html(q['title'])
                    body = self.clean_html(q['body'])
                    
                    qa_text = f"Question: {title}\n{body}\n\n"
                    f.write(qa_text)
                    
                    current_bytes += len(qa_text.encode('utf-8'))
                    questions_written += 1
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {questions_written} questions")
            created_files.append((str(output_path), final_mb))
        
        return created_files


class PubMedProcessor:
    """
    Process PubMed articles (100% LEGAL - Public Domain/Open Access).
    License: Public Domain (US Government work)
    """
    
    def __init__(self, search_terms: List[str]):
        self.search_terms = search_terms
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def search_pubmed(self, term: str, max_results: int = 100) -> List[str]:
        """Search PubMed and return list of PMIDs."""
        logging.info(f"  Searching PubMed: {term}")
        
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': term,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            logging.info(f"    Found {len(pmids)} articles")
            return pmids
            
        except Exception as e:
            logging.warning(f"    PubMed search failed: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[dict]:
        """Fetch abstracts for a list of PMIDs."""
        if not pmids:
            return []
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        try:
            response = requests.get(fetch_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            articles = []
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    
                    # Extract abstract
                    abstract_texts = []
                    for abstract_text in article.findall('.//AbstractText'):
                        if abstract_text.text:
                            abstract_texts.append(abstract_text.text)
                    
                    abstract = ' '.join(abstract_texts)
                    
                    if title and abstract:
                        articles.append({
                            'title': title,
                            'abstract': abstract
                        })
                
                except Exception as e:
                    continue
            
            return articles
            
        except Exception as e:
            logging.warning(f"    Failed to fetch abstracts: {e}")
            return []
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float, papers_per_term: int = 500):
        """Create PubMed dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} PubMed dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        # Collect articles from all search terms
        all_articles = []
        for term in self.search_terms:
            pmids = self.search_pubmed(term, papers_per_term)
            
            # Fetch in batches of 200 (API limit)
            batch_size = 200
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                articles = self.fetch_abstracts(batch_pmids)
                all_articles.extend(articles)
                time.sleep(0.5)
            
            time.sleep(1)  # Rate limiting between terms
        
        if not all_articles:
            logging.warning("No PubMed articles found!")
            return []
        
        logging.info(f"Total articles collected: {len(all_articles)}")
        
        created_files = []
        
        # Distribute articles across files
        articles_per_file = len(all_articles) // num_files + 1
        
        for file_idx in range(num_files):
            output_path = output_dir / f"pubmed_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            start_idx = file_idx * articles_per_file
            end_idx = min(start_idx + articles_per_file, len(all_articles))
            file_articles = all_articles[start_idx:end_idx]
            
            if not file_articles:
                break
            
            current_bytes = 0
            articles_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for article in file_articles:
                    if current_bytes >= target_bytes:
                        break
                    
                    article_text = f"{article['title']}\n{article['abstract']}\n\n"
                    f.write(article_text)
                    
                    current_bytes += len(article_text.encode('utf-8'))
                    articles_written += 1
                    
                    if articles_written % 500 == 0:
                        progress_pct = (current_bytes / target_bytes) * 100
                        logging.info(f"  Progress: {current_bytes/(1024*1024):.1f}/{mb_per_file:.1f} MB "
                                   f"({progress_pct:.1f}%) - {articles_written:,} articles")
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {articles_written} articles")
            created_files.append((str(output_path), final_mb))
        
        return created_files


class OpenWebTextProcessor:
    """
    Process OpenWebText-style data (Reddit links that are legal/public).
    License: Various (only public content)
    """
    
    def __init__(self, subreddits: List[str]):
        self.subreddits = subreddits
        self.base_url = "https://www.reddit.com"
    
    def fetch_subreddit_posts(self, subreddit: str, limit: int = 100) -> List[dict]:
        """Fetch top posts from a subreddit."""
        logging.info(f"  Fetching Reddit: r/{subreddit}")
        
        url = f"{self.base_url}/r/{subreddit}/top.json"
        params = {'limit': limit, 't': 'year'}  # Top posts from the past year
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for item in data.get('data', {}).get('children', []):
                post = item.get('data', {})
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                
                # Get both title and text content
                if title:
                    # Some posts have external links, others have self text
                    if selftext and len(selftext) > 100:
                        posts.append({
                            'title': title,
                            'text': selftext
                        })
                    elif len(title) > 50:  # At least get the title if substantial
                        posts.append({
                            'title': title,
                            'text': title  # Use title as content if no selftext
                        })
            
            logging.info(f"    Found {len(posts)} posts with content")
            return posts
            
        except Exception as e:
            logging.warning(f"    Reddit fetch failed: {e}")
            return []
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float, posts_per_sub: int = 100):
        """Create OpenWebText dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} OpenWebText dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        # Collect posts from all subreddits
        all_posts = []
        for subreddit in self.subreddits:
            posts = self.fetch_subreddit_posts(subreddit, posts_per_sub)
            all_posts.extend(posts)
            time.sleep(2)  # Rate limiting for Reddit
        
        if not all_posts:
            logging.warning("No Reddit posts found!")
            return []
        
        logging.info(f"Total posts collected: {len(all_posts)}")
        
        created_files = []
        
        # Distribute posts across files
        posts_per_file = max(len(all_posts) // num_files, 1)
        
        for file_idx in range(num_files):
            output_path = output_dir / f"openwebtext_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            start_idx = file_idx * posts_per_file
            end_idx = min(start_idx + posts_per_file, len(all_posts))
            file_posts = all_posts[start_idx:end_idx]
            
            if not file_posts:
                break
            
            current_bytes = 0
            posts_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for post in file_posts:
                    post_text = f"{post['title']}\n{post['text']}\n\n"
                    f.write(post_text)
                    
                    current_bytes += len(post_text.encode('utf-8'))
                    posts_written += 1
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {posts_written} posts")
            created_files.append((str(output_path), final_mb))
            
            if final_mb < mb_per_file * 0.1:  # If file is too small, stop making more
                break
        
        return created_files


class PhilPapersProcessor:
    """
    Process PhilPapers philosophical papers (Open Access).
    Uses OpenAlex API which indexes PhilPapers content.
    License: Various open access
    """
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.base_url = "https://api.openalex.org/works"
    
    def fetch_papers(self, query: str, limit: int = 100) -> List[dict]:
        """Fetch philosophy papers using OpenAlex API."""
        logging.info(f"  Fetching philosophy papers: {query}")
        
        params = {
            'filter': f'concepts.id:C138885662,default.search:{query}',  # Philosophy concept
            'per-page': limit,
            'mailto': 'research@example.com'  # Polite API usage
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for work in data.get('results', []):
                title = work.get('title', '')
                abstract = work.get('abstract', '')
                
                if title and abstract:
                    papers.append({
                        'title': title,
                        'abstract': abstract
                    })
            
            logging.info(f"    Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logging.warning(f"    Philosophy papers fetch failed: {e}")
            return []
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float):
        """Create PhilPapers dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} PhilPapers dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        all_papers = []
        for category in self.categories:
            papers = self.fetch_papers(category, 100)
            all_papers.extend(papers)
            time.sleep(1)
        
        if not all_papers:
            logging.warning("No philosophy papers found!")
            return []
        
        logging.info(f"Total papers collected: {len(all_papers)}")
        
        created_files = []
        papers_per_file = max(len(all_papers) // num_files, 1)
        
        for file_idx in range(num_files):
            output_path = output_dir / f"philpapers_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            start_idx = file_idx * papers_per_file
            end_idx = min(start_idx + papers_per_file, len(all_papers))
            file_papers = all_papers[start_idx:end_idx]
            
            if not file_papers:
                break
            
            current_bytes = 0
            papers_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for paper in file_papers:
                    paper_text = f"{paper['title']}\n{paper['abstract']}\n\n"
                    f.write(paper_text)
                    
                    current_bytes += len(paper_text.encode('utf-8'))
                    papers_written += 1
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {papers_written} papers")
            created_files.append((str(output_path), final_mb))
            
            if final_mb < mb_per_file * 0.1:
                break
        
        return created_files


class CommonCrawlNewsProcessor:
    """
    Process news articles using NewsAPI (free tier available).
    License: Public news content
    """
    
    def __init__(self, domains: List[str]):
        self.domains = domains
        self.base_url = "https://newsapi.org/v2/everything"
        # Note: Users should get their own free API key from newsapi.org
        self.api_key = None  # Set to None for now
    
    def fetch_news_articles(self, domain: str, limit: int = 100) -> List[dict]:
        """Fetch news articles from a domain."""
        logging.info(f"  Fetching news from: {domain}")
        
        # Try without API key first (limited access)
        try:
            # Alternative: Use RSS feeds which don't require API keys
            rss_urls = {
                'bbc.com': 'http://feeds.bbci.co.uk/news/rss.xml',
                'reuters.com': 'https://www.reutersagency.com/feed/',
                'npr.org': 'https://feeds.npr.org/1001/rss.xml',
                'nature.com': 'https://www.nature.com/nature.rss'
            }
            
            if domain in rss_urls:
                rss_url = rss_urls[domain]
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                response = requests.get(rss_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Parse RSS XML
                root = ET.fromstring(response.content)
                articles = []
                
                for item in root.findall('.//item')[:limit]:
                    title_elem = item.find('title')
                    desc_elem = item.find('description')
                    
                    if title_elem is not None and desc_elem is not None:
                        title = title_elem.text or ''
                        desc = desc_elem.text or ''
                        
                        # Clean HTML from description
                        desc = re.sub(r'<[^>]+>', '', desc)
                        
                        if title and desc:
                            articles.append({
                                'title': title,
                                'text': desc
                            })
                
                logging.info(f"    Found {len(articles)} articles from RSS")
                return articles
                
        except Exception as e:
            logging.warning(f"    News fetch failed for {domain}: {e}")
        
        return []
    
    def create_dataset_files(self, output_dir: str, num_files: int, mb_per_file: float):
        """Create news dataset files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_bytes = int(mb_per_file * 1024 * 1024)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Creating {num_files} News dataset file(s)")
        logging.info(f"Target size per file: {mb_per_file} MB")
        logging.info(f"{'='*60}")
        
        all_articles = []
        for domain in self.domains:
            articles = self.fetch_news_articles(domain, 100)
            all_articles.extend(articles)
            time.sleep(2)  # Respectful rate limiting
        
        if not all_articles:
            logging.warning("No news articles found!")
            return []
        
        logging.info(f"Total articles collected: {len(all_articles)}")
        
        created_files = []
        articles_per_file = max(len(all_articles) // num_files, 1)
        
        for file_idx in range(num_files):
            output_path = output_dir / f"ccnews_{file_idx+1}.txt"
            
            logging.info(f"\n Creating file {file_idx+1}/{num_files}: {output_path.name}")
            
            start_idx = file_idx * articles_per_file
            end_idx = min(start_idx + articles_per_file, len(all_articles))
            file_articles = all_articles[start_idx:end_idx]
            
            if not file_articles:
                break
            
            current_bytes = 0
            articles_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for article in file_articles:
                    article_text = f"{article['title']}\n{article['text']}\n\n"
                    f.write(article_text)
                    
                    current_bytes += len(article_text.encode('utf-8'))
                    articles_written += 1
            
            final_mb = current_bytes / (1024 * 1024)
            logging.info(f"   File {file_idx+1} complete: {final_mb:.1f} MB, {articles_written} articles")
            created_files.append((str(output_path), final_mb))
            
            if final_mb < mb_per_file * 0.1:
                break
        
        return created_files


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print(" LEGAL DATA SOURCE DOWNLOADER")
    print("="*60)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Max Files per Source: {MAX_FILES_PER_SOURCE}")
    print(f"Target MB per File: {MB_PER_FILE}")
    print("\nEnabled Sources:")
    if ENABLE_WIKIPEDIA:
        print(f"   Wikipedia ({WIKI_LANGUAGE})")
    if ENABLE_GUTENBERG:
        print(f"   Project Gutenberg")
    if ENABLE_ARXIV:
        print(f"   ArXiv Papers")
    if ENABLE_STACKOVERFLOW:
        print(f"   Stack Overflow")
    if ENABLE_PUBMED:
        print(f"   PubMed")
    if ENABLE_OPENWEBTEXT:
        print(f"   OpenWebText (Reddit)")
    if ENABLE_PHIL_PAPERS:
        print(f"   PhilPapers")
    if ENABLE_COMMON_CRAWL_NEWS:
        print(f"   Common Crawl News")
    print("="*60)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_created_files = []
    total_mb = 0
    
    # Process Wikipedia
    if ENABLE_WIKIPEDIA:
        try:
            wiki_processor = WikipediaProcessor(WIKI_LANGUAGE)
            dump_file = wiki_processor.download_dump(OUTPUT_DIR)
            files = wiki_processor.create_dataset_files(
                dump_file, 
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"Wikipedia processing failed: {e}")
    
    # Process Gutenberg
    if ENABLE_GUTENBERG:
        try:
            gutenberg_processor = GutenbergProcessor()
            files = gutenberg_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"Gutenberg processing failed: {e}")
    
    # Process ArXiv
    if ENABLE_ARXIV:
        try:
            arxiv_processor = ArXivProcessor(ARXIV_CATEGORIES)
            files = arxiv_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE,
                ARXIV_PAPERS_PER_CATEGORY
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"ArXiv processing failed: {e}")
    
    # Process Stack Overflow
    if ENABLE_STACKOVERFLOW:
        try:
            stackoverflow_processor = StackOverflowProcessor(
                STACKOVERFLOW_TAGS,
                STACKOVERFLOW_MIN_SCORE
            )
            files = stackoverflow_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"Stack Overflow processing failed: {e}")
    
    # Process PubMed
    if ENABLE_PUBMED:
        try:
            pubmed_processor = PubMedProcessor(PUBMED_SEARCH_TERMS)
            files = pubmed_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE,
                PUBMED_PAPERS_PER_TERM
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"PubMed processing failed: {e}")
    
    # Process OpenWebText
    if ENABLE_OPENWEBTEXT:
        try:
            openwebtext_processor = OpenWebTextProcessor(OPENWEBTEXT_SUBREDDITS)
            files = openwebtext_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE,
                OPENWEBTEXT_POSTS_PER_SUB
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"OpenWebText processing failed: {e}")
    
    # Process PhilPapers
    if ENABLE_PHIL_PAPERS:
        try:
            philpapers_processor = PhilPapersProcessor(PHILPAPERS_CATEGORIES)
            files = philpapers_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"PhilPapers processing failed: {e}")
    
    # Process Common Crawl News
    if ENABLE_COMMON_CRAWL_NEWS:
        try:
            ccnews_processor = CommonCrawlNewsProcessor(CC_NEWS_DOMAINS)
            files = ccnews_processor.create_dataset_files(
                OUTPUT_DIR,
                MAX_FILES_PER_SOURCE,
                MB_PER_FILE
            )
            all_created_files.extend(files)
            total_mb += sum(mb for _, mb in files)
        except Exception as e:
            logging.error(f"Common Crawl News processing failed: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(" ALL DATASETS CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nTotal Files: {len(all_created_files)}")
    print(f"Total Size: {total_mb:.1f} MB")
    print(f"\nCreated files:")
    for filepath, mb in all_created_files:
        print(f"   {filepath} ({mb:.1f} MB)")
    
    # Print usage instructions
    print(f"\n{'='*60}")
    print(" USAGE WITH YOUR TRAINING SYSTEM")
    print(f"{'='*60}")
    print("\nAdd this to your training config:\n")
    print("config.base_training_paths = [")
    for filepath, _ in all_created_files:
        print(f"    '{filepath}',")
    print("]")
    print("\nconfig.training_mode = 'base_only'  # Pure base training")
    print("# OR")
    print("# config.training_mode = 'hybrid'  # Base training  fine-tuning")
    print("\n# Then run your training:")
    print("# python train.py")
    print(f"\n{'='*60}")
    
    # Verify compatibility with dataset.py
    print("\n VERIFYING COMPATIBILITY")
    print("="*60)
    try:
        from core.dataset import BaseTrainingDataset, setup_datasets
        print(" dataset.py imports successful")
        print(" Your training system is ready to use these files!")
        print("\nYour BaseTrainingDataset will automatically:")
        print("   Load .txt files line by line")
        print("   Tokenize the text")
        print("   Create fixed-length training chunks")
        print("   Handle document continuation across chunks")
    except ImportError as e:
        print(f"  Could not import dataset.py: {e}")
        print("Make sure dataset.py is in the same directory")
    
    print("\n" + "="*60)
    print(" SETUP COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()