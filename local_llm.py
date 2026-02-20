import os
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate               # ← Updated import
from langchain_core.documents import Document
import re
import hashlib
import shutil
import json
import requests
import time
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bparser import BibTexParser
import unicodedata

try:
    from tqdm.notebook import tqdm  # For Jupyter Notebook
except ImportError:
    from tqdm import tqdm  # Fallback for other environments
from fuzzywuzzy import fuzz  # For title/author filtering

class LocalLLM:
    """A class for managing a local LLM with RAG over a PDF database (FAISS only)."""

    def __init__(self, pdf_dir=None, model="llama3", embedding_model="nomic-embed-text", 
                 vector_store_dir="vector_store_db",
                 context="Focus on Indigenous Traditional Ecological Knowledge",
                 use_web_enrichment=True, use_llm_inference=True, verbose=False,
                 bibtex_file="references.bib", query_only_mode=False):
        
        self.query_only_mode = query_only_mode
        
        # Skip pdf_dir validation when in query-only mode (browser/server usage)
        if not self.query_only_mode:
            if pdf_dir is None:
                help_message = (
                    "Error: No arguments provided or missing required parameter 'pdf_dir' for LocalLLM initialization.\n"
                    "How to Use:\n"
                    "- Purpose: Initialize the LocalLLM class to process PDFs and manage a vector store for RAG.\n"
                    "- Required Parameters:\n"
                    "  - pdf_dir: A string specifying the path to a directory containing PDF files.\n"
                    "- Optional Parameters:\n"
                    "  - model: LLM model name (default: 'llama3').\n"
                    "  - embedding_model: Embedding model name (default: 'nomic-embed-text').\n"
                    "  - vector_store_dir: Directory for vector store (default: 'vector_store_db').\n"
                    "  - context: Context for queries (default: 'Focus on Indigenous Traditional Ecological Knowledge').\n"
                    "  - use_web_enrichment: Enable web metadata enrichment (default: True).\n"
                    "  - use_llm_inference: Enable LLM metadata inference (default: True).\n"
                    "  - verbose: Enable detailed output during initialization (default: False).\n"
                    "  - bibtex_file: Path to BibTeX file for metadata storage (default: 'references.bib').\n"
                    "- Example:\n"
                    "```python\n"
                    "from local_llm import LocalLLM\n"
                    "llm = LocalLLM(pdf_dir='path/to/pdfs', verbose=True)\n"
                    "```"
                )
                print(help_message)
                raise ValueError("Missing pdf_dir: Must provide a valid directory path.")
            
            if not isinstance(pdf_dir, str):
                raise ValueError("Invalid pdf_dir: Must be a string path.")
            
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir)
                print(f"Created directory {pdf_dir} as it did not exist.")
            
            if not os.path.isdir(pdf_dir):
                raise ValueError("Invalid pdf_dir: Must be a valid directory path.")
        
        self.pdf_dir = pdf_dir
        self.model = model
        self.embedding_model = embedding_model
        self.vector_store_dir = vector_store_dir
        self.context = context
        self.use_web_enrichment = use_web_enrichment
        self.use_llm_inference = use_llm_inference
        self.verbose = verbose
        self.bibtex_file = bibtex_file
        self.bib_dict = self._load_bibtex()
        
        if not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)
        
        self.llm = OllamaLLM(model=self.model, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        self.vector_store = None
        self.documents = None
        self.metadata_fallback_report = []
        
        self._initialize_vector_store()
        
    def _load_bibtex(self):
        """Parse references.bib into a dict: {filename: {'dc.title': ..., 'dc.creator': ..., 'dc.date': ..., 'dc.type': ...}}"""
        if not os.path.exists(self.bibtex_file):
            if self.verbose:
                print(f"BibTeX file '{self.bibtex_file}' not found. Using extraction fallbacks.")
            return {}
        
        try:
            with open(self.bibtex_file, encoding="utf-8") as bibtex_file:
                parser = bibtexparser.bparser.BibTexParser(
                    common_strings=True,
                    ignore_nonstandard_types=False,
                    homogenize_fields=False,
                    interpolate_strings=False
                )
                bib_database = bibtexparser.load(bibtex_file, parser=parser)
            
            bib_dict = {}
            bibtex_to_display_type = {
                'misc': 'Document',
                'techreport': 'Report',
                'article': 'Article',
                'book': 'Book',
                'inproceedings': 'Conference',
                'phdthesis': 'Thesis'
            }
            
            for entry in bib_database.entries:
                filename = entry.get('note', '').replace('Filename: ', '').strip()
                if not filename:
                    if self.verbose:
                        print(f"Skipping BibTeX entry {entry.get('ID', 'Unknown')}: No filename in note field.")
                    continue
                
                display_type = bibtex_to_display_type.get(entry['ENTRYTYPE'].lower(), 'Document')
                bib_dict[filename] = {
                    'dc.title': entry.get('title', 'Untitled Document'),
                    'dc.creator': entry.get('author', 'Unknown Author'),
                    'dc.date': entry.get('year', 'n.d.'),
                    'dc.type': display_type
                }
                if self.verbose:
                    print(f"Loaded BibTeX metadata for {filename}")
            
            if self.verbose:
                print(f"Loaded {len(bib_dict)} entries from BibTeX.")
            return bib_dict
        
        except Exception as e:
            self.metadata_fallback_report.append(f"Error parsing BibTeX file: {str(e)}")
            if self.verbose:
                print(f"Error parsing BibTeX: {str(e)}. Falling back to extraction.")
            return {}
            
    def _initialize_vector_store(self):
        """Initialize or load the FAISS vector store."""
        folder_path = self.vector_store_dir
        index_file = os.path.join(folder_path, "index.faiss")

        if self.query_only_mode:
            if os.path.exists(index_file):
                try:
                    self.vector_store = FAISS.load_local(
                        folder_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    if self.verbose:
                        print(f"Query-only mode: Loaded FAISS from {folder_path} ({self.vector_store.index.ntotal} vectors)")
                except Exception as e:
                    print(f"Failed to load FAISS in query-only mode: {e}")
                    self.vector_store = None
            else:
                print(f"No FAISS index found in query-only mode: {index_file}")
                self.vector_store = None
            return

        # ── Full/build mode ──────────────────────────────────────────────
        loaded = False
        if os.path.exists(index_file):
            try:
                self.vector_store = FAISS.load_local(
                    folder_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                loaded = True
                if self.verbose:
                    print(f"Loaded existing FAISS from {folder_path} ({self.vector_store.index.ntotal} vectors)")
            except Exception as e:
                print(f"FAISS load failed: {e}. Will create new when documents are processed.")
        
        if not loaded:
            print("No valid FAISS index found or load failed → will create new when documents are added.")
            self.vector_store = None

        # If no store yet and PDFs exist → build it
        if self.vector_store is None and self.pdf_dir and os.path.isdir(self.pdf_dir):
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")]
            if pdf_files:
                if self.verbose:
                    print(f"Building new FAISS from {len(pdf_files)} PDFs")
                documents = self._load_pdfs(self.pdf_dir, verbose=self.verbose)
                if documents:
                    self.vector_store = self._setup_vector_store(documents, verbose=self.verbose)                
                
    def _setup_vector_store(self, documents, verbose=False):
        """Split documents and create/update the FAISS vector store."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        if self.vector_store is None:
            # First-time creation
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        else:
            # Add to existing
            self.vector_store.add_documents(split_docs)

        self.vector_store.save_local(self.vector_store_dir)
        if verbose:
            print(f"FAISS updated with {len(split_docs)} chunks → saved to {self.vector_store_dir}")
        return self.vector_store

    def _load_pdfs(self, pdf_dir, verbose=False):
        """Load PDFs from directory, extract text and metadata (prioritizing BibTeX)."""
        documents = []
        pdf_files = self._remove_duplicate_pdfs(pdf_dir)  # Existing duplicate removal
        
        for filename in tqdm(pdf_files, desc="Loading PDFs"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                pdf = fitz.open(filepath)
                text = "".join(page.get_text("text") or "" for page in pdf)
                metadata = {'source': filename}
                
                # Prioritize BibTeX metadata
                if filename in self.bib_dict:
                    metadata.update(self.bib_dict[filename])
                    if verbose:
                        print(f"Using BibTeX metadata for {filename}: {metadata}")
                else:
                    # Fallback to extraction/inference
                    if verbose:
                        print(f"No BibTeX entry for {filename}. Extracting metadata.")
                    metadata["dc.title"] = self._extract_title_from_pdf(pdf) or self._clean_filename_title(filename)
                    metadata["dc.creator"] = self._extract_author_from_pdf(pdf) or "Unknown Author"
                    metadata["dc.date"] = self._extract_year_from_pdf(pdf) or "n.d."
                    metadata["dc.type"] = self._detect_document_type(text)
                    
                    if self.use_web_enrichment:
                        metadata = self._enrich_metadata_with_web_search(metadata)
                    if self.use_llm_inference:
                        metadata = self._infer_metadata_with_llm(metadata, text)
                    
                    # Add extracted metadata to BibTeX for future sync
                    self._add_to_bibtex(metadata)
                    self.bib_dict[filename] = {k: v for k, v in metadata.items() if k.startswith('dc.')}
                    if verbose:
                        print(f"Added extracted metadata for {filename} to BibTeX.")
                
                documents.append(Document(page_content=text, metadata=metadata))
                pdf.close()
            except Exception as e:
                self.metadata_fallback_report.append(f"Error loading {filename}: {str(e)}")
                if verbose:
                    print(f"Skipping {filename}: {str(e)}")
        
        return documents

    def _remove_from_bibtex(self, filename):
        """Remove a BibTeX entry for the given filename."""
        if not os.path.exists(self.bibtex_file):
            if self.verbose:
                print(f"BibTeX file '{self.bibtex_file}' not found. No entry to remove for {filename}.")
            return
        
        try:
            with open(self.bibtex_file, encoding="utf-8") as bibtex_file:
                parser = bibtexparser.bparser.BibTexParser(
                    common_strings=True,
                    ignore_nonstandard_types=False,
                    homogenize_fields=False,
                    interpolate_strings=False
                )
                bib_database = bibtexparser.load(bibtex_file, parser=parser)
            
            # Filter out the entry for the given filename
            entries = [entry for entry in bib_database.entries 
                      if entry.get('note', '').replace('Filename: ', '').strip() != filename]
            
            if len(entries) == len(bib_database.entries):
                if self.verbose:
                    print(f"No BibTeX entry found for {filename}.")
                self.metadata_fallback_report.append(f"No BibTeX entry found for {filename}.")
                return
            
            # Write updated BibTeX file
            bib_database.entries = entries
            writer = BibTexWriter()
            with open(self.bibtex_file, 'w', encoding="utf-8") as bibtex_file:
                bibtex_file.write(writer.write(bib_database))
            
            # Update in-memory bib_dict
            self.bib_dict.pop(filename, None)
            self.metadata_fallback_report.append(f"Removed BibTeX entry for {filename}.")
            if self.verbose:
                print(f"Removed BibTeX entry for {filename}.")
        
        except Exception as e:
            self.metadata_fallback_report.append(f"Error removing BibTeX entry for {filename}: {str(e)}")
            if self.verbose:
                print(f"Error removing BibTeX entry for {filename}: {str(e)}")

    def sync_metadata_from_bibtex(self):
        """Sync metadata from references.bib to vector store chunks. Use after manual BibTeX edits."""
        if not self.vector_store:
            message = "Error: Vector store not initialized. Cannot sync metadata."
            self.metadata_fallback_report.append(message)
            if self.verbose:
                print(message)
            return message
        
        self.bib_dict = self._load_bibtex()
        if not self.bib_dict:
            message = "No BibTeX entries found to sync."
            self.metadata_fallback_report.append(message)
            if self.verbose:
                print(message)
            return message
        
        updated_count = 0
        max_batch_size = 5000  # Safe batch size below Chroma's limit of 5461
        
        for filename, new_meta in self.bib_dict.items():
            # Find chunks for this filename
            results = self.vector_store._collection.get(where={"source": filename}, include=["metadatas"])
            ids = results["ids"]
            metadatas = results["metadatas"]
            
            if not ids:
                if self.verbose:
                    print(f"No chunks found for {filename} in vector store.")
                continue
            
            # Update each chunk's metadata in batches
            updated_metadatas = []
            for meta in metadatas:
                # Preserve non-dc fields (e.g., chunk_id, source)
                meta.update({k: v for k, v in new_meta.items() if k.startswith('dc.')})
                updated_metadatas.append(meta)
            
            # Process updates in batches
            try:
                for i in range(0, len(ids), max_batch_size):
                    batch_ids = ids[i:i + max_batch_size]
                    batch_metadatas = updated_metadatas[i:i + max_batch_size]
                    self.vector_store._collection.update(ids=batch_ids, metadatas=batch_metadatas)
                    updated_count += len(batch_ids)
                    if self.verbose:
                        print(f"Updated {len(batch_ids)} chunks for {filename} in batch {i//max_batch_size + 1}")
            except Exception as e:
                self.metadata_fallback_report.append(f"Error updating chunks for {filename}: {str(e)}")
                if self.verbose:
                    print(f"Error updating chunks for {filename}: {str(e)}")
                continue
        
        message = f"Synced metadata for {updated_count} chunks from BibTeX."
        self.metadata_fallback_report.append(message)
        if self.verbose:
            print(message)
        return message

    def delete_document(self, filename=None):
        """
        Delete a specific PDF file and its associated chunks from the vector store and BibTeX.

        Args:
            filename (str): The name of the PDF file to delete (e.g., 'science.aau6170 two-eyed seeing.pdf').

        Returns:
            str: A message indicating success or failure of the deletion.
        """
        if filename is None:
            help_message = (
                "Error: Missing required parameter 'filename' for delete_document.\n"
                "How to Use:\n"
                "- Purpose: Delete a specific PDF file and its associated chunks from the vector store.\n"
                "- Required Parameters:\n"
                "  - filename: A string specifying the name of a PDF file (must end with '.pdf').\n"
                "- Optional Parameters:\n"
                "  - None\n"
                "- Example:\n"
                "```python\n"
                "from local_llm import LocalLLM\n"
                "llm = LocalLLM(pdf_dir='path/to/pdfs')\n"
                "result = llm.delete_document(filename='science.aau6170 two-eyed seeing.pdf')\n"
                "print(result)\n"
                "```"
            )
            print(help_message)
            return help_message
        if not isinstance(filename, str) or not filename.endswith(".pdf"):
            help_message = (
                "Error: Invalid required parameter 'filename' for delete_document.\n"
                "How to Use:\n"
                "- Purpose: Delete a specific PDF file and its associated chunks from the vector store.\n"
                "- Required Parameters:\n"
                "  - filename: A string specifying the name of a PDF file (must end with '.pdf').\n"
                "- Optional Parameters:\n"
                "  - None\n"
                "- Example:\n"
                "```python\n"
                "from local_llm import LocalLLM\n"
                "llm = LocalLLM(pdf_dir='path/to/pdfs')\n"
                "result = llm.delete_document(filename='science.aau6170 two-eyed seeing.pdf')\n"
                "print(result)\n"
                "```"
            )
            print(help_message)
            return help_message
        
        if not self.vector_store:
            error_message = "Error: Vector store not initialized."
            print(error_message)
            return error_message

        # Check if the document exists in the vector store
        results = self.vector_store._collection.get(where={"source": filename}, include=["metadatas"])
        document_ids = results["ids"]

        if not document_ids:
            error_message = f"Error: No chunks found for '{filename}' in the vector store."
            print(error_message)
            return error_message

        # Delete all chunks associated with the filename
        try:
            self.vector_store._collection.delete(ids=document_ids)
            self.metadata_fallback_report.append(
                f"File: {filename} - Successfully deleted {len(document_ids)} chunks from vector store"
            )
            # Remove from BibTeX
            self._remove_from_bibtex(filename)
            success_message = f"Successfully deleted '{filename}' and {len(document_ids)} associated chunks from vector store and BibTeX."
            print(success_message)
            return success_message
        except Exception as e:
            error_message = f"Error: Failed to delete '{filename}' from vector store or BibTeX: {str(e)}"
            self.metadata_fallback_report.append(error_message)
            print(error_message)
            return error_message




    def _hash_file(self, filepath):
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _remove_duplicate_pdfs(self, pdf_dir):
        """Remove duplicate PDFs based on file hash and APA citation (ignoring document type)."""
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDFs found to check for duplicates.")
            return pdf_files
        
        print("Checking for duplicate PDFs by hash...")
        hash_dict = {}
        hash_duplicates = []
        
        for filename in tqdm(pdf_files, desc="Hashing PDFs"):
            filepath = os.path.join(pdf_dir, filename)
            file_hash = self._hash_file(filepath)
            if file_hash in hash_dict:
                hash_duplicates.append(filename)
            else:
                hash_dict[file_hash] = filename
        
        if hash_duplicates:
            print(f"Found {len(hash_duplicates)} hash-based duplicates: {hash_duplicates}")
            confirm = input("Remove hash-based duplicates? (yes/no): ").lower()
            if confirm == "yes":
                for dup in hash_duplicates:
                    os.remove(os.path.join(pdf_dir, dup))
                    self.metadata_fallback_report.append(f"Removed hash duplicate: {dup}")
                    print(f"Removed {dup}")
                    self._remove_from_bibtex(dup)  # Remove duplicates from BibTeX
                pdf_files = [f for f in pdf_files if f not in hash_duplicates]
            else:
                print("Keeping hash-based duplicates.")
        else:
            print("No hash-based duplicates found.")
        
        print("Checking for duplicate APA citations (ignoring document type)...")
        citation_to_files = {}
        for filename in tqdm(pdf_files, desc="Generating APA Citations"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                pdf = fitz.open(filepath)
                text = "".join(page.get_text("text") or "" for page in pdf)
                metadata = {
                    "source": filename,
                    "dc.title": "Untitled Document",
                    "dc.creator": "Unknown Author",
                    "dc.date": "n.d.",
                    "dc.type": self._detect_document_type(text)
                }
                if filename in self.bib_dict:
                    metadata.update(self.bib_dict[filename])
                else:
                    if self.use_llm_inference:
                        metadata = self._infer_metadata_with_llm(metadata, text)
                    if not metadata["dc.title"] or metadata["dc.title"] == "Untitled Document":
                        metadata["dc.title"] = self._extract_title_from_content(text) or self._clean_filename_title(filename)
                    if not metadata["dc.creator"] or metadata["dc.creator"] == "Unknown Author":
                        metadata["dc.creator"] = self._extract_author_from_content(text)
                    year_match = re.search(r"\b(19|20)\d{2}\b", text[:1000])
                    if year_match:
                        metadata["dc.date"] = year_match.group(0)
                    self._add_to_bibtex(metadata)
                    self.bib_dict[filename] = {k: v for k, v in metadata.items() if k.startswith('dc.')}
                citation = self._format_apa_citation(metadata)
                base_citation = re.sub(r'\s*\(\w+\)\s*\.\s*\[Filename:.*\]$', '', citation)
                citation_to_files.setdefault(base_citation, []).append((filename, citation))
                pdf.close()
            except Exception as e:
                self.metadata_fallback_report.append(f"Error processing {filename} for citation: {str(e)}")
                continue
        
        citation_duplicates = {citation: files for citation, files in citation_to_files.items() if len(files) > 1}
        if citation_duplicates:
            print(f"Found {len(citation_duplicates)} citations with multiple files:")
            files_to_remove = []
            for citation, files in citation_duplicates.items():
                filenames = [f[0] for f in files]
                full_citations = [f[1] for f in files]
                print(f"Base Citation: {citation}")
                print(f"Files: {filenames}")
                keep_file = max(filenames, key=lambda f: (
                    1 if re.search(r'guidance|best practices|traditional|noaa', f.lower()) else 0,
                    -len(f)  # Prefer shorter filenames
                ))
                files_to_remove.extend(f for f in filenames if f != keep_file)
                print(f"Keeping: {keep_file}, Removing: {[f for f in filenames if f != keep_file]}")
            
            if files_to_remove:
                confirm = input(f"Remove {len(files_to_remove)} duplicate files based on APA citations? (yes/no): ").lower()
                if confirm == "yes":
                    for dup in files_to_remove:
                        filepath = os.path.join(pdf_dir, dup)
                        try:
                            os.remove(filepath)
                            self.metadata_fallback_report.append(f"Removed APA citation duplicate: {dup}")
                            print(f"Removed {dup}")
                            self._remove_from_bibtex(dup)  # Remove from BibTeX
                            if self.vector_store:
                                self.vector_store._collection.delete(where={"source": dup})
                                self.metadata_fallback_report.append(f"Removed chunks for {dup} from vector store")
                        except Exception as e:
                            self.metadata_fallback_report.append(f"Error removing {dup}: {str(e)}")
                    pdf_files = [f for f in pdf_files if f not in files_to_remove]
                else:
                    print("Keeping APA citation duplicates.")
        else:
            print("No APA citation duplicates found.")
        
        return pdf_files

    def _extract_metadata_from_filename(self, filename):
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split("_")
        author = parts[0] if parts else "Unknown Author"
        year = None
        for part in parts:
            if re.match(r"\d{4}", part):
                year = part
                break
        return author, year if year else "n.d.", base_name

    def _clean_filename_title(self, filename):
        """Clean filename to extract a temporary title, handling resolutions and journal articles."""
        base_name = os.path.splitext(filename)[0]
        if re.match(r'^ILL\d+', base_name):
            return f"Document {base_name}"
        if re.search(r'REN-\d+-\d+', base_name):
            return "NCAI Resolution {}".format(re.search(r'REN-\d+-\d+', base_name).group(0))
        # Improved cleaning for journal articles
        cleaned = re.sub(r'^(science|nature|journal)\.\w+', '', base_name, flags=re.IGNORECASE)  # Remove journal prefixes
        cleaned = re.sub(r'[-_]', ' ', cleaned).strip()  # Replace delimiters with spaces
        cleaned = re.sub(r'\s*(et al\.|[\d]{4}|vol\.|no\.|pp\.).*', '', cleaned, flags=re.IGNORECASE)  # Remove suffixes
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize spaces
        cleaned = cleaned[:150] if cleaned else None
        if cleaned and not re.search(r'\b(introduction|chapter|abstract|keywords|united states|considering|sciencemag|nature|journal)\b', cleaned.lower()) and len(cleaned) > 10:
            return cleaned.title()  # Capitalize words
        return f"Document {base_name}"

    def _extract_title_from_content(self, text):
        """Extract a concise title from the entire document, prioritizing ITEK-related phrases and filename hints."""
        # Split text into sections (approximating articles) by headers or page breaks
        sections = re.split(r'\n\s*\n|INSIGHTS\s*\|\s*PERSPECTIVES|RESEARCH\s*ARTICLE|REVIEW|COMMENTARY', text, flags=re.IGNORECASE)
        valid_titles = []
        
        # ITEK-related keywords for prioritization
        itek_keywords = [
            "two-eyed seeing", "indigenous knowledge", "traditional ecological knowledge",
            "wildlife health", "participatory epidemiology", "inuit knowledge", "mi'kmaq principle"
        ]
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lines = section.split("\n")[:50]  # Limit to first 50 lines of each section
            
            # Check for section headers
            for i, line in enumerate(lines):
                line = line.strip()
                if re.match(r'^(INSIGHTS|PERSPECTIVES|RESEARCH ARTICLE|REVIEW|COMMENTARY)\b', line, re.IGNORECASE):
                    next_lines = lines[i+1:i+3]
                    for next_line in next_lines:
                        next_line = next_line.strip()
                        if (10 < len(next_line) < 150 and 
                            not re.search(r'\b(introduction|chapter|abstract|keywords|university|department|jstor|united states|considering|sciencemag)\b', next_line.lower())):
                            valid_titles.append(next_line[:150])
                
                # Check for ITEK-related phrases
                for keyword in itek_keywords:
                    if keyword.lower() in line.lower():
                        phrase = line[:150]
                        if (10 < len(phrase) < 150 and 
                            not re.search(r'\b(introduction|chapter|abstract|keywords|university|department|jstor|united states|considering|sciencemag)\b', phrase.lower())):
                            valid_titles.append(phrase)
            
            # Check for prominent phrases in section
            for line in lines:
                line = line.strip()
                if (10 < len(line) < 150 and 
                    not line.lower().startswith(("abstract", "keywords", "http", "www", "doi:", "©", "copyright", "introduction", "chapter", "executive committee")) and
                    not re.match(r"^\d+\s", line) and
                    not re.search(r"\b(University|Department|Association|JSTOR|EXECUTIVE COMMITTEE|sciencemag)\b", line, re.IGNORECASE)):
                    if (sum(1 for c in line if c.isupper()) > 2 or 
                        re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", line)):
                        valid_titles.append(line[:150])
        
        # Filter and rank titles
        if valid_titles:
            # Prioritize titles with ITEK keywords
            ranked_titles = sorted(
                valid_titles,
                key=lambda t: sum(1 for k in itek_keywords if k.lower() in t.lower()) * 100 + len(t),
                reverse=True
            )
            return ranked_titles[0]
        
        # Fallback to resolution numbers
        resolution_match = re.search(r'REN-\d+-\d+', text)
        if resolution_match:
            resolution_num = resolution_match.group(0)
            content_snippet = text[:2000].lower()
            for phrase in ["traditional knowledge", "climate change", "tribal rights", "federal guidance"]:
                if phrase in content_snippet:
                    return f"NCAI Resolution {resolution_num}: {phrase.title()}".replace(" And ", " and ")[:150]
            return f"NCAI Resolution {resolution_num}"[:150]
        
        # Fallback to generic title extraction
        valid_lines = [line.strip() for line in text.split("\n") if 
                       10 < len(line.strip()) < 150 and 
                       not re.search(r"\b(introduction|chapter|abstract|keywords|university|department|jstor|united states|considering|sciencemag)\b", line.lower())]
        if valid_lines:
            return max(valid_lines, key=len)[:150]
        return None

    def _extract_author_from_content(self, text):
        """Extract authors from the entire document, prioritizing individual authors over organizations."""
        sections = re.split(r'\n\s*\n|INSIGHTS\s*\|\s*PERSPECTIVES|RESEARCH\s*ARTICLE|REVIEW|COMMENTARY', text, flags=re.IGNORECASE)
        authors = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lines = section.split("\n")[:50]
            
            # Check for explicit author lines
            for i, line in enumerate(lines):
                line = line.strip()
                if re.match(r'^Author\(s\):|^by\s+', line, re.IGNORECASE):
                    author_line = line.split(":", 1)[1].strip() if ":" in line else line.split("by ", 1)[1].strip()
                    author_line = re.sub(r'[^A-Za-z\'\s,.&]', '', author_line)
                    author_parts = re.split(r",| and | & |;|\n", author_line)
                    for part in author_parts:
                        part = part.strip()
                        if re.match(r"[A-Za-z'\-]+,?\s+[A-Z]\.?(?:\s+[A-Za-z'\-]+)?", part):
                            authors.append(part)
                    if authors:
                        break
            
            # Check for bylines in journal articles
            if not authors:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if re.search(r'\b[A-Z][a-z]+,\s*[A-Z]\.\s*(?:[A-Z][a-z]+,\s*[A-Z]\.\s*)?(?:and|&)?\s*[A-Z][a-z]+,\s*[A-Z]\.', line):
                        author_parts = re.split(r",| and | & ", line)
                        for part in author_parts:
                            part = part.strip()
                            if re.match(r"[A-Za-z'\-]+,?\s+[A-Z]\.?(?:\s+[A-Za-z'\-]+)?", part):
                                authors.append(part)
                        if authors:
                            break
            
            # Fallback to reference section
            if not authors:
                ref_matches = re.findall(r'([A-Z][a-z]+(?:\s[A-Z]\.)+)(?:\s+et al\.)?\s*\([^0-9]+\d{4}\)', section)
                if ref_matches:
                    authors = list(dict.fromkeys(ref_matches))[:10]  # Limit to 10 unique authors
                    break
        
        # Check for organizational authors only if no individual authors are found
        if not authors:
            org_patterns = [
                r"National Congress of American Indians",
                r"USDA\b",
                r"NRCS\b",
                r"Inuit Circumpolar Council",
                r"Karuk Tribe",
                r"NOAA\b",
                r"NOAA Fisheries and National Ocean Service",
                r"Assembly of Alaska Native Educators",
                r"American Association for the Advancement of Science",
                r"AAAS\b"
            ]
            for pattern in org_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return re.search(pattern, text, re.IGNORECASE).group(0)
        
        if not authors:
            return "Unknown Author"
        elif len(authors) == 1:
            return authors[0]
        elif len(authors) <= 20:
            return ", ".join(authors[:-1]) + ", & " + authors[-1]
        else:
            return ", ".join(authors[:19]) + ", …, & " + authors[-1]

    def _detect_document_type(self, text):
        """Detect document type, prioritizing Articles for journal content."""
        text_snippet = " ".join(text.split()[:1000]).lower()
        if re.search(r"\b(journal|volume|issue|doi:|science, technology|new england journal|sciencemag|nature|published by aaas)\b", text_snippet):
            return "Article"
        elif re.search(r"\b(whereas|be it resolved|resolution \w+-\d+-\d+)\b", text_snippet):
            return "Resolution"
        elif re.search(r"\b(guidance|best practices)\b", text_snippet) and re.search(r"\b(noaa|national ocean service)\b", text_snippet):
            return "Guidelines"
        elif re.search(r"\b(report|prepared by|submitted to|committee|advisory|usda|nrcs|fws|noaa|epa)\b", text_snippet):
            return "Report"
        elif re.search(r"\b(handbook|guidelines|factsheet)\b", text_snippet):
            return "Handbook" if "handbook" in text_snippet else "Factsheet" if "factsheet" in text_snippet else "Guidelines"
        elif re.search(r"\b(letter|memo|to:|from:)\b", text_snippet):
            return "Letter"
        elif re.search(r"\b(dissertation|thesis)\b", text_snippet):
            return "Dissertation or Thesis"
        elif re.search(r"\b(website|html|url|accessed \d{1,2}\s+[a-z]+\s+\d{4})\b", text_snippet):
            return "Website"
        return "Document"

    def _enrich_metadata_from_web(self, metadata, content):
        """Enrich metadata using Crossref or OpenAlex API, with title fallback if no DOI."""
        if not self.use_web_enrichment:
            return metadata
        
        doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        doi_match = re.search(doi_pattern, content[:1000], re.IGNORECASE)
        query = {}
        query_type = ""
        
        if doi_match:
            query = {"doi": doi_match.group(0)}
            query_type = "DOI"
        elif metadata["dc.title"] and metadata["dc.title"] != "Untitled Document":
            query = {"filter": f"title.search:{metadata['dc.title']}"}
            query_type = "Title"
        else:
            self.metadata_fallback_report.append(
                f"File: {metadata['source']} - Web enrichment skipped (no DOI or valid title)"
            )
            return metadata
        
        local_title = metadata.get("dc.title", "").lower()
        
        for attempt in range(5):
            try:
                time.sleep(0.2 * (2 ** attempt))
                if query_type == "DOI":
                    response = requests.get("https://api.crossref.org/works", params=query, timeout=10)
                else:
                    response = requests.get("https://api.openalex.org/works", params=query, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("message", {}).get("items", []) if query_type == "DOI" else data.get("results", [])
                    if items:
                        item = items[0]
                        api_title = item.get("title", [""])[0].lower() if query_type == "DOI" else item.get("display_name", "").lower()
                        if fuzz.token_sort_ratio(local_title, api_title) < 80 and local_title != "untitled document":
                            self.metadata_fallback_report.append(
                                f"File: {metadata['source']} - Web enrichment rejected: API title '{api_title[:50]}' too dissimilar to local '{local_title[:50]}'"
                            )
                            return metadata
                        enriched = metadata.copy()
                        enriched["dc.title"] = item.get("title", [metadata["dc.title"]])[0][:150] if query_type == "DOI" else item.get("display_name", metadata["dc.title"])[:150]
                        authors = []
                        if query_type == "DOI":
                            authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in item.get("author", []) if a.get('family')]
                        else:
                            authors = [f"{a.get('author', {}).get('display_name', '')}" for a in item.get("authorships", [])]
                        enriched["dc.creator"] = "Unknown Author" if not authors else (
                            authors[0] if len(authors) == 1 else
                            ", ".join(authors[:-1]) + ", & " + authors[-1] if len(authors) <= 20 else
                            ", ".join(authors[:19]) + ", …, & " + authors[-1]
                        )
                        enriched["dc.date"] = (
                            item.get("published-print", {}).get("date-parts", [[metadata["dc.date"]]])[0][0] if query_type == "DOI"
                            else item.get("publication_year", metadata["dc.date"])
                        )
                        enriched["dc.type"] = item.get("type", metadata["dc.type"]).replace("-", " ").title() if query_type == "DOI" else item.get("type", metadata["dc.type"]).title()
                        if enriched["dc.type"].lower() in ["journal-article", "article"]:
                            enriched["dc.type"] = "Article"
                        self.metadata_fallback_report.append(
                            f"File: {metadata['source']} - Web enrichment successful via {query_type}"
                        )
                        return enriched
                    else:
                        self.metadata_fallback_report.append(
                            f"File: {metadata['source']} - Web enrichment failed: No results from {query_type} query"
                        )
                else:
                    self.metadata_fallback_report.append(
                        f"File: {metadata['source']} - Web enrichment failed: HTTP {response.status_code} ({query_type})"
                    )
            except Exception as e:
                self.metadata_fallback_report.append(
                    f"File: {metadata['source']} - Web enrichment failed: {str(e)} ({query_type})"
                )
        
        return metadata

    def _infer_metadata_with_llm(self, metadata, content, verbose=False):
        """Use LLM to extract metadata, focusing on relevant article section."""
        cleaned_content = re.sub(r'\s+', ' ', content[:2000]).strip()  # Increased to 2000 chars
        cleaned_content = re.sub(r'[^\x20-\x7E]', '', cleaned_content)
        
        filename = metadata["source"]
        filename_hint = self._clean_filename_title(filename).lower()
        
        prompt = (
            "Output ONLY a JSON object with the following structure:\n"
            "{\n"
            "  \"title\": \"<main title of the document, 10-150 chars, from relevant section>\",\n"
            "  \"authors\": \"<author1, author2, & author3 or organization name or Unknown Author, APA format, max 100 chars, from relevant section>\",\n"
            "  \"year\": \"<year, 1900-2025 or n.d., from publication date>\",\n"
            "  \"document_type\": \"<Article|Report|Resolution|Letter|Dissertation or Thesis|Handbook|Factsheet|Guidelines|Website|Document>\"\n"
            "}\n"
            "Extract metadata DIRECTLY from the content below, focusing on the article section related to Indigenous Traditional Ecological Knowledge (ITEK) or matching the filename hint. "
            "Ignore unrelated sections (e.g., previous articles or references about unrelated topics like cancer). "
            "For journal articles, prioritize author bylines (e.g., 'Author(s):', 'by') and titles from section headers (e.g., 'INSIGHTS | PERSPECTIVES') or prominent ITEK-related phrases (e.g., 'Two-Eyed Seeing'). "
            "For resolutions, use the resolution number (e.g., 'REN-13-035') and key topic. "
            "Titles must be 10-150 characters, descriptive, and exclude 'chapter', 'introduction', 'abstract', 'keywords', 'united states', 'considering', or 'sciencemag'. "
            "Authors can be individual names (e.g., 'Kutz, S.' from bylines or references), organizations (e.g., 'AAAS'), or 'Unknown Author' if none found. "
            "Prioritize individual authors over organizations unless explicitly stated (e.g., 'Published by AAAS'). "
            "Use 'n.d.' if no year is found. Prefer publication dates (e.g., '21 JUNE 2019') over reference years (e.g., '2012'). "
            "Set document_type to 'Resolution' for 'WHEREAS', 'BE IT RESOLVED', or 'REN-*-*' patterns. "
            "Set document_type to 'Guidelines' for NOAA documents with 'guidance' or 'best practices'. "
            "Set document_type to 'Article' for journal publications (e.g., 'sciencemag', 'Published by AAAS'). "
            "Set document_type to 'Website' for content with URLs or 'accessed' dates.\n"
            f"Filename hint: {filename_hint}\n"
            f"Content: {cleaned_content}\n"
            "Do not include any text outside the JSON object."
        )
        
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                inferred = json.loads(response)
                if not isinstance(inferred, dict):
                    raise ValueError("Response is not a JSON object")
                if not all(k in inferred for k in ["title", "authors", "year", "document_type"]):
                    raise ValueError("Missing required JSON fields")
                
                if inferred["year"] != "n.d." and not re.match(r"^(19|20)\d{2}$", str(inferred["year"])):
                    inferred["year"] = "n.d."
                
                if inferred["document_type"] not in ["Article", "Report", "Resolution", "Letter", "Dissertation or Thesis", "Handbook", "Factsheet", "Guidelines", "Website", "Document"]:
                    inferred["document_type"] = "Document"
                
                if (not inferred["title"] or len(inferred["title"]) < 10 or len(inferred["title"]) > 150 or
                    re.search(r"\b(introduction|chapter|abstract|keywords|united states|considering|executive committee|sciencemag)\b", inferred["title"].lower())):
                    raise ValueError(f"Invalid title: {inferred['title']}")
                
                if len(inferred["authors"]) > 100 or inferred["authors"].lower() == inferred["title"].lower():
                    inferred["authors"] = "Unknown Author"
                
                title_snippet = inferred["title"].lower()[:50]
                if title_snippet not in cleaned_content.lower() and title_snippet != "untitled document" and title_snippet not in filename_hint:
                    raise ValueError(f"Title '{title_snippet}' not found in content or filename hint")
                if inferred["authors"] != "Unknown Author" and not any(name.lower() in cleaned_content.lower() for name in inferred["authors"].split(", ")):
                    inferred["authors"] = "Unknown Author"
                
                enriched = metadata.copy()
                enriched["dc.title"] = inferred["title"]
                enriched["dc.creator"] = inferred["authors"]
                enriched["dc.date"] = inferred["year"]
                enriched["dc.type"] = inferred["document_type"]
                self.metadata_fallback_report.append(
                    f"File: {metadata['source']} - LLM inference successful: {json.dumps(inferred)}"
                )
                return enriched
            except Exception as e:
                self.metadata_fallback_report.append(
                    f"File: {metadata['source']} - LLM inference failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}, response: {response[:100]}"
                )
                if attempt == max_retries:
                    return metadata
                time.sleep(1)            
   
    def add_new_files(self, pdf_dir=None, verbose=False):
        # Validate required parameter: pdf_dir
        if pdf_dir is None:
            help_message = (
                "Error: No arguments provided or missing required parameter 'pdf_dir' for add_new_files.\n"
                "How to Use:\n"
                "- Purpose: Add new PDF files from a directory to the vector store, skipping duplicates.\n"
                "- Required Parameters:\n"
                "  - pdf_dir: A string specifying the path to a directory containing PDF files.\n"
                "- Optional Parameters:\n"
                "  - verbose: Enable detailed output during processing (default: False).\n"
                "- Example:\n"
                "```python\n"
                "from local_llm import LocalLLM\n"
                "llm = LocalLLM(pdf_dir='path/to/pdfs')\n"
                "llm.add_new_files(pdf_dir='path/to/new_pdfs', verbose=True)\n"
                "```"
            )
            print(help_message)
            raise ValueError("Missing pdf_dir: Must provide a valid directory path.")
        if not isinstance(pdf_dir, str) or not os.path.isdir(pdf_dir):
            help_message = (
                "Error: Invalid required parameter 'pdf_dir' for add_new_files.\n"
                "How to Use:\n"
                "- Purpose: Add new PDF files from a directory to the vector store, skipping duplicates.\n"
                "- Required Parameters:\n"
                "  - pdf_dir: A string specifying the path to a directory containing PDF files.\n"
                "- Optional Parameters:\n"
                "  - verbose: Enable detailed output during processing (default: False).\n"
                "- Example:\n"
                "```python\n"
                "from local_llm import LocalLLM\n"
                "llm = LocalLLM(pdf_dir='path/to/pdfs')\n"
                "llm.add_new_files(pdf_dir='path/to/new_pdfs', verbose=True)\n"
                "```"
            )
            print(help_message)
            raise ValueError("Invalid pdf_dir: Must be a valid directory path.")
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        
        if not os.path.exists(pdf_dir):
            print(f"Error: Directory '{pdf_dir}' not found.")
            return
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"No PDFs found in {pdf_dir}.")
            return
        
        # Check existing filenames in vector store
        metadatas = self.vector_store._collection.get(include=["metadatas"])["metadatas"]
        existing_sources = {meta["source"] for meta in metadatas}
        
        # Filter out files with exact matching filenames
        new_files = [f for f in pdf_files if f not in existing_sources]
        if not new_files:
            print(f"No new files detected in {pdf_dir} after filtering existing filenames.")
            return
        
        print(f"Checking {len(new_files)} new files for duplicates: {new_files}")
        existing_citations = {re.sub(r'\s*\(\w+\)\s*\.\s*\[Filename:.*\]$', '', self._format_apa_citation(meta)) for meta in metadatas}
        new_documents = []
        metadata_counts = {"author": 0, "title": 0, "year": 0, "regex_fallback": 0}
        
        for filename in tqdm(new_files, desc="Loading New PDFs"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                pdf = fitz.open(filepath)
                text = ""
                for page_num, page in enumerate(pdf):
                    text += page.get_text("text") or ""
                
                title = None
                authors = "Unknown Author"
                year = "n.d."
                doc_type = self._detect_document_type(text)
                used_file_metadata = []

                metadata = {
                    "source": filename,
                    "dc.title": "Untitled Document",
                    "dc.creator": "Unknown Author",
                    "dc.date": "n.d.",
                    "dc.type": doc_type
                }
                if self.use_llm_inference:
                    metadata = self._infer_metadata_with_llm(metadata, text, verbose=verbose)
                
                title = metadata["dc.title"]
                authors = metadata["dc.creator"]
                year = metadata["dc.date"]
                doc_type = metadata["dc.type"]

                if not title or title == "Untitled Document" or len(title) < 10 or re.search(r"\b(introduction|chapter|abstract|keywords|united states|considering|executive committee|sciencemag)\b", title.lower()):
                    title = self._extract_title_from_content(text)
                    used_file_metadata.append("title (regex)")
                    metadata_counts["title"] += 1
                    metadata_counts["regex_fallback"] += 1
                
                if not authors or authors == "Unknown Author":
                    authors = self._extract_author_from_content(text)
                    used_file_metadata.append("author (regex)")
                    metadata_counts["author"] += 1
                    metadata_counts["regex_fallback"] += 1
                
                # Improved year extraction
                year_match = re.search(r'\b(\d{1,2}\s+[A-Z][a-z]+\s+(20\d{2}))\b', text) or \
                             re.search(r'\bVOL\s+\d+\s+ISSUE\s+\d+\s+\(\d{1,2}\s+[A-Z][a-z]+\s+(20\d{2})\)', text, re.IGNORECASE)
                if year_match:
                    year = year_match.group(1 if year_match.lastindex == 1 else 2)
                else:
                    year_match = re.search(r"\b(20\d{2})\b", text)
                    if year_match:
                        year = year_match.group(0)
                    else:
                        year_from_file = self._extract_metadata_from_filename(filename)[1]
                        year = year_from_file
                        if year == "n.d.":
                            used_file_metadata.append("year")
                            metadata_counts["year"] += 1

                file_author, _, file_title = self._extract_metadata_from_filename(filename)
                if not title or title.strip() == "" or len(title) < 10 or re.search(r'\b(introduction|chapter|abstract|keywords|united states|considering|executive committee|sciencemag)\b', title.lower()):
                    title = self._clean_filename_title(filename)
                    used_file_metadata.append("title (filename)")
                    metadata_counts["title"] += 1
                    metadata_counts["regex_fallback"] += 1

                if len(authors) > 100 or authors.lower() == title.lower():
                    self.metadata_fallback_report.append(
                        f"File: {filename} - Invalid author detected: {authors[:50]}...; using Unknown Author"
                    )
                    authors = "Unknown Author"
                    used_file_metadata.append("author (corrected)")
                    metadata_counts["author"] += 1

                metadata = {
                    "source": filename,
                    "dc.title": title,
                    "dc.creator": authors,
                    "dc.date": year,
                    "dc.type": doc_type
                }

                if verbose:
                    print(f"Processing {filename} with metadata: {metadata}")

                new_citation = re.sub(r'\s*\(\w+\)\s*\.\s*\[Filename:.*\]$', '', self._format_apa_citation(metadata))
                is_duplicate = any(fuzz.ratio(new_citation.lower(), existing.lower()) >= 95 for existing in existing_citations)
                if is_duplicate:
                    self.metadata_fallback_report.append(f"Rejected {filename}: Duplicate APA citation detected")
                    print(f"Rejected {filename}: Duplicate APA citation")
                    pdf.close()
                    continue

                metadata = self._enrich_metadata_from_web(metadata, text)

                if len(metadata["dc.title"]) > 150 or re.search(r'\b(introduction|chapter|abstract|keywords|united states|considering|executive committee|sciencemag)\b', metadata["dc.title"].lower()) or len(metadata["dc.title"]) < 10:
                    self.metadata_fallback_report.append(
                        f"File: {filename} - Post-enrichment invalid title: {metadata['dc.title'][:50]}...; using filename title"
                    )
                    metadata["dc.title"] = self._clean_filename_title(filename)
                    used_file_metadata.append("title (post-enrichment)")
                    metadata_counts["title"] += 1

                if not metadata["dc.title"] or metadata["dc.title"].strip() == "":
                    self.metadata_fallback_report.append(
                        f"File: {filename} - Missing title; using default"
                    )
                    metadata["dc.title"] = "Untitled Document"

                if len(metadata["dc.creator"]) > 100 or metadata["dc.creator"].lower() == metadata["dc.title"].lower():
                    self.metadata_fallback_report.append(
                        f"File: {filename} - Post-enrichment invalid author: {metadata['dc.creator'][:50]}...; using Unknown Author"
                    )
                    metadata["dc.creator"] = "Unknown Author"
                    used_file_metadata.append("author (post-enrichment)")
                    metadata_counts["author"] += 1

                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                new_documents.append(doc)
                pdf.close()
            except Exception as e:
                print(f"Error: Failed to process {filename}: {str(e)}")
                continue
        
        if new_documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            print("Splitting new documents into chunks...")
            new_chunks = text_splitter.split_documents(new_documents)
            print("Adding chunks to vector store...")
            batch_size = 100
            for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding and Indexing New Chunks"):
                batch = new_chunks[i:i + batch_size]
                for j, doc in enumerate(batch):
                    doc.metadata["chunk_id"] = f"{doc.metadata['source']}_chunk_{i+j}"
                    if verbose:
                        print(f"Adding chunk {doc.metadata['chunk_id']} with metadata: {doc.metadata}")
                self.vector_store.add_documents(batch)
            print(f"Added {len(new_documents)} new documents to the vector store.")
        
        if self.metadata_fallback_report and verbose:
            print("\nMetadata Fallback Report for New Files:")
            for entry in self.metadata_fallback_report[-len(new_files):]:
                print(entry)
            print(f"Total files using file metadata: {len(self.metadata_fallback_report[-len(new_files):])}")
            print(f" - Files using metadata for author: {metadata_counts['author']}")
            print(f" - Files using metadata for title: {metadata_counts['title']}")
            print(f" - Files using metadata for year: {metadata_counts['year']}")
            print(f" - Files using regex fallback: {metadata_counts['regex_fallback']}")
        
       
    def migrate_to_bibtex(self):
        """
        Migrate references from vector store metadata to a BibTeX file, validating each entry.

        Returns:
            dict: Summary of migration (total entries, valid entries, skipped entries, and report).
        """
        if not self.vector_store:
            error_message = "Error: No vector store available for migration."
            self.metadata_fallback_report.append(error_message)
            if self.verbose:
                print(error_message)
            return {"total": 0, "valid": 0, "skipped": 0, "report": [error_message]}

        metadatas = self.vector_store._collection.get(include=["metadatas"])["metadatas"]
        source_to_metadata = {}
        for meta in metadatas:
            source = meta.get("source", "")
            if source and source.endswith(".pdf") and source not in source_to_metadata:
                source_to_metadata[source] = meta

        bib_database = bibtexparser.bibdatabase.BibDatabase()
        existing_keys = set()
        skipped_entries = []

        for source, meta in source_to_metadata.items():
            try:
                entry_str, key = self._generate_bibtex_entry(meta, existing_keys)
                temp_parser = bibtexparser.bparser.BibTexParser(
                    common_strings=True,
                    ignore_nonstandard_types=False,
                    homogenize_fields=False,  # Avoid field normalization issues
                    interpolate_strings=False
                )
                parsed = bibtexparser.loads(entry_str, parser=temp_parser)
                if parsed.entries and parsed.entries[0].get('ID') and parsed.entries[0].get('ENTRYTYPE'):
                    bib_database.entries.append(parsed.entries[0])
                    existing_keys.add(key)
                    if self.verbose:
                        print(f"Added valid entry for {source} with key {key}")
                else:
                    skipped_entries.append(f"Invalid entry for {source}: Missing ID or ENTRYTYPE")
                    if self.verbose:
                        print(f"Warning: Skipped invalid entry for {source}")
            except Exception as e:
                skipped_entries.append(f"Error generating/validating entry for {source}: {str(e)}")
                if self.verbose:
                    print(f"Error for {source}: {str(e)}")

        writer = BibTexWriter()
        with open(self.bibtex_file, 'w', encoding='utf-8') as f:
            f.write(writer.write(bib_database))

        total = len(source_to_metadata)
        valid = len(bib_database.entries)
        skipped = total - valid
        report = self.metadata_fallback_report + skipped_entries
        if skipped > 0:
            report.append(f"Warning: Skipped {skipped} entries during migration. Check metadata_fallback_report for details.")
            if self.verbose:
                print(f"Warning: Skipped {skipped} entries out of {total}. See metadata_fallback_report.")
        if self.verbose:
            print(f"Migration complete: Wrote {valid} valid entries to {self.bibtex_file}")

        return {"total": total, "valid": valid, "skipped": skipped, "report": report}
           
    def check_file_in_database(self, filename=None, title=None, similarity_threshold=80):
        """
        Check if a file or title is in the vector store or has a similar citation or filename, prioritizing BibTeX if available.

        Args:
            filename (str, optional): The name of the PDF file to check (e.g., 'R47563.pdf').
            title (str, optional): The title of the paper to check (e.g., 'Tribal Co-management of Federal Lands').
            similarity_threshold (int): Similarity threshold for citation or title comparison (default: 80).

        Returns:
            dict: A dictionary with the following keys:
                - exists: Boolean indicating if the exact filename or title is in the vector store.
                - similar_files: List of filenames with exact matches or similar APA citations/titles.
                - similar_citations: List of tuples (filename, citation, similarity_score) for similar citations/titles.
                - message: Descriptive message summarizing the findings.
        """
        # Validate required parameters: at least one of filename or title
        if filename is None and title is None:
            help_message = (
                "Error: Missing required parameter 'filename' or 'title' for check_file_in_database.\n"
                "How to Use:\n"
                "- Purpose: Check if a PDF file or paper title exists in the vector store or has similar citations/titles.\n"
                "- Required Parameters (at least one):\n"
                "  - filename: A string specifying the name of a PDF file (must end with '.pdf').\n"
                "  - title: A string specifying the title of the paper (full or partial).\n"
                "- Optional Parameters:\n"
                "  - similarity_threshold: Similarity threshold for citation or title comparison (default: 80).\n"
                "- Example:\n"
                "```python\n"
                "from local_llm import LocalLLM\n"
                "llm = LocalLLM(pdf_dir='path/to/pdfs')\n"
                "result = llm.check_file_in_database(filename='R47563.pdf')\n"
                "result = llm.check_file_in_database(title='Tribal Co-management of Federal Lands')\n"
                "print(result['message'])\n"
                "```"
            )
            print(help_message)
            return {
                "exists": False,
                "similar_files": [],
                "similar_citations": [],
                "message": help_message
            }

        if filename and (not isinstance(filename, str) or not filename.endswith(".pdf")):
            help_message = (
                "Error: Invalid 'filename' parameter for check_file_in_database.\n"
                "Filename must be a string ending with '.pdf'.\n"
                "Example: llm.check_file_in_database(filename='R47563.pdf')"
            )
            print(help_message)
            return {
                "exists": False,
                "similar_files": [],
                "similar_citations": [],
                "message": help_message
            }

        if title and (not isinstance(title, str) or not title.strip()):
            help_message = (
                "Error: Invalid 'title' parameter for check_file_in_database.\n"
                "Title must be a non-empty string.\n"
                "Example: llm.check_file_in_database(title='Tribal Co-management of Federal Lands')"
            )
            print(help_message)
            return {
                "exists": False,
                "similar_files": [],
                "similar_citations": [],
                "message": help_message
            }

        if not self.vector_store:
            error_message = "Error: Vector store not initialized."
            print(error_message)
            return {
                "exists": False,
                "similar_files": [],
                "similar_citations": [],
                "message": error_message
            }

        # Helper function to normalize strings (remove extra spaces, qualifiers, case-insensitive)
        def normalize_string(s):
            if not s:
                return ""
            s = re.sub(r'\s*\([^)]+\)\s*$', '', s)  # Remove trailing qualifiers like (Article)
            s = re.sub(r'\s+', ' ', s.strip())  # Normalize spaces
            return s.lower()

        # Prioritize BibTeX if available
        existing_sources = {}
        existing_citations = {}
        existing_titles = {}
        if os.path.exists(self.bibtex_file):
            try:
                with open(self.bibtex_file, encoding="utf-8") as bibtex_file:
                    bib_database = bibtexparser.load(bibtex_file)
                    for entry in bib_database.entries:
                        filename_from_note = entry.get('note', '').replace('Filename: ', '') if 'note' in entry else 'Unknown File'
                        existing_sources[normalize_string(filename_from_note)] = filename_from_note
                        citation = f"{entry.get('author', 'Unknown Author')}. ({entry.get('year', 'n.d.')}). {entry.get('title', 'Untitled Document')}"
                        existing_citations[filename_from_note] = citation
                        existing_titles[filename_from_note] = normalize_string(entry.get('title', ''))
            except UnicodeDecodeError as e:
                self.metadata_fallback_report.append(f"Error reading BibTeX file: {str(e)}")
                if self.verbose:
                    print(f"Error reading BibTeX file: {str(e)}. Falling back to vector store.")

        # Fallback to vector store metadata if BibTeX not fully populated or missing
        if not existing_sources:
            metadatas = self.vector_store._collection.get(include=["metadatas"])["metadatas"]
            existing_sources = {normalize_string(meta["source"]): meta["source"] for meta in metadatas if "source" in meta}
            existing_citations = {
                meta["source"]: re.sub(r'\s*\(\w+\)\s*\.\s*\[Filename:.*\]$', '', self._format_apa_citation(meta))
                for meta in metadatas if "source" in meta
            }
            existing_titles = {
                meta["source"]: normalize_string(meta["dc.title"])
                for meta in metadatas if "dc.title" in meta and meta["dc.title"]
            }

        # Initialize result variables
        file_exists = False
        title_exists = False
        similar_files = []
        similar_citations = []

        # Check for exact filename match
        if filename:
            normalized_filename = normalize_string(filename)
            if normalized_filename in existing_sources:
                file_exists = True
                original_filename = existing_sources[normalized_filename]
                similar_files.append(original_filename)
                # Add the citation for the matched file
                citation = existing_citations.get(original_filename, "Citation not available")
                similar_citations.append((original_filename, citation, 100))

        # Check for exact or near-exact title match
        if title:
            normalized_title = normalize_string(title)
            for source, stored_title in existing_titles.items():
                if normalized_title == stored_title:
                    title_exists = True
                    if source not in similar_files:
                        similar_files.append(source)
                        citation = existing_citations.get(source, "Citation not available")
                        similar_citations.append((source, citation, 100))

        # Generate APA citation for the input filename (if provided and exists in pdf_dir)
        generated_citation = None
        if filename and not file_exists:
            filepath = os.path.join(self.pdf_dir, filename)
            if os.path.exists(filepath):
                try:
                    pdf = fitz.open(filepath)
                    text = "".join(page.get_text("text") or "" for page in pdf)
                    metadata = {
                        "source": filename,
                        "dc.title": "Untitled Document",
                        "dc.creator": "Unknown Author",
                        "dc.date": "n.d.",
                        "dc.type": self._detect_document_type(text)
                    }
                    if self.use_llm_inference:
                        metadata = self._infer_metadata_with_llm(metadata, text)
                    if not metadata["dc.title"] or metadata["dc.title"] == "Untitled Document":
                        metadata["dc.title"] = self._extract_title_from_content(text) or self._clean_filename_title(filename)
                    if not metadata["dc.creator"] or metadata["dc.creator"] == "Unknown Author":
                        metadata["dc.creator"] = self._extract_author_from_content(text)
                    year_match = re.search(r"\b(19|20)\d{2}\b", text[:1000])
                    if year_match:
                        metadata["dc.date"] = year_match.group(0)
                    generated_citation = re.sub(r'\s*\(\w+\)\s*\.\s*\[Filename:.*\]$', '', self._format_apa_citation(metadata))
                    pdf.close()
                except Exception as e:
                    self.metadata_fallback_report.append(f"File: {filename} - Citation generation failed: {str(e)}")

        # Check for similar citations and titles
        for source in existing_citations:
            citation_similarity = 0
            title_similarity = 0

            # Compare citations if generated_citation is available
            if generated_citation:
                normalized_citation = normalize_string(generated_citation)
                existing_citation = normalize_string(existing_citations[source])
                if normalized_citation == existing_citation:
                    citation_similarity = 100
                else:
                    citation_similarity = fuzz.ratio(normalized_citation, existing_citation)
                if citation_similarity >= similarity_threshold and source not in similar_files:
                    similar_citations.append((source, existing_citations[source], citation_similarity))
                    if source not in similar_files:
                        similar_files.append(source)

            # Compare titles if title is provided
            if title and source in existing_titles:
                normalized_title = normalize_string(title)
                existing_title = existing_titles[source]
                if normalized_title == existing_title:
                    title_similarity = 100
                else:
                    title_similarity = fuzz.partial_ratio(normalized_title, existing_title)
                if title_similarity >= similarity_threshold and (source, existing_citations[source], title_similarity) not in similar_citations:
                    similar_citations.append((source, existing_citations[source], title_similarity))
                    if source not in similar_files:
                        similar_files.append(source)

        # Adjust title_exists based on high-similarity matches
        if title and any(similarity >= 95 for _, _, similarity in similar_citations):
            title_exists = True

        # Remove duplicates and sort similar citations by similarity score (descending)
        similar_citations = list(dict.fromkeys(similar_citations))  # Remove duplicates
        similar_citations.sort(key=lambda x: x[2], reverse=True)
        similar_files = list(dict.fromkeys(similar_files))  # Remove duplicate filenames

        # Construct result message
        message_parts = []
        if filename:
            if file_exists:
                message_parts.append(f"File '{filename}' exists in the vector store.")
            else:
                message_parts.append(f"File '{filename}' does not exist in the vector store.")
        if title:
            if title_exists:
                message_parts.append(f"Title '{title}' exists in the vector store.")
            else:
                message_parts.append(f"Title '{title}' does not exist in the vector store.")
        if similar_citations:
            message_parts.append(f"Found {len(similar_citations)} similar citations/titles (threshold: {similarity_threshold}%):")
            for source, citation, similarity in similar_citations:
                message_parts.append(f" - {source}: {citation} (Similarity: {similarity}%)")
        else:
            message_parts.append("No similar citations or titles found.")
            if title and not title_exists:
                message_parts.append("Try using a more specific title or lowering the similarity threshold.")

        message = "\n".join(message_parts)

        print(message)
        return {
            "exists": file_exists or title_exists,
            "similar_files": similar_files,
            "similar_citations": similar_citations,
            "message": message
        }
        
    def _generate_bibtex_entry(self, metadata, existing_keys):
        """
        Generate a BibTeX entry and unique key from metadata, handling collisions, missing values,
        and mapping non-standard dc.type to standard BibTeX entry types.

        Args:
            metadata (dict): Document metadata.
            existing_keys (set): Set of existing BibTeX keys to check for collisions.

        Returns:
            tuple: (BibTeX entry string, unique BibTeX key)
        """
        def sanitize(s):
            """Sanitize string to ensure valid UTF-8 and BibTeX-compliant characters."""
            if not isinstance(s, str):
                s = str(s)
            latex_map = {
                'é': r"{\'e}", 'è': r"{\`e}", 'ê': r"{\^e}", 'ë': r"{\:e}",
                'á': r"{\'a}", 'à': r"{\`a}", 'â': r"{\^a}", 'ä': r"{\:a}",
                'í': r"{\'i}", 'ì': r"{\`i}", 'î': r"{\^i}", 'ï': r"{\:i}",
                'ó': r"{\'o}", 'ò': r"{\`o}", 'ô': r"{\^o}", 'ö': r"{\:o}",
                'ú': r"{\'u}", 'ù': r"{\`u}", 'û': r"{\^u}", 'ü': r"{\:u}",
                'ñ': r"{\~n}", 'ç': r"{\c c}", 'å': r"{\aa}", 'ø': r"{\o}",
                'ā': r"{\=a}", 'ē': r"{\=e}", 'ī': r"{\=i}", 'ō': r"{\=o}", 'ū': r"{\=u}",
                'ǂ': r"{\textbardbl}",
            }
            # Escape BibTeX special characters, but preserve spaces
            s = s.replace('{', r'\{').replace('}', r'\}').replace('#', r'\#').replace('%', r'\%').replace('&', r'\&')
            result = ''
            for c in s:
                if ord(c) < 128 and (c.isprintable() or c.isspace()) and c not in '#%&{}':
                    result += c
                elif c in latex_map:
                    result += latex_map[c]
            return result.strip() or 'Unknown'

        # Handle author list specifically
        raw_author = metadata.get('dc.creator', 'Unknown Author')
        if self.verbose:
            print(f"Raw author for {metadata.get('source', 'Unknown File')}: {raw_author}")
        if raw_author == 'Unknown Author' or not raw_author.strip():
            author = 'Unknown Author'
        else:
            # Clean raw author: replace & with comma, normalize spaces
            raw_author = re.sub(r'\s*&\s*', ',', raw_author)  # Replace & with comma
            raw_author = re.sub(r'\s+', ' ', raw_author.strip())  # Normalize spaces
            raw_author = re.sub(r',\s*', ',', raw_author)  # Ensure single comma
            if self.verbose:
                print(f"Cleaned raw author: {raw_author}")
            # Split on commas, preserving Last, First format
            authors = []
            parts = [p.strip() for p in raw_author.split(',') if p.strip()]
            i = 0
            while i < len(parts):
                # Check if current and next part form "Last, First"
                if i + 1 < len(parts):
                    # Combine as "Last, First" (e.g., "Abdullah, M.S.")
                    author_name = f"{sanitize(parts[i])}, {sanitize(parts[i + 1])}"
                    authors.append(author_name)
                    i += 2
                else:
                    # Single part, sanitize as is
                    author_name = sanitize(parts[i])
                    authors.append(author_name)
                    i += 1
            author = ' and '.join(authors) if authors else 'Unknown Author'
            if self.verbose:
                print(f"Processed author: {author}")

        year = sanitize(metadata.get('dc.date', 'n.d.'))
        title = sanitize(metadata.get('dc.title', 'Untitled Document'))
        filename = sanitize(metadata.get('source', 'Unknown File'))
        dc_type = sanitize(metadata.get('dc.type', 'misc')).lower()

        # Safeguard for empty or invalid fields
        if not author.strip():
            author = 'Unknown Author'
        if not title.strip():
            title = 'Untitled Document'
        if not year.strip() or not re.match(r'^(19|20)\d{2}$|^n\.d\.$', year):
            year = 'n.d.'

        bibtex_type_map = {
            'document': 'misc',
            'report': 'techreport',
            'guidelines': 'misc',
            'review': 'article',
            'resolution': 'misc',
            'dissertation': 'phdthesis',
            'article': 'article',
            'book': 'book',
            'conference': 'inproceedings'
        }
        doc_type = bibtex_type_map.get(dc_type, 'misc')

        # Generate unique key
        if author == 'Unknown Author':
            author_last = 'unknown'
        else:
            author_last = author.split(' and ')[0].strip().lower()
            author_last = author_last.split(',')[0].strip() if ',' in author_last else author_last
        base_key = re.sub(r'[\W_]+', '', author_last) + year.replace('n.d.', 'nodate')
        key = base_key
        suffix = 'a'
        while key in existing_keys:
            key = base_key + suffix
            suffix = chr(ord(suffix) + 1)

        # Truncate long fields
        author = author[:200]
        title = title[:200]
        filename = filename[:100]

        entry = f"""@{doc_type}{{{key},
          author = {{{author}}},
          title = {{{title}}},
          year = {{{year}}},
          note = {{Filename: {filename}}}
        }}"""
        return entry, key

  
    def list_references(self, save_to_file=None):
        """
        List all unique references from BibTeX file or fallback to vector store.

        Args:
            save_to_file (str, optional): File path to save references (e.g., 'references.csv').

        Returns:
            list: Sorted list of dictionaries with keys: author, date, title, document_type, filename.
        """
        unique_references = []
        if os.path.exists(self.bibtex_file):
            try:
                with open(self.bibtex_file, encoding="utf-8") as bibtex_file:
                    parser = bibtexparser.bparser.BibTexParser(
                        common_strings=True,
                        ignore_nonstandard_types=False,
                        homogenize_fields=False,  # Avoid field normalization issues
                        interpolate_strings=False
                    )
                    bib_database = bibtexparser.load(bibtex_file, parser=parser)
                expected_count = sum(1 for line in open(self.bibtex_file, encoding="utf-8") if line.strip().startswith('@'))
                parsed_count = len(bib_database.entries)
                if parsed_count < expected_count:
                    skipped = expected_count - parsed_count
                    self.metadata_fallback_report.append(
                        f"Warning: Parsed {parsed_count} BibTeX entries, expected {expected_count}. {skipped} entries may be malformed or skipped."
                    )
                    if self.verbose:
                        print(f"Warning: {skipped} BibTeX entries failed to parse. Inspect references.bib for issues like unbalanced braces, invalid years, or missing fields.")
                for entry in bib_database.entries:
                    try:
                        # Relaxed validation to include entries with partial fields
                        bibtex_to_display_type = {
                            'misc': 'Document',
                            'techreport': 'Report',
                            'article': 'Article',
                            'book': 'Book',
                            'inproceedings': 'Conference',
                            'phdthesis': 'Thesis'
                        }
                        display_type = bibtex_to_display_type.get(entry.get('ENTRYTYPE', 'misc').lower(), 'Document')
                        unique_references.append({
                            "author": entry.get('author', 'Unknown Author'),
                            "date": entry.get('year', 'n.d.'),
                            "title": entry.get('title', 'Untitled Document'),
                            "document_type": display_type,
                            "filename": entry.get('note', '').replace('Filename: ', '') if 'note' in entry else 'Unknown File'
                        })
                        if self.verbose:
                            print(f"Processed entry {entry.get('ID', 'Unknown')}: {entry.get('title', 'Untitled')}")
                    except Exception as e:
                        skipped_id = entry.get('ID', 'Unknown')
                        self.metadata_fallback_report.append(
                            f"Failed to parse BibTeX entry {skipped_id}: {str(e)}"
                        )
                        if self.verbose:
                            print(f"Warning: Skipped BibTeX entry {skipped_id}: {str(e)}")
            except UnicodeDecodeError as e:
                self.metadata_fallback_report.append(f"Error reading BibTeX file: {str(e)}")
                if self.verbose:
                    print(f"Error reading BibTeX file: {str(e)}. Falling back to vector store.")
                # Fallback to vector store
                unique_references = self._list_references_from_vector_store()
        else:
            # Fallback to vector store
            unique_references = self._list_references_from_vector_store()

        unique_references = sorted(unique_references, key=lambda x: x["author"])
        if self.verbose:
            print(f"Returning {len(unique_references)} unique references")

        # Generate CSV output
        csv_output = "Author|Date|Title|Document Type|Filename\n"
        for ref in unique_references:
            escaped_fields = [str(field).replace("|", "\\|") for field in [
                ref["author"], ref["date"], ref["title"], ref["document_type"], ref["filename"]
            ]]
            csv_output += "|".join(escaped_fields) + "\n"

        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(csv_output)
            if self.verbose:
                print(f"Wrote {len(unique_references)} references to {save_to_file}")
                print("\nFirst 20 References from local database (CSV format):")
                print(csv_output.split("\n")[0])
                for line in csv_output.split("\n")[1:21]:
                    if line.strip():
                        print(line)
        else:
            if self.verbose:
                print("\nAll References from local database (CSV format):")
                print(csv_output)

        return unique_references


    def get_document_metadata(self, filename):
        """
        Retrieve the BibTeX metadata for a specific document as a dictionary.

        Args:
            filename (str): The filename of the document to retrieve metadata for (case-insensitive).

        Returns:
            dict: A dictionary containing the BibTeX metadata for the specified file, with keys such as
                  'dc.title', 'dc.creator', 'dc.date', 'dc.type', 'source', etc. Returns an empty
                  dictionary if the file is not found in the BibTeX dictionary.
        """
        bib_dict = self.bib_dict or {}
        for bib_filename, metadata in bib_dict.items():
            if bib_filename.lower() == filename.lower():
                return metadata
        return {}

    def _normalize_filename(self, filename):
        """
        Normalize a filename by removing LaTeX escape sequences and converting to a consistent format.

        Args:
            filename (str): The filename to normalize.

        Returns:
            str: Normalized filename.
        """
        # Replace common LaTeX escape sequences (e.g., {\'e} to é)
        latex_replacements = {
            r"{\'e}": "é",
            r"{\`e}": "è",
            r"{\^e}": "ê",
            r"{\"e}": "ë",
            r"{\'a}": "á",
            r"{\`a}": "à",
            r"{\^a}": "â",
            r"{\'i}": "í",
            r"{\`i}": "ì",
            r"{\^i}": "î",
            r"{\'o}": "ó",
            r"{\`o}": "ò",
            r"{\^o}": "ô",
            r"{\'u}": "ú",
            r"{\`u}": "ù",
            r"{\^u}": "û",
            r"{\'c}": "ç",
            r"{\'n}": "ñ",
        }
        normalized = filename
        for latex_char, unicode_char in latex_replacements.items():
            normalized = normalized.replace(latex_char, unicode_char)
        # Remove any remaining LaTeX braces and normalize unicode
        normalized = re.sub(r'[{}]', '', normalized)
        normalized = unicodedata.normalize('NFC', normalized).lower().strip()
        return normalized


    def _add_to_bibtex(self, metadata):
        """
        Add or update a BibTeX entry for the given metadata, ensuring no duplicates.

        Args:
            metadata (dict): Dictionary containing 'source', 'dc.title', 'dc.creator', 'dc.date', 'dc.type'.
        """
        if not metadata.get('source'):
            error_message = "Error: Cannot add to BibTeX, no 'source' provided."
            self.metadata_fallback_report.append(error_message)
            if self.verbose:
                print(error_message)
            return

        try:
            # Load existing BibTeX
            bib_database = bibtexparser.bparser.BibDatabase()
            existing_keys = set()
            if os.path.exists(self.bibtex_file):
                with open(self.bibtex_file, encoding="utf-8") as bibtex_file:
                    parser = BibTexParser(
                        common_strings=True,
                        ignore_nonstandard_types=False,
                        homogenize_fields=False,
                        interpolate_strings=False
                    )
                    bib_database = bibtexparser.load(bibtex_file, parser=parser)
                    existing_keys = {entry['ID'] for entry in bib_database.entries}

            # Remove existing entries for this filename
            filename = metadata['source']
            normalized_filename = self._normalize_filename(filename)
            original_entry_count = len(bib_database.entries)
            removed_entries = []
            kept_entries = []

            for entry in bib_database.entries:
                note = entry.get('note', '')
                normalized_note = self._normalize_filename(note.replace('Filename: ', '').strip())
                if normalized_note == normalized_filename:
                    removed_entries.append(entry)
                else:
                    kept_entries.append(entry)

            bib_database.entries = kept_entries
            removed_count = original_entry_count - len(bib_database.entries)
            if self.verbose:
                print(f"Checking for existing BibTeX entries for filename '{filename}' (normalized: '{normalized_filename}')")
                if removed_entries:
                    print(f"Removed {removed_count} existing BibTeX entries:")
                    for entry in removed_entries:
                        print(f"  ID: {entry['ID']}, Note: {entry.get('note', 'None')}")
                else:
                    print(f"No existing BibTeX entries found for filename '{filename}'")

            # Generate unique BibTeX key
            base_key = f"{metadata.get('dc.creator', 'unknown').split(',')[0].lower().replace(' ', '')}{self._normalize_filename(metadata.get('dc.date', 'nodate'))}"
            key = base_key
            suffix = 0
            while key in existing_keys:
                suffix += 1
                key = f"{base_key}{chr(97 + suffix)}"  # Append 'a', 'b', etc.
            existing_keys.add(key)

            # Create new BibTeX entry
            entry = {
                'ENTRYTYPE': 'article',
                'ID': key,
                'author': metadata.get('dc.creator', 'Unknown Author'),
                'title': metadata.get('dc.title', 'Untitled Document'),
                'year': metadata.get('dc.date', 'n.d.'),
                'note': f"Filename: {filename}",
                'type': metadata.get('dc.type', 'Document')
            }
            bib_database.entries.append(entry)

            if self.verbose:
                print(f"Adding new BibTeX entry with key '{key}':")
                print(f"  {entry}")

            # Write updated BibTeX file
            writer = BibTexWriter()
            with open(self.bibtex_file, 'w', encoding="utf-8") as bibtex_file:
                bibtex_file.write(writer.write(bib_database))
            
            self.metadata_fallback_report.append(f"Added/Updated BibTeX entry for {filename} with key {key}")
            if self.verbose:
                print(f"Successfully wrote BibTeX entry to '{self.bibtex_file}'")
        except Exception as e:
            error_message = f"Error adding/updating BibTeX for {metadata.get('source', 'Unknown')}: {str(e)}"
            self.metadata_fallback_report.append(error_message)
            if self.verbose:
                print(error_message)


    def query(self, query, top_k=100, similarity_threshold=70, specific_title=None, specific_author=None, specific_file=None, verbose=False, includereferences=True):
        """
        Query the RAG system with a given query, optionally filtering by specific title, author, or file.

        Args:
            query (str): The query string.
            top_k (int): Number of chunks to retrieve (default: 20).
            similarity_threshold (int): Similarity threshold for title/author filtering (default: 80).
            specific_title (str, optional): Specific title to filter for.
            specific_author (str, optional): Specific author to filter for.
            specific_file (str, optional): Specific filename to restrict chunks to.
            verbose (bool): If True, print only the chunks used in the final response and diagnostic metadata for citations.

        Returns:
            tuple: (answer, references) where answer is the generated response, and references is a list of APA-formatted citations.
        """
        # Sync vector store metadata with BibTeX to ensure consistency

        base_context = self.context or "Focus on Indigenous Traditional Ecological Knowledge"
        # prompt_template = PromptTemplate.from_template(
            # """Context: {context}

    # Documents:
    # {documents}

    # Query: {query}

    # Answer the query based on the provided documents and context. Always support your answer with inline citations to the relevant references using their numbers, e.g., [1] for the first reference, [2] for the second, and so on. Use multiple citations where appropriate to substantiate claims."""
        # )




        prompt_template = PromptTemplate.from_template(
            """Context: {context}

    Documents:
    {documents}

    Query: {query}

    Answer the query based on the provided documents and context. """
        )
        if includereferences:
            prompt_template += """Always support your answer with inline citations to the relevant references using their numbers, e.g., [1] for the first reference, [2] for the second, and so on. Use multiple citations where appropriate to substantiate claims."""


        constraints = []
        if specific_title:
            constraints.append(f"papers with titles similar to '{specific_title}'")
        if specific_author:
            constraints.append(f"papers by '{specific_author}'")
        if specific_file:
            constraints.append(f"chunks from the file '{specific_file}'")
        constraint_text = " and ".join(constraints) if constraints else "relevant sources"
        
        raw_scores = []
        title_pass_count = 0
        author_pass_count = 0
        file_pass_count = 0
        
        filtered_chunks = []
        
        if specific_file:
            # Explicitly retrieve all chunks from the specified file using .get() for reliability
            retrieval_results = self.vector_store.get()
            documents = retrieval_results.get('documents', [])
            metadatas = retrieval_results.get('metadatas', [])
            
            for doc_content, meta in zip(documents, metadatas):
                bib_meta = self.bib_dict.get(meta.get('source', ''), {})
                updated_meta = {**meta, **bib_meta}
                
                title_match = not specific_title
                author_match = not specific_author
                file_match = updated_meta.get("source", "").lower() == specific_file.lower()
                
                if specific_title:
                    title_similarity = fuzz.token_sort_ratio(specific_title.lower(), updated_meta.get("dc.title", "").lower())
                    title_match = title_similarity >= similarity_threshold
                    if title_match:
                        title_pass_count += 1
                
                if specific_author:
                    author_similarity = fuzz.token_sort_ratio(specific_author.lower(), updated_meta.get("dc.creator", "").lower())
                    author_match = author_similarity >= similarity_threshold
                    if author_match:
                        author_pass_count += 1
                
                if file_match:
                    file_pass_count += 1
                
                if title_match and author_match and file_match:
                    chunk = Document(page_content=doc_content, metadata=updated_meta)
                    filtered_chunks.append((chunk, 0))  # No similarity score, use 0
            raw_scores = [0] * len(filtered_chunks)
            
            # Sort chunks by page number for sequential order, useful for summaries
            filtered_chunks.sort(key=lambda x: int(x[0].metadata.get('page', 0)) if 'page' in x[0].metadata else 0)
        
        else:
            effective_top_k = top_k
            if specific_title or specific_author:
                effective_top_k = 200  # Increase retrieval count to improve fuzz matching chances
            
            retrieved_chunks_with_scores = self.vector_store.similarity_search_with_score(query, k=effective_top_k)
            
            raw_scores = []
            for chunk, score in retrieved_chunks_with_scores:
                meta = chunk.metadata
                bib_meta = self.bib_dict.get(meta['source'], {})
                meta = {**meta, **bib_meta}
                chunk.metadata = meta
                
                title_match = not specific_title
                author_match = not specific_author
                
                if specific_title:
                    title_similarity = fuzz.token_sort_ratio(specific_title.lower(), meta.get("dc.title", "").lower())
                    title_match = title_similarity >= similarity_threshold
                    if title_match:
                        title_pass_count += 1
                
                if specific_author:
                    author_similarity = fuzz.token_sort_ratio(specific_author.lower(), meta.get("dc.creator", "").lower())
                    author_match = author_similarity >= similarity_threshold
                    if author_match:
                        author_pass_count += 1
                
                raw_scores.append(score)
                if score < 0:
                    score = 0
                
                if title_match and author_match:
                    filtered_chunks.append((chunk, score))
        
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            max_score = 1.0
        similarity_scores = [1 - (score / max_score) for score in raw_scores]
        
        processed_chunks = [chunk for chunk, _ in filtered_chunks]
        processed_chunks = processed_chunks[:10]
        
        if not processed_chunks:
            answer = f"No relevant chunks found for this query{' from the specified file' if specific_file else ''}."
            references = []
            return answer, references
        
        reference_to_id = {}
        current_id = 1
        documents_text = ""
        valid_identifiers = set()
        for chunk in processed_chunks:
            meta = chunk.metadata
            bib_meta = self.bib_dict.get(meta['source'], {})
            meta = {**meta, **bib_meta}  # Ensure BibTeX metadata overrides
            if verbose:
                print(f"\n[CITATION DIAGNOSTIC] Metadata for source '{meta['source']}':")
                print(f"  Title: {meta.get('dc.title', 'N/A')}")
                print(f"  Creator: {meta.get('dc.creator', 'N/A')}")
                print(f"  Date: {meta.get('dc.date', 'N/A')}")
                print(f"  Type: {meta.get('dc.type', 'N/A')}")
            apa_reference = self._format_apa_citation(meta)
            if apa_reference not in reference_to_id:
                reference_to_id[apa_reference] = current_id
                valid_identifiers.add(current_id)
                documents_text += (
                    f"Reference: [{current_id}] {apa_reference}\n"
                    f"Content:\n{chunk.page_content}\n\n"
                )
                current_id += 1
        
        full_prompt = prompt_template.format(
            context=f"{base_context}\n\nUse only {constraint_text} that are highly relevant to ITEK.",
            documents=documents_text,
            query=query
        )
        answer = self.llm.invoke(full_prompt)
        
        cited_identifiers = set()
        identifier_matches = re.findall(r'\[(\d+)\]', answer)
        for identifier in identifier_matches:
            try:
                id_num = int(identifier)
                if id_num in valid_identifiers:
                    cited_identifiers.add(id_num)
            except ValueError:
                continue
        
        new_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(cited_identifiers), 1)}
        references = []
        for ref, ref_id in reference_to_id.items():
            if ref_id in cited_identifiers:
                new_id = new_id_mapping[ref_id]
                references.append(f"[{new_id}] {ref}")
        references.sort(key=lambda x: int(x.split("]")[0][1:]))
        references = references[:5]
        
        for old_id, new_id in new_id_mapping.items():
            answer = re.sub(rf'\[{old_id}\]', f'[{new_id}]', answer)
        
        if verbose:
            print(f"\n[CITED CHUNKS] Chunks used in the final statement (from {len(cited_identifiers)} cited identifiers):")
            cited_sources = set()
            for ref, ref_id in reference_to_id.items():
                if ref_id in cited_identifiers:
                    source_match = re.search(r'\[Filename:\s*(.+?)\]$', ref)
                    if source_match:
                        cited_sources.add(source_match.group(1))
            for source in cited_sources:
                matching_chunks = [chunk for chunk in processed_chunks if chunk.metadata.get('source') == source]
                for j, chunk in enumerate(matching_chunks, 1):
                    print(f"[CITED CHUNKS] Chunk {j} from source '{source}':")
                    print(f"  Title: {chunk.metadata.get('dc.title', 'N/A')}")
                    print(f"  Creator: {chunk.metadata.get('dc.creator', 'N/A')}")
                    print(f"  Full content:\n{chunk.page_content}\n")
        
        return answer, references



    def _format_apa_citation(self, metadata, verbose=False):
        """
        Format metadata into an APA-style citation.
        
        Args:
            metadata (dict): Metadata containing dc.title, dc.creator, dc.date, etc.
            verbose (bool): If True, print diagnostic information about the citation process.
        
        Returns:
            str: APA-formatted citation string.
        """
        if verbose:
            print(f"[CITATION DIAGNOSTIC] Processing citation for source '{metadata.get('source', 'N/A')}':")
            print(f"  dc.creator: {metadata.get('dc.creator', 'N/A')}")
            print(f"  dc.title: {metadata.get('dc.title', 'N/A')}")
            print(f"  dc.date: {metadata.get('dc.date', 'N/A')}")
            print(f"  dc.type: {metadata.get('dc.type', 'N/A')}")
            print(f"  journal: {metadata.get('journal', 'N/A')}")
            print(f"  doi: {metadata.get('doi', 'N/A')}")
            print(f"  publisher: {metadata.get('publisher', 'N/A')}")

        # Get creator (author) and handle multi-author formatting
        creator = metadata.get('dc.creator', 'Unknown Author')
        if creator and creator != 'Unknown Author':
            authors = [author.strip() for author in creator.split(' and ')]
            if len(authors) > 20:
                authors = authors[:19] + ['et al.']
            elif len(authors) > 2:
                authors[-1] = f"& {authors[-1]}"
            formatted_creator = ", ".join(authors)
        else:
            formatted_creator = 'Unknown Author'

        # Get other fields
        title = metadata.get('dc.title', 'Untitled')
        year = metadata.get('dc.date', 'n.d.')
        doc_type = metadata.get('dc.type', 'Document')
        source = metadata.get('source', 'Unknown Source')
        journal = metadata.get('journal', '')
        doi = metadata.get('doi', '')
        publisher = metadata.get('publisher', '')

        # Construct APA citation
        citation = f"{formatted_creator}. ({year}). {title}"
        if journal:
            citation += f". {journal}"
        if doi:
            citation += f". https://doi.org/{doi}"
        elif publisher:
            citation += f". {publisher}"
        citation += f" ({doc_type}). [Filename: {source}]"

        if verbose:
            print(f"  Formatted APA citation: {citation}")

        return citation



    def generate_annotated_summary(self, filename, top_k=1000, include_references=False):
            """
            Generates an annotated summary for a specific document in the vector store.
            
            Args:
                filename (str): The name of the PDF file to summarize.
                top_k (int): Number of top chunks to retrieve from the vector store (default: 1000).
                include_references (bool): Whether to include references in the query response (default: False).
            
            Returns:
                tuple: (success: bool, summary: str, metadata: dict)
                       - success: True if summary generated, False if file or metadata not found.
                       - summary: Formatted summary with metadata and generated text.
                       - metadata: The document's metadata.
            """
            filename = self._normalize_filename(filename)
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Get metadata
            metadata = self.get_document_metadata(filename)
            if not metadata:
                if self.verbose:
                    print(f"[SUMMARY] No metadata found for {filename}")
                return False, f"No metadata found for {filename}", {}
            
            # Query for summary
            query_text = (
                "Provide an annotated summary of the entire document. Be as descriptive as possible, provide no insight, just summarize the documents text. "
                "Don't mention the title of the paper in the summary, or provide comments, don't even describe it as an annotated summary. "
                "Only provide relevant text, such that a reader would understand the documents meaning and purpose."
            )
            
            try:
                answer, references = self.query(query=query_text, specific_file=filename, top_k=top_k, includereferences=include_references)
                if self.verbose:
                    print(f"[SUMMARY] Generated summary for {filename}: {answer[:100]}...")
            except Exception as e:
                if self.verbose:
                    print(f"[SUMMARY] Failed to generate summary for {filename}: {str(e)}")
                self.metadata_fallback_report.append(f"Summary generation failed for {filename}: {str(e)}")
                return False, f"Failed to generate summary for {filename}: {str(e)}", metadata
            
            # Format summary
            summary = f"""
    Title: {metadata.get('dc.title', 'N/A')}
    Author(s): {metadata.get('dc.creator', 'N/A')}
    Year: {metadata.get('dc.date', 'N/A')}
    Summary:
    {answer}
    """
            
            if self.verbose:
                print(f"[SUMMARY] Formatted summary for {filename}")
            self.metadata_fallback_report.append(f"Generated summary for {filename}")
            
            return True, summary, metadata

    def generate_all_annotated_summaries(self, sort_by='author', output_file='annotated_summaries.md'):
        """
        Generates annotated summaries for all documents in the vector store, sorts them, and saves to a file.
        Displays progress (processed/total documents) during execution.
        
        Args:
            sort_by (str): Sort by 'author', 'title', or 'filename' (default: 'author').
            output_file (str): Path to save the summaries (default: 'annotated_summaries.md').
        
        Returns:
            tuple: (success: bool, message: str, num_summaries: int)
        """
        if not self.vector_store:
            return False, "Vector store not initialized.", 0
        
        # Retrieve unique filenames from vector store
        metadatas = self.vector_store._collection.get(include=["metadatas"])["metadatas"]
        unique_files = list(set(meta.get('source', '') for meta in metadatas if meta.get('source', '').endswith('.pdf')))
        
        if not unique_files:
            return False, "No documents found in vector store.", 0
        
        total_files = len(unique_files)
        summaries = []
        processed_files = 0
        
        for filename in unique_files:
            processed_files += 1
            success, summary, metadata = self.generate_annotated_summary(filename)
            if success:
                summaries.append({
                    'filename': filename,
                    'metadata': metadata,
                    'summary': summary
                })
                if self.verbose:
                    print(f"[SUMMARY] Generated summary for {filename} (Processed {processed_files}/{total_files} documents)")
            else:
                self.metadata_fallback_report.append(f"Failed to generate summary for {filename}: {summary}")
                if self.verbose:
                    print(f"[SUMMARY] Failed for {filename}: {summary} (Processed {processed_files}/{total_files} documents)")
                else:
                    print(f"Processed {processed_files}/{total_files} documents")
            
            # Print standalone progress if not verbose
            if not self.verbose:
                print(f"Processed {processed_files}/{total_files} documents")
        
        if not summaries:
            return False, "No summaries generated.", 0
        
        # Sort summaries
        if sort_by == 'author':
            summaries.sort(key=lambda x: x['metadata'].get('dc.creator', '').lower())
        elif sort_by == 'title':
            summaries.sort(key=lambda x: x['metadata'].get('dc.title', '').lower())
        elif sort_by == 'filename':
            summaries.sort(key=lambda x: x['filename'].lower())
        else:
            self.metadata_fallback_report.append(f"Invalid sort_by '{sort_by}', defaulting to author.")
            summaries.sort(key=lambda x: x['metadata'].get('dc.creator', '').lower())
        
        # Write to file (Markdown format)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Annotated Summaries of ITEK Documents\n\n")
                for i, entry in enumerate(summaries):
                    f.write(entry['summary'])
                    if i < len(summaries) - 1:  # Add separator except for the last entry
                        f.write("\n---\n\n")
            
            self.metadata_fallback_report.append(f"Saved {len(summaries)} summaries to {output_file}")
            if self.verbose:
                print(f"[SUMMARY] Saved {len(summaries)} summaries to {output_file}")
            
            return True, f"Generated and saved {len(summaries)} summaries to {output_file}", len(summaries)
        except Exception as e:
            self.metadata_fallback_report.append(f"Failed to save summaries to {output_file}: {str(e)}")
            if self.verbose:
                print(f"[SUMMARY] Failed to save summaries: {str(e)}")
            return False, f"Failed to save summaries to {output_file}: {str(e)}", len(summaries)

    def update_document_metadata(self, filename, use_scholarly=False, **metadata):
        """
        Updates metadata for a document in the vector store and BibTeX.
        If use_scholarly is True, attempts to infer metadata using _infer_metadata_with_llm if no metadata provided.
        
        Args:
            filename (str): The name of the PDF file to update.
            use_scholarly (bool): If True, try LLM inference if no metadata provided.
            **metadata: Key-value pairs of metadata fields to update (e.g., dc_title, dc_creator).
        
        Returns:
            tuple: (success: bool, message: str, updated_metadata: dict)
        """
        filename = self._normalize_filename(filename)
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        current_metadata = self.get_document_metadata(filename)
        if not current_metadata:
            return False, f"No document found with filename {filename}", {}
        
        updated_metadata = current_metadata.copy()
        updated_fields = []
        
        # Map allowed fields to BibTeX-compatible keys
        allowed_fields = {
            'dc_title': 'dc.title',
            'dc_creator': 'dc.creator',
            'dc_date': 'dc.date',
            'dc_type': 'dc.type',
            'journal': 'journal',
            'doi': 'doi',
            'publisher': 'publisher'
        }
        
        # Apply user-provided metadata
        for key, value in metadata.items():
            if key in allowed_fields:
                bib_key = allowed_fields[key]
                if value and value != updated_metadata.get(bib_key):
                    updated_metadata[bib_key] = value
                    updated_fields.append(bib_key)
        
        # If no metadata provided and use_scholarly=True, try LLM inference
        if not updated_fields and use_scholarly:
            try:
                inferred_metadata = self._infer_metadata_with_llm(filename)
                if self.verbose:
                    print(f"[LLM INFERENCE] Inferred metadata: {inferred_metadata}")
                for key, value in inferred_metadata.items():
                    bib_key = allowed_fields.get(key, key)
                    if bib_key in allowed_fields.values() and value and bib_key not in updated_metadata:
                        updated_metadata[bib_key] = value
                        updated_fields.append(bib_key)
                if updated_fields:
                    self.metadata_fallback_report.append(f"LLM inference updated {filename}: {', '.join(updated_fields)}")
            except Exception as e:
                self.metadata_fallback_report.append(f"LLM inference failed for {filename}: {str(e)}")
                if self.verbose:
                    print(f"[LLM INFERENCE] Error: {str(e)}. Using existing metadata.")
        
        # If no updates, return current metadata
        if not updated_fields:
            return True, f"No updates provided for {filename}. Current metadata returned.", updated_metadata
        
        # Update BibTeX
        self._add_to_bibtex({'source': filename, **updated_metadata})
        
        # Update vector store
        results = self.vector_store._collection.get(where={'source': filename}, include=['metadatas'])
        ids = results['ids']
        if ids:
            updated_metadatas = []
            for meta in results['metadatas']:
                updated_meta = meta.copy()
                dc_mapping = {'title': 'dc.title', 'creator': 'dc.creator', 'date': 'dc.date', 'type': 'dc.type'}
                for k, v in updated_metadata.items():
                    mapped_k = dc_mapping.get(k, k)
                    if mapped_k.startswith('dc.'):
                        updated_meta[mapped_k] = v
                    else:
                        updated_meta[k] = v
                updated_metadatas.append(updated_meta)
            self.vector_store._collection.update(ids=ids, metadatas=updated_metadatas)
            self.metadata_fallback_report.append(f"Updated {len(ids)} chunks for {filename}")
        
        message = f"Updated {filename}: {', '.join(updated_fields) if updated_fields else 'no fields changed'}"
        if use_scholarly:
            message += " (LLM inference attempted if no metadata provided)"
        
        return True, message, updated_metadata

    
    def _list_references_from_vector_store(self):
        """Helper method to extract references from vector store metadata."""
        unique_references = []
        if self.vector_store is not None:
            metadatas = self.vector_store._collection.get(include=["metadatas"])["metadatas"]
            source_to_metadata = {}
            for meta in metadatas:
                source = meta.get("source", "")
                if source and source.endswith(".pdf") and source not in source_to_metadata:
                    source_to_metadata[source] = meta
            for source, meta in source_to_metadata.items():
                try:
                    citation = self._format_apa_citation(meta)
                    match = re.match(r'^(.*?)\.\s*\((.*?)\)\.\s*(.*?)\s*(?:\((.*?)\))?\s*\.\s*\[Filename:\s*(.*?)\]$', citation)
                    if match:
                        author, date, title, doc_type, filename = match.groups()
                        doc_type = doc_type or "Document"
                        unique_references.append({
                            "author": author.strip() if author else "Unknown Author",
                            "date": date.strip() if date else "n.d.",
                            "title": title.strip() if title else "Untitled Document",
                            "document_type": doc_type.strip(),
                            "filename": filename.strip()
                        })
                    else:
                        self.metadata_fallback_report.append(f"Failed to parse citation for {source}: {citation[:100]}...")
                        if self.verbose:
                            print(f"Warning: Failed to parse citation for {source}: {citation[:100]}...")
                except Exception as e:
                    self.metadata_fallback_report.append(f"Error processing metadata for {source}: {str(e)}")
                    if self.verbose:
                        print(f"Warning: Skipped metadata for {source}: {str(e)}")
        else:
            self.metadata_fallback_report.append("No documents or vector store available.")
            if self.verbose:
                print("No documents or vector store available.")
        return unique_references
        """
        List all unique references from BibTeX file or fallback to vector store.

        Args:
            save_to_file (str, optional): File path to save references (e.g., 'references.csv').

        Returns:
            list: Sorted list of dictionaries with keys: author, date, title, document_type, filename.
        """
        unique_references = []
        if os.path.exists(self.bibtex_file):
            try:
                with open(self.bibtex_file, encoding="utf-8") as bibtex_file:
                    parser = bibtexparser.bparser.BibTexParser(
                        common_strings=True,
                        ignore_nonstandard_types=False,
                        homogenize_fields=True,
                        interpolate_strings=False
                    )
                    bib_database = bibtexparser.load(bibtex_file, parser=parser)
                expected_count = sum(1 for line in open(self.bibtex_file, encoding="utf-8") if line.strip().startswith('@'))
                parsed_count = len(bib_database.entries)
                if parsed_count < expected_count:
                    skipped = expected_count - parsed_count
                    self.metadata_fallback_report.append(
                        f"Warning: Parsed {parsed_count} BibTeX entries, expected ~{expected_count}. {skipped} entries may be malformed or skipped."
                    )
                    if self.verbose:
                        print(f"Warning: {skipped} BibTeX entries failed to parse. Inspect references.bib for issues like unbalanced braces or invalid characters.")
                for entry in bib_database.entries:
                    try:
                        # Validate required fields
                        if not entry.get('ID') or not entry.get('ENTRYTYPE'):
                            skipped_note = entry.get('note', 'Unknown')
                            self.metadata_fallback_report.append(
                                f"Skipped entry with missing ID or ENTRYTYPE: {skipped_note}"
                            )
                            if self.verbose:
                                print(f"Skipping entry missing ID/ENTRYTYPE: {skipped_note}")
                            continue
                        # Map BibTeX entry type to display-friendly document type
                        bibtex_to_display_type = {
                            'misc': 'Document',
                            'techreport': 'Report',
                            'article': 'Article',
                            'book': 'Book',
                            'inproceedings': 'Conference',
                            'phdthesis': 'Thesis'
                        }
                        display_type = bibtex_to_display_type.get(entry['ENTRYTYPE'].lower(), 'Document')
                        unique_references.append({
                            "author": entry.get('author', 'Unknown Author'),
                            "date": entry.get('year', 'n.d.'),
                            "title": entry.get('title', 'Untitled Document'),
                            "document_type": display_type,
                            "filename": entry.get('note', '').replace('Filename: ', '') if 'note' in entry else 'Unknown File'
                        })
                    except Exception as e:
                        skipped_id = entry.get('ID', 'Unknown')
                        self.metadata_fallback_report.append(
                            f"Failed to parse BibTeX entry {skipped_id}: {str(e)}"
                        )
                        if self.verbose:
                            print(f"Warning: Skipped BibTeX entry {skipped_id}: {str(e)}")
            except UnicodeDecodeError as e:
                self.metadata_fallback_report.append(f"Error reading BibTeX file: {str(e)}")
                if self.verbose:
                    print(f"Error reading BibTeX file: {str(e)}. Falling back to vector store.")
        else:
            # Fallback to vector store metadata (existing code remains unchanged here for brevity)
            if self.vector_store is not None:
                metadatas = self.vector_store._collection.get(include=["metadatas"])["metadatas"]
                existing_sources = {meta["source"] for meta in metadatas if "source" in meta}
                # ... (rest of fallback logic as in original code)

        unique_references = sorted(unique_references, key=lambda x: x["author"])
        if self.verbose:
            print(f"Returning {len(unique_references)} unique references")

        # Generate CSV output with '|' delimiter (existing code remains unchanged)
        # ...

        return unique_references
       

    def export_vector_store(self, output_file="itek_vectorstore.zip"):
        """Export the FAISS vector store folder as a zip file for distribution."""
        if self.vector_store is None:
            raise RuntimeError("No FAISS vector store loaded.")

        import zipfile
        from pathlib import Path

        folder_path = Path(self.vector_store_dir).resolve()
        zip_path = Path(output_file).resolve()

        # Save latest state
        self.vector_store.save_local(str(folder_path))

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file in (".DS_Store", "Thumbs.db"):
                        continue
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(folder_path)
                    zipf.write(file_path, arcname)

        if self.verbose:
            print(f"Exported to: {zip_path}")
            print(f"Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
            print("Unzip and use the extracted folder as vector_store_dir")

    # ... (keep your other methods: query, update metadata, list references, etc.)