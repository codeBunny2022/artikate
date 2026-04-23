from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
import pypdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    document: str
    page: int
    chunk_index: int
    char_start: int
    char_end: int


class DocumentChunker:
    """
    splits text into overlapping chunks using paragraph/sentence boundaries.
    advances by (chunk_size - overlap) on every step, guaranteed forward progress.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150,
                 min_chunk_size: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def chunk_text(self, text: str, document: str, page: int,
                   start_index: int = 0) -> list[TextChunk]:
        segments = self._split(text)
        chunks = []
        for i, (seg, cs, ce) in enumerate(segments):
            if len(seg.strip()) < self.min_chunk_size:
                continue
            chunks.append(TextChunk(
                text=seg.strip(),
                document=document,
                page=page,
                chunk_index=start_index + i,
                char_start=cs,
                char_end=ce,
            ))
        return chunks

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        # Entire text fits in one chunk
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        results = []
        step = self.chunk_size - self.chunk_overlap  # always > 0
        pos = 0

        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            # Try to find a clean semantic boundary before 'end'
            split_at = self._find_boundary(text, pos, end)
            chunk = text[pos:split_at]
            if len(chunk.strip()) >= self.min_chunk_size:
                results.append((chunk, pos, split_at))
            if split_at >= len(text):
                break
            # Advance by fixed step — guaranteed to make progress
            pos += step

        return results

    def _find_boundary(self, text: str, start: int, end: int) -> int:
        """find the nearest clean boundary searching backwards from end"""
        for sep in self.separators:
            idx = text.rfind(sep, start, end)
            if idx != -1 and idx > start + self.min_chunk_size:
                return idx + len(sep)
        return end


class PDFIngestionPipeline:
    def __init__(self, chunker: DocumentChunker | None = None):
        self.chunker = chunker or DocumentChunker()

    def ingest(self, pdf_path: str | Path) -> list[TextChunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Ingesting: %s", pdf_path.name)
        all_chunks = []
        chunk_index = 0

        try:
            reader = pypdf.PdfReader(str(pdf_path))
        except Exception as e:
            raise ValueError(f"Cannot open PDF {pdf_path}: {e}")

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = self._clean(raw_text)
            if not cleaned.strip():
                continue
            page_chunks = self.chunker.chunk_text(
                text=cleaned,
                document=pdf_path.name,
                page=page_num,
                start_index=chunk_index,
            )
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        logger.info("Ingested %s → %d chunks", pdf_path.name, len(all_chunks))
        return all_chunks

    def ingest_directory(self, directory: str | Path) -> list[TextChunk]:
        directory = Path(directory)
        pdfs = sorted(directory.glob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs found in %s", directory)
            return []
        all_chunks = []
        for pdf in pdfs:
            try:
                all_chunks.extend(self.ingest(pdf))
            except Exception as e:
                logger.error("Failed %s: %s", pdf.name, e)
        logger.info("Total chunks: %d from %d documents",
                    len(all_chunks), len(pdfs))
        return all_chunks

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
