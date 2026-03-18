import hashlib
import logging
import os
from typing import List, Optional

import fitz
import numpy as np
import torch
from llama_index.core import Document
from paddleocr import PaddleOCR
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, pipeline
from unstructured.documents.elements import Element, Table
from unstructured.partition.auto import partition

from .config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "doc_processor.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("DocProcessor")


class FullDocProcessor:
    def __init__(self) -> None:
        self.ocr_lang = Config.OCR_LANG

        logger.info("Initializing PaddleOCR")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.ocr_lang,
            show_log=False,
            use_gpu=bool(torch.cuda.is_available()),
        )

        self.table_detector_processor: Optional[AutoImageProcessor] = None
        self.table_detector_model: Optional[AutoModelForObjectDetection] = None
        self.table_structure_model = None

        try:
            logger.info("Initializing table transformer models")
            self.table_detector_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.table_detector_model = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.table_structure_model = pipeline(
                "table-question-answering",
                model="microsoft/table-transformer-structure-recognition",
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as exc:
            logger.warning("Table transformer initialization skipped: %s", exc)

    @staticmethod
    def _get_text_hash(text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _is_scanned_pdf(page: fitz.Page) -> bool:
        text = page.get_text().strip()
        if len(text) < 30:
            return True
        blocks = page.get_text("blocks")
        text_blocks = [block for block in blocks if len(block[4].strip()) > 0]
        return len(text_blocks) < 3 and len(text) < 200

    def _extract_tables_with_transformer(self, image: Image.Image) -> List[str]:
        if (
            self.table_detector_processor is None
            or self.table_detector_model is None
            or self.table_structure_model is None
        ):
            return []

        tables: List[str] = []
        try:
            inputs = self.table_detector_processor(images=image, return_tensors="pt")
            outputs = self.table_detector_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.table_detector_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.7,
            )[0]

            for box in results["boxes"]:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(image.width, x2 + 10)
                y2 = min(image.height, y2 + 10)
                cropped = image.crop((x1, y1, x2, y2))
                result = self.table_structure_model(cropped)
                if result:
                    tables.append(f"[Table Content]\n{result}")
        except Exception as exc:
            logger.warning("Table extraction failed: %s", exc)
        return tables

    def _process_native_pdf(self, file_path: str) -> List[Document]:
        docs: List[Document] = []
        pdf = fitz.open(file_path)
        try:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                tables: List[str] = []
                try:
                    for table in page.find_tables():
                        df = table.to_pandas()
                        tables.append(f"[Table Content]\n{df.to_markdown(index=False)}")
                except Exception as exc:
                    logger.warning("PyMuPDF table extraction failed: %s", exc)

                full_content = "\n\n".join(part for part in [text, "\n\n".join(tables)] if part)
                if full_content.strip():
                    docs.append(
                        Document(
                            text=full_content.strip(),
                            metadata={
                                "source": os.path.basename(file_path),
                                "file_path": file_path,
                                "page": page_num,
                                "doc_type": "native_pdf",
                                "text_hash": self._get_text_hash(full_content),
                            },
                        )
                    )
        finally:
            pdf.close()
        return docs

    def _process_scanned_or_image(self, file_path: str) -> List[Document]:
        docs: List[Document] = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            pdf = fitz.open(file_path)
            try:
                for page_num, page in enumerate(pdf, start=1):
                    pix = page.get_pixmap(dpi=300)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    docs.extend(self._process_single_image(image, file_path, page_num))
            finally:
                pdf.close()
        else:
            image = Image.open(file_path).convert("RGB")
            docs.extend(self._process_single_image(image, file_path, 1))

        return docs

    def _process_single_image(self, image: Image.Image, file_path: str, page_num: int) -> List[Document]:
        docs: List[Document] = []
        ocr_result = self.ocr.ocr(np.array(image), cls=True)
        text_lines: List[str] = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) > 1:
                    text_lines.append(line[1][0])
        text = "\n".join(text_lines)

        tables = self._extract_tables_with_transformer(image)
        full_content = "\n\n".join(part for part in [text, "\n\n".join(tables)] if part)
        if full_content.strip():
            docs.append(
                Document(
                    text=full_content.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "page": page_num,
                        "doc_type": "scanned_or_image",
                        "text_hash": self._get_text_hash(full_content),
                    },
                )
            )
        return docs

    def _process_office_or_html(self, file_path: str) -> List[Document]:
        docs: List[Document] = []
        elements: List[Element] = partition(filename=file_path)
        text_parts: List[str] = []
        tables: List[str] = []

        for elem in elements:
            if isinstance(elem, Table):
                dataframe = getattr(elem.metadata, "dataframe", None)
                if dataframe is not None:
                    try:
                        tables.append(f"[Table Content]\n{dataframe.to_markdown(index=False)}")
                        continue
                    except Exception as exc:
                        logger.warning("Table markdown conversion failed: %s", exc)
                if elem.text:
                    tables.append(f"[Table Content]\n{elem.text}")
                continue

            if elem.text:
                text_parts.append(elem.text)

        full_content = "\n\n".join(part for part in ["\n\n".join(text_parts), "\n\n".join(tables)] if part)
        if full_content.strip():
            docs.append(
                Document(
                    text=full_content.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "page": 1,
                        "doc_type": "office_or_html",
                        "text_hash": self._get_text_hash(full_content),
                    },
                )
            )
        return docs

    def process_and_split(self, file_path: str) -> List[Document]:
        logger.info("Processing document: %s", os.path.basename(file_path))
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            pdf = fitz.open(file_path)
            try:
                if pdf.page_count == 0:
                    return []
                is_scanned = self._is_scanned_pdf(pdf[0])
            finally:
                pdf.close()
            return self._process_scanned_or_image(file_path) if is_scanned else self._process_native_pdf(file_path)

        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            return self._process_scanned_or_image(file_path)

        if ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".html", ".htm", ".txt"]:
            return self._process_office_or_html(file_path)

        raise ValueError(f"Unsupported file type: {ext}")
