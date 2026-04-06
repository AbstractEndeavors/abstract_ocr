## Part of the Abstract Media Intelligence Platform

This module provides layout-aware OCR as part of a larger media processing system.

abstract_ocr focuses on extraction:
- multi-engine OCR (Tesseract / EasyOCR / PaddleOCR)
- column detection and region segmentation
- structured, position-aware text output

Full system: https://github.com/AbstractEndeavors/abstract-media-intelligence

---

## **abstract_ocr / layout_ocr — Layout-Aware OCR Pipeline**

A structured OCR pipeline designed for **layout-aware text extraction from complex documents**, combining preprocessing, column detection, region classification, and ordered OCR assembly.

Built to handle:

* multi-column PDFs
* mixed-content layouts (text, figures, captions)
* noisy or scanned documents
* large-scale document ingestion pipelines

---

## 🔹 What This System Is

This is not a simple OCR wrapper — it is a **typed, multi-stage processing pipeline**:

* transforms raw images into structured page representations
* detects document layout (columns, headers, regions)
* classifies content blocks (text, figures, captions)
* applies OCR at the region level
* reconstructs output in correct reading order

The system is designed for **deterministic, reproducible extraction** rather than heuristic text scraping.

---

## 🔹 Pipeline Overview

```text
PageImage
    ↓
Preprocess (denoise + binarize)
    ↓
Layout Detection
    ├─ Column detection (vertical projection)
    ├─ Header detection
    └─ Region classification (text / figure / caption)
    ↓
Region OCR
    ├─ Crop per region
    ├─ Apply OCR (Tesseract)
    ├─ Fallback to column-level OCR if needed
    ↓
Reading Order Assembly
    ↓
OCRResult (structured blocks + raw text)
```

---

## 🔹 Core Capabilities

* **Layout Detection**

  * Column detection via vertical projection valleys
  * Header segmentation via density scanning
  * Multi-column classification (single / dual / mixed)

* **Region Classification**

  * Connected-component analysis
  * Density-based classification (text vs figure vs caption)
  * Column-aware region assignment

* **Region-Level OCR**

  * OCR applied per detected block (not full-page)
  * Adaptive Tesseract configuration by region type
  * Automatic fallback to column-level OCR when detection fails

* **Reading Order Reconstruction**

  * Column-aware ordering
  * Top-to-bottom sequencing within columns
  * Header/body/caption prioritization

* **Typed Pipeline Execution**

  * All steps validated via explicit input/output types
  * Registry-driven execution model
  * No implicit coupling between pipeline stages

---

## 🔹 Architecture

The pipeline is built around a **step registry + type-safe execution chain**:

* Each step declares:

  * input type
  * output type
* The pipeline validates compatibility before execution
* Execution is explicit, deterministic, and observable

Example chain:

```python
["preprocess", "detect_layout", "ocr_regions"]
```

Each step is independently replaceable and composable.

---

## 🔹 Key Design Decisions

### **Typed Data Flow**

All intermediate results are structured dataclasses:

* `PageImage`
* `PreprocessedImage`
* `LayoutDetection`
* `OCRResult`

No ad-hoc dictionaries — ensures:

* traceability
* consistency
* debuggability

---

### **Layout-First OCR**

OCR is applied **after structure is understood**, not before.

This prevents:

* column interleaving
* incorrect reading order
* misclassification of content

---

### **Fallback Over Failure**

If region detection fails:

* system falls back to column-level OCR
* ensures output is still usable

---

### **Determinism Over Heuristics**

* explicit thresholds (config-driven)
* no hidden behavior
* reproducible results across runs

---

## 🔹 Why This Exists

Traditional OCR pipelines:

* ignore layout
* operate on full pages
* produce inconsistent reading order
* fail silently on complex documents

This system:

* understands document structure
* isolates regions before OCR
* enforces reading order
* produces structured outputs suitable for downstream systems

---

## 🔹 Example Use Cases

* PDF → structured text extraction
* research document ingestion pipelines
* financial filings parsing
* multi-column article extraction
* preprocessing for NLP / LLM pipelines
* search indexing and document analysis

---

## 🔹 Integration Context

This module is designed to plug into:

* document ingestion systems
* OCR + NLP pipelines (e.g. abstract_hugpy)
* search and indexing systems
* large-scale document processing workflows

---

## 🔹 Design Philosophy

* **Structure before extraction**
* **Determinism over convenience**
* **Typed pipelines over implicit flows**
* **Fallback over failure**

---
