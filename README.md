# 🔍 RAG System: LLM + FAISS + Hybrid Search Evaluation

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** avanzado, especializado en el análisis de normativa financiera peruana (Resolución SBS). Utiliza una arquitectura híbrida de búsqueda y modelos cuantizados para garantizar precisión y trazabilidad.

---

## 🚀 Características Principales

* **🧠 Brain:** Microsoft **Phi-2** (2.7B) cuantizado en 4-bits para ejecución eficiente.
* **🔎 Búsqueda Híbrida:** Combinación de **FAISS** (densidad semántica) + **BM25** (fidelidad léxica).
* **🎯 Reranking:** Re-clasificación de contextos con **Cross-Encoder** para máxima relevancia.
* **🛡️ Hallucination Guard:** Sistema de evaluación automática con **Grounding Score** y métricas NLP (BLEU/ROUGE).
* **📍 Trazabilidad:** Citación automática por oración vinculada directamente a los fragmentos del PDF.

---

## 🛠️ Arquitectura del Pipeline

El sistema sigue un flujo de procesamiento de lenguaje natural de extremo a extremo:

1.  **Ingesta:** Carga de PDFs y segmentación (Chunking).
2.  **Indexing:** Generación de Embeddings y almacenamiento en FAISS.
3.  **Retrieval:** Búsqueda combinada (Semántica + Palabras clave).
4.  **Rerank:** Filtrado de los top-K mejores fragmentos.
5.  **Generation:** Respuesta del LLM fundamentada en el contexto.
6.  **Evaluación:** Cálculo de fidelidad (Grounding) y métricas de calidad.

---

## 🧪 Ejemplo de Implementación

```python
# Proceso de consulta y validación
query = "¿Qué es Gestión de activos y pasivos?"

# 1. Recuperación y Reranking
docs = retrieve(query, k=10)
top = rerank(query, docs, top_k=3)

# 2. Generación con Grounding
context = "\n".join(c["text"] for c in top)
answer = ask_llm(context, query)

# 3. Evaluación de alucinaciones
score, verdict = hallucination_guard(answer, top)

print(f"Respuesta: {answer}")
print(f"Grounding Score: {score} {verdict}")
