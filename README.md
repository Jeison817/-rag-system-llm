# 📚 Sistema RAG con LLM y Métricas de Evaluación

Un pipeline completo de **Retrieval-Augmented Generation (RAG)** implementado en Python que combina búsqueda semántica híbrida, reranking y generación de respuestas con el modelo **Microsoft Phi-2 (4-bit)**, diseñado para responder preguntas a partir de documentos PDF con alta fidelidad y trazabilidad.

---

## 🧠 ¿Qué hace este proyecto?

El sistema responde preguntas usando **exclusivamente** la información contenida en los documentos proporcionados, minimizando alucinaciones y asegurando trazabilidad por oración en cada respuesta.

### Pipeline completo:

```
PDFs → Chunking → Embeddings → FAISS HNSW + BM25 → Reranker → Phi-2 (4-bit) → Respuesta con citas + métricas
```

---

## ⚙️ Componentes del Sistema

| Etapa | Tecnología | Descripción |
|---|---|---|
| Carga de documentos | `pypdf` | Extracción de texto página por página |
| Chunking | Custom | Fragmentación con overlap (size=400, overlap=100) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Vectores semánticos multilingües |
| Búsqueda semántica | `FAISS HNSW` | Índice jerárquico de grafos navegables |
| Búsqueda léxica | `BM25Okapi` | Recuperación clásica por palabras clave |
| Búsqueda híbrida | FAISS + BM25 | Score = α × FAISS + (1−α) × BM25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Evaluación precisa de pares (query, chunk) |
| Generación | `microsoft/phi-2` (4-bit) | LLM compacto con cuantización BitsAndBytes |
| Citas por oración | Custom | Trazabilidad de fuentes en la respuesta |
| Evaluación | Grounding Score + BLEU + ROUGE | Métricas automáticas de calidad |

---

## 📦 Instalación

```bash
pip install -U transformers accelerate bitsandbytes sentencepiece
pip install faiss-cpu sentence-transformers datasets
pip install rank_bm25 rouge_score nltk pypdf
```

> **Requisito:** GPU recomendada para cargar Phi-2 en 4-bit. Compatible con Google Colab (T4 o superior).

---

## 🚀 Uso

### 1. Subir documentos

Crea la carpeta `/content/docs` y sube tus archivos PDF:

```python
import os
os.makedirs("/content/docs", exist_ok=True)
# Sube tus PDFs a la carpeta docs en el panel de archivos
```

### 2. Ejecutar el notebook

Ejecuta las celdas en orden. El pipeline completo se activa en la **Sección 13 (Consulta)**:

```python
query = "¿Qué es Gestión de activos y pasivos?"

docs    = retrieve(query, k=10)        # FAISS + BM25 híbrido
top     = rerank(query, docs, top_k=3) # Cross-Encoder reranking
context = "\n".join(c["text"] for c in top)
answer  = ask_llm(context, query)      # Generación con Phi-2

score, verdict = hallucination_guard(answer, top)
print("RESPUESTA:\n", answer)
print("\nGrounding score:", score, verdict)
```

### 3. Consultas en lote

Puedes evaluar múltiples preguntas con referencias usando la sección **"Varias consultas"**, que genera automáticamente un reporte con BLEU, ROUGE y Grounding Score.

---

## 📊 Métricas de Evaluación

| Métrica | Descripción | Umbral |
|---|---|---|
| **Grounding Score** | % de términos de la respuesta presentes en el contexto | ✅ ≥ 0.70 |
| **BLEU-1** | Coincidencia léxica con respuesta de referencia | 0.1–0.3 aceptable |
| **ROUGE-L** | Secuencia lógica más larga compartida con la referencia | Mayor = mejor |

### Veredictos del Grounding Score:

- `✅ FUNDAMENTADO` → score ≥ 0.70
- `⚠️ ADVERTENCIA` → 0.40 ≤ score < 0.70
- `❌ ALUCINACIÓN` → score < 0.40

---

## 🧪 Resultados de Ejemplo

Resultados obtenidos sobre la Resolución SBS 1660-2025:

| Pregunta | Grounding | BLEU-1 | ROUGE-L | Veredicto |
|---|---|---|---|---|
| Flujos en instrumentos de deuda (ASA) | 1.0000 | 0.1636 | 0.2466 | ✅ FUNDAMENTADO |
| Nivel mínimo del indicador GHO | 0.7586 | 0.0989 | 0.1525 | ✅ FUNDAMENTADO |
| Función del Comité de GAP | 0.8750 | 0.2917 | 0.3077 | ✅ FUNDAMENTADO |

---

## 🏗️ Estructura del Proyecto

```
📁 /content/docs/         ← PDFs de entrada
📓 PF_IAGenerativa_phi2.ipynb  ← Notebook principal
```

### Secciones del Notebook:

```
0. Instalación de librerías
1. Preparación de carpeta de documentos
2. Carga de PDFs
3. Chunking con Overlap
4. Embeddings
5. FAISS HNSW
6. Búsqueda Híbrida (BM25 + FAISS)
7. Reranker Cross-Encoder
8. LLM Microsoft Phi-2 (4-bit)
9. Prompt + Generación de respuestas
10. Citation per Sentence
11. Hallucination Guard + Grounding Score
12. BLEU + ROUGE
13. Consulta individual
14. Consultas en lote
15. Reporte final
```

---

## 🔧 Parámetros Clave

| Parámetro | Valor | Descripción |
|---|---|---|
| `chunk_size` | 400 | Tamaño de cada fragmento de texto |
| `overlap` | 100 | Solapamiento entre chunks |
| `k` (retrieval) | 10 | Chunks recuperados por FAISS + BM25 |
| `top_k` (reranking) | 3 | Chunks finales enviados al LLM |
| `alpha` | 0.7 | Peso de FAISS vs BM25 en búsqueda híbrida |
| `M` (HNSW) | 32 | Conexiones por nodo en el grafo |
| `efConstruction` | 200 | Esfuerzo al construir el índice |
| `efSearch` | 50 | Esfuerzo al buscar vecinos |
| `max_new_tokens` | 200 | Longitud máxima de respuesta del LLM |
| `temperature` | 0.3 | Temperatura de generación (baja = más determinista) |

---

## 📝 Notas

- **Phi-2 vs Qwen:** Phi-2 no soporta `apply_chat_template`, por lo que se usa el formato `Instruct: ... Output:` propio del modelo.
- **Multilingüe:** El modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` soporta español e inglés.
- **Cuantización 4-bit:** Reduce significativamente el uso de memoria GPU sin pérdida notable de calidad.

---

## 📄 Autores
- **Jeison Josimar Contreras Meza**
