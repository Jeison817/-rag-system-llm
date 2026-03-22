Ahora tengo todo el contexto del notebook. El LLM cambió de Qwen a Microsoft Phi-2, los PDFs son sobre normativa financiera (Resolución SBS 1660-2025), y hay detalles adicionales como citas por oración y el formato de prompt Instruct/Output. Aquí va el README actualizado:

🔍 Sistema RAG con LLM, FAISS y Evaluación
Proyecto que implementa un sistema completo de Retrieval-Augmented Generation (RAG) para responder preguntas sobre documentos PDF utilizando modelos de lenguaje y recuperación semántica.
Este proyecto forma parte de un trabajo académico de IA Generativa, donde se integran técnicas modernas de NLP, búsqueda híbrida y evaluación automática aplicadas a normativa financiera peruana.

🎯 Objetivo
Construir un sistema capaz de:

Consultar documentos propios (PDFs de normativa financiera — Resolución SBS 1660-2025)
Generar respuestas con un LLM cuantizado en 4-bit
Reducir alucinaciones mediante grounding y prompting estricto
Evaluar automáticamente la calidad y trazabilidad de las respuestas


⚙️ Arquitectura del sistema
PDFs → Chunking (size=400, overlap=100) → Embeddings → FAISS HNSW + BM25 → Cross-Encoder Reranker → LLM (Phi-2 4-bit) → Evaluación

🧠 Tecnologías utilizadas

Python
HuggingFace Transformers
Microsoft Phi-2 (~2.7B parámetros, cuantizado 4-bit)
FAISS HNSW (búsqueda vectorial jerárquica)
BM25 — rank_bm25 (búsqueda por palabras clave)
SentenceTransformers — paraphrase-multilingual-MiniLM-L12-v2
Cross-Encoder — ms-marco-MiniLM-L-6-v2 (reranking)
NLTK / BLEU / ROUGE (evaluación automática)
BitsAndBytes (cuantización 4-bit)
pypdf (extracción de texto de PDFs)


🚀 Cómo ejecutarlo
bashgit clone https://github.com/Jeison817/rag-system-llm.git
cd rag-system-llm
pip install -r requirements.txt
jupyter notebook notebook/PF_IAGenerativa_phi2.ipynb

⚠️ Sube tus PDFs a la carpeta /content/docs antes de ejecutar el pipeline.


🧪 Ejemplo de uso
pythonquery = "¿Qué es Gestión de activos y pasivos?"
docs    = retrieve(query, k=10)
top     = rerank(query, docs, top_k=3)
context = "\n".join(c["text"] for c in top)
answer  = ask_llm(context, query)
cited   = cite_sentences(answer, top)
score, verdict = hallucination_guard(answer, top)

print("RESPUESTA:\n", answer)
print("\nGrounding score:", score, verdict)
```

---

## 📊 Evaluación

El sistema incluye métricas automáticas organizadas en un reporte final:

| Métrica | Descripción | Rango esperado |
|---|---|---|
| **Grounding Score** | % de términos de la respuesta anclados al contexto | > 0.70 ✅ FUNDAMENTADO |
| **BLEU-1** | Coincidencia léxica con respuesta de referencia | 0.1 – 0.3 aceptable |
| **ROUGE-L** | Coherencia estructural (LCS) | Complementaria al BLEU |

### Resultados obtenidos
```
=================================================================
Pregunta                       Grounding     BLEU-1   ROUGE-L  Veredicto
-----------------------------------------------------------------
¿Qué flujos se consideran...     1.0000        ...       ...    ✅ FUNDAMENTADO
¿Cuál es el nivel mínimo...      0.7586        ...       ...    ✅ FUNDAMENTADO
¿Cuál es una función del...        ...         ...       ...    ✅ FUNDAMENTADO
=================================================================
```

---

## 🔖 Sistema de citas por oración

El sistema incluye trazabilidad de fuentes: cada oración de la respuesta es vinculada al chunk del PDF que la respalda mediante overlap léxico, permitiendo auditar la procedencia de cada afirmación.

---

## 📂 Estructura del proyecto
```
rag-system-llm/
├── notebook/
│   └── PF_IAGenerativa_phi2.ipynb
├── assets/
│   └── pipeline_arquitectura.png
├── requirements.txt
└── README.md

📌 Alcance del proyecto
Este proyecto es:

✔️ Un sistema funcional de RAG sobre normativa financiera (SBS 1660-2025)
✔️ Enfocado en aprendizaje avanzado de IA Generativa
✔️ Aplicación de búsqueda híbrida, reranking y evaluación automática
✔️ Con trazabilidad de fuentes por oración (citation per sentence)

No es:

❌ Un sistema en producción
❌ Optimizado para grandes volúmenes de datos


📈 Lo que demuestra

Integración de LLMs cuantizados (Phi-2 4-bit) con recuperación de información
Uso de FAISS HNSW y búsqueda híbrida semántica + palabras clave
Implementación de reranking con Cross-Encoder
Sistema de citas automáticas por oración para trazabilidad
Evaluación multi-métrica (Grounding, BLEU, ROUGE)
Reducción de alucinaciones mediante prompting estricto y grounding score


🙋‍♂️ Autores:
