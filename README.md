🔍 RAG System with LLM, FAISS & Evaluation

Retrieval-Augmented Generation aplicado a normativa financiera peruana

Proyecto académico de IA Generativa que implementa un sistema completo de Retrieval-Augmented Generation (RAG) para consultar documentos PDF y generar respuestas fundamentadas, reduciendo alucinaciones y asegurando trazabilidad.

🎯 Objetivo

Desarrollar un sistema capaz de:

📄 Consultar documentos propios (PDFs — Resolución SBS 1660-2025)
🧠 Generar respuestas usando un LLM cuantizado (Phi-2 4-bit)
🔗 Reducir alucinaciones mediante grounding y prompting controlado
📊 Evaluar automáticamente la calidad de las respuestas
📌 Garantizar trazabilidad con citas por oración
🧩 Arquitectura del Sistema
PDFs 
 → Chunking (size=400, overlap=100) 
 → Embeddings 
 → FAISS (HNSW) + BM25 
 → Cross-Encoder Reranker 
 → LLM (Phi-2 4-bit) 
 → Evaluación (Grounding + BLEU + ROUGE)
🧠 Tecnologías Utilizadas
Python
HuggingFace Transformers
Microsoft Phi-2 (2.7B, cuantizado 4-bit)
FAISS HNSW (búsqueda vectorial eficiente)
BM25 (rank_bm25, búsqueda léxica)
SentenceTransformers
paraphrase-multilingual-MiniLM-L12-v2
Cross-Encoder
ms-marco-MiniLM-L-6-v2
NLTK / BLEU / ROUGE
BitsAndBytes (cuantización)
PyPDF (extracción de texto)
🚀 Instalación y Ejecución
git clone https://github.com/Jeison817/rag-system-llm.git
cd rag-system-llm
pip install -r requirements.txt
jupyter notebook notebook/PF_IAGenerativa_phi2.ipynb

⚠️ Antes de ejecutar, agrega tus PDFs en:

/content/docs
🧪 Ejemplo de Uso
query = "¿Qué es Gestión de activos y pasivos?"

docs    = retrieve(query, k=10)
top     = rerank(query, docs, top_k=3)
context = "\n".join(c["text"] for c in top)

answer  = ask_llm(context, query)
cited   = cite_sentences(answer, top)

score, verdict = hallucination_guard(answer, top)

print("RESPUESTA:\n", answer)
print("\nGrounding score:", score, verdict)
📊 Evaluación del Sistema

El sistema incluye métricas automáticas para medir calidad y confiabilidad:

Métrica	Descripción	Interpretación
Grounding Score	% de términos alineados con el contexto	> 0.70 → ✅ Confiable
BLEU-1	Coincidencia léxica	0.1 – 0.3 aceptable
ROUGE-L	Coherencia estructural	Complementaria
📈 Resultados
=================================================================
Pregunta                       Grounding     Veredicto
-----------------------------------------------------------------
¿Qué flujos se consideran...     1.0000      ✅ FUNDAMENTADO
¿Cuál es el nivel mínimo...      0.7586      ✅ FUNDAMENTADO
¿Cuál es una función del...        ...       ✅ FUNDAMENTADO
=================================================================
🔖 Sistema de Citas (Trazabilidad)

Cada respuesta generada:

Se divide en oraciones
Se vincula automáticamente con chunks del documento
Permite auditar de dónde proviene cada afirmación

✔️ Esto reduce alucinaciones
✔️ Aumenta la confiabilidad del sistema

📂 Estructura del Proyecto
rag-system-llm/
├── notebook/
│   └── PF_IAGenerativa_phi2.ipynb
├── assets/
│   └── pipeline_arquitectura.png
├── requirements.txt
└── README.md
📌 Alcance
✔️ Este proyecto es:
Un sistema funcional de RAG aplicado a normativa financiera
Implementación completa de pipeline moderno de NLP
Uso de búsqueda híbrida (semántica + léxica)
Integración de LLMs cuantizados
Evaluación automática multi-métrica
Sistema de citas por oración (explainability)
❌ No es:
Un sistema en producción
Escalable a gran volumen (aún)
Optimizado para latencia
📈 Contribuciones Técnicas

Este proyecto demuestra:

Integración de LLMs ligeros (Phi-2 4-bit) en entornos limitados
Uso de FAISS HNSW + BM25 (hybrid search)
Implementación de reranking con Cross-Encoder
Diseño de métrica propia de Grounding Score
Pipeline completo de evaluación (BLEU + ROUGE + grounding)
Sistema de reducción de alucinaciones basado en contexto
🙋‍♂️ Autor
