RAG System with LLM, FAISS and Evaluation

Sistema de Retrieval-Augmented Generation (RAG) aplicado a documentos PDF de normativa financiera peruana.
El proyecto integra modelos de lenguaje, recuperación híbrida y evaluación automática para generar respuestas fundamentadas y trazables.

Objetivo

Desarrollar un sistema capaz de:

Consultar documentos PDF propios
Generar respuestas usando un modelo de lenguaje cuantizado (Phi-2 4-bit)
Reducir alucinaciones mediante grounding
Evaluar automáticamente la calidad de las respuestas
Proveer trazabilidad mediante citas por oración
Arquitectura del sistema
PDFs 
→ Chunking (size=400, overlap=100) 
→ Embeddings 
→ FAISS (HNSW) + BM25 
→ Reranking (Cross-Encoder) 
→ LLM (Phi-2 4-bit) 
→ Evaluación
Tecnologías utilizadas
Python
HuggingFace Transformers
Microsoft Phi-2 (2.7B parámetros, 4-bit)
FAISS (HNSW)
BM25 (rank_bm25)
SentenceTransformers
Cross-Encoder (ms-marco-MiniLM-L-6-v2)
NLTK (BLEU, ROUGE)
BitsAndBytes
PyPDF
Instalación y ejecución
git clone https://github.com/Jeison817/rag-system-llm.git
cd rag-system-llm
pip install -r requirements.txt
jupyter notebook notebook/PF_IAGenerativa_phi2.ipynb

Antes de ejecutar, agrega tus documentos en:

/content/docs
Ejemplo de uso
query = "¿Qué es Gestión de activos y pasivos?"

docs    = retrieve(query, k=10)
top     = rerank(query, docs, top_k=3)
context = "\n".join(c["text"] for c in top)

answer  = ask_llm(context, query)
cited   = cite_sentences(answer, top)

score, verdict = hallucination_guard(answer, top)

print("RESPUESTA:\n", answer)
print("\nGrounding score:", score, verdict)
Evaluación

El sistema incluye métricas automáticas para medir calidad y confiabilidad:

Métrica	Descripción	Interpretación
Grounding Score	% de términos alineados al contexto	> 0.70 confiable
BLEU-1	Coincidencia léxica con referencia	0.1 – 0.3 aceptable
ROUGE-L	Coherencia estructural	Complementaria
Resultados
=================================================================
Pregunta                       Grounding     Veredicto
-----------------------------------------------------------------
¿Qué flujos se consideran...     1.0000      FUNDAMENTADO
¿Cuál es el nivel mínimo...      0.7586      FUNDAMENTADO
¿Cuál es una función del...        ...       FUNDAMENTADO
=================================================================
Sistema de citas

Cada respuesta generada:

Se divide en oraciones
Se vincula con fragmentos del documento (chunks)
Permite identificar la fuente de cada afirmación

Esto mejora la interpretabilidad y reduce alucinaciones.

Estructura del proyecto
rag-system-llm/
├── notebook/
│   └── PF_IAGenerativa_phi2.ipynb
├── assets/
│   └── pipeline_arquitectura.png
├── requirements.txt
└── README.md
Alcance

Este proyecto es:

Un sistema funcional de RAG
Implementación de búsqueda híbrida (semántica + léxica)
Integración de modelos cuantizados
Pipeline completo de evaluación
Sistema con trazabilidad de respuestas

No es:

Un sistema en producción
Escalable a gran volumen
Optimizado para latencia
Contribuciones

El proyecto demuestra:

Integración de LLMs ligeros en entornos limitados
Uso combinado de FAISS y BM25
Reranking con Cross-Encoder
Evaluación automática con múltiples métricas
Reducción de alucinaciones mediante grounding
Autor
