📚 Sistema RAG con LLM y Evaluación de Respuestas

Implementación completa de un sistema Retrieval-Augmented Generation (RAG) en Python, que integra búsqueda híbrida, reranking y generación de lenguaje natural usando Microsoft Phi-2 (cuantizado en 4-bit).

El sistema está diseñado para responder preguntas basadas exclusivamente en documentos PDF, reduciendo alucinaciones y garantizando trazabilidad a nivel de oración.

🧠 Descripción General

Este proyecto implementa un pipeline RAG moderno que combina:

Recuperación semántica (vectorial)
Recuperación léxica (BM25)
Reordenamiento (reranking)
Generación controlada con LLM
Evaluación automática de calidad
🔄 Pipeline
PDFs → Chunking → Embeddings → FAISS (HNSW) + BM25 → Reranking → Phi-2 → Respuesta con citas + métricas
🚀 Características Principales

✅ Búsqueda híbrida (semántica + léxica)
✅ Reranking con Cross-Encoder
✅ Generación eficiente con LLM cuantizado (4-bit)
✅ Citas por oración (trazabilidad completa)
✅ Evaluación automática (Grounding, BLEU, ROUGE)
✅ Soporte multilingüe (español / inglés)

⚙️ Arquitectura del Sistema
Etapa	Tecnología	Función
Ingesta	pypdf	Extracción de texto desde PDFs
Chunking	Custom	División en fragmentos con solapamiento
Embeddings	sentence-transformers	Representación vectorial semántica
Índice vectorial	FAISS HNSW	Búsqueda eficiente por similitud
Índice léxico	BM25Okapi	Recuperación basada en palabras clave
Fusión híbrida	FAISS + BM25	Score combinado ponderado
Reranking	Cross-Encoder	Reordenamiento de relevancia
Generación	microsoft/phi-2	Respuestas naturales
Evaluación	BLEU, ROUGE, Grounding	Medición de calidad
📦 Instalación
pip install -U transformers accelerate bitsandbytes sentencepiece
pip install faiss-cpu sentence-transformers datasets
pip install rank_bm25 rouge_score nltk pypdf

⚠️ Recomendación: usar GPU (Google Colab T4 o superior) para ejecutar Phi-2 en 4-bit.

▶️ Uso
1. Preparar documentos
import os
os.makedirs("/content/docs", exist_ok=True)

Sube tus archivos PDF a la carpeta /content/docs.

2. Ejecutar pipeline
query = "¿Qué es Gestión de activos y pasivos?"

docs    = retrieve(query, k=10)
top     = rerank(query, docs, top_k=3)
context = "\n".join(c["text"] for c in top)

answer  = ask_llm(context, query)

score, verdict = hallucination_guard(answer, top)

print("RESPUESTA:\n", answer)
print("\nGrounding score:", score, verdict)
3. Evaluación en lote

El notebook incluye una sección para evaluar múltiples preguntas y generar automáticamente métricas:

BLEU
ROUGE
Grounding Score
📊 Métricas de Evaluación
Métrica	Descripción	Interpretación
Grounding Score	Proporción de términos sustentados en el contexto	≥ 0.70 → confiable
BLEU-1	Coincidencia léxica	0.1 – 0.3 aceptable
ROUGE-L	Coincidencia estructural	Mayor = mejor
🧾 Veredictos
✅ Fundamentado → ≥ 0.70
⚠️ Advertencia → 0.40 – 0.69
❌ Alucinación → < 0.40
🧪 Resultados de Ejemplo

Evaluación sobre documentos regulatorios:

Consulta	Grounding	Veredicto
Flujos en instrumentos de deuda	1.000	✅
Indicador GHO mínimo	0.758	✅
Función del comité GAP	0.875	✅
🏗️ Estructura del Proyecto
📁 docs/
📓 PF_IAGenerativa_phi2.ipynb
📌 Secciones del Notebook
0. Instalación
1. Ingesta de documentos
2. Chunking
3. Embeddings
4. FAISS
5. Búsqueda híbrida
6. Reranking
7. LLM (Phi-2)
8. Generación
9. Citas por oración
10. Grounding Score
11. BLEU / ROUGE
12. Consulta individual
13. Consultas en lote
14. Reporte final
🔧 Parámetros Importantes
Parámetro	Valor	Descripción
chunk_size	400	Tamaño del fragmento
overlap	100	Solapamiento
k	10	Recuperación inicial
top_k	3	Contexto final
alpha	0.7	Peso FAISS vs BM25
efSearch	50	Precisión en búsqueda
temperature	0.3	Control de aleatoriedad
🧩 Consideraciones Técnicas

Phi-2 usa formato tipo:

Instruct: ...
Output:
No requiere chat_template
Cuantización en 4-bit reduce consumo de memoria
Embeddings multilingües permiten consultas en español
🎯 Aplicaciones
Sistemas de preguntas sobre documentos legales
Asistentes académicos
Análisis de normativa
QA sobre PDFs empresariales
📄 Licencia

Proyecto desarrollado con fines académicos y de investigación.
