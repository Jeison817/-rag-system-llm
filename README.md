🔍 RAG System — LLM + FAISS + Evaluation

Este proyecto presenta una implementación completa de un sistema RAG (Retrieval-Augmented Generation) aplicado a documentos PDF de normativa financiera peruana.

Forma parte de un proyecto académico de IA Generativa, donde se integran modelos de lenguaje, búsqueda híbrida y evaluación automática para generar respuestas fundamentadas.

🎯 Objetivo

Construir un sistema capaz de:

Consultar documentos PDF propios (Resolución SBS 1660-2025)
Generar respuestas con un LLM (Phi-2 cuantizado en 4-bit)
Reducir alucinaciones mediante grounding
Evaluar automáticamente la calidad de las respuestas
Proveer trazabilidad con citas por oración
🧠 Contexto académico

Este proyecto integra conceptos clave de IA Generativa:

Retrieval-Augmented Generation (RAG)
Búsqueda híbrida (semántica + léxica)
Modelos de lenguaje cuantizados
Evaluación automática de texto generado
Reducción de alucinaciones en LLMs

El sistema implementa un pipeline completo desde documentos hasta respuestas evaluadas.

⚙️ ¿Cómo funciona?

PDF → Chunking → Embeddings → FAISS + BM25 → Reranking → LLM → Evaluación

🧪 Ejemplo de uso
query = "¿Qué es Gestión de activos y pasivos?"

docs    = retrieve(query, k=10)
top     = rerank(query, docs, top_k=3)
context = "\n".join(c["text"] for c in top)

answer  = ask_llm(context, query)
cited   = cite_sentences(answer, top)

score, verdict = hallucination_guard(answer, top)

print("RESPUESTA:\n", answer)
print("\nGrounding score:", score, verdict)
🔖 Sistema de citas

El sistema incluye trazabilidad de fuentes:

Cada respuesta se divide en oraciones
Cada oración se vincula a un fragmento del PDF
Permite verificar el origen de cada afirmación

Esto mejora la confiabilidad y reduce alucinaciones.

📊 Evaluación

Se utilizan métricas automáticas para medir la calidad:

Grounding Score → alineación con el contexto (> 0.70 confiable)
BLEU-1 → coincidencia léxica
ROUGE-L → coherencia estructural
🛠️ Tecnologías
Python
HuggingFace Transformers
Microsoft Phi-2 (2.7B, 4-bit)
FAISS (HNSW)
BM25 (rank_bm25)
SentenceTransformers
Cross-Encoder (ms-marco-MiniLM-L-6-v2)
NLTK (BLEU, ROUGE)
BitsAndBytes
PyPDF
🚀 Ejecución
git clone https://github.com/Jeison817/rag-system-llm.git
cd rag-system-llm
pip install -r requirements.txt
jupyter notebook notebook/PF_IAGenerativa_phi2.ipynb

⚠️ Agrega tus PDFs en:

/content/docs
📊 Alcance del proyecto

Este proyecto es:

✔️ Un sistema RAG funcional
✔️ Enfocado en aprendizaje avanzado
✔️ Implementación completa de pipeline NLP
✔️ Sistema con trazabilidad de respuestas

No es:

❌ Un sistema en producción
❌ Escalable a gran volumen
❌ Optimizado para latencia
📈 Lo que demuestra
Integración de LLMs cuantizados en aplicaciones reales
Uso de FAISS + BM25 (búsqueda híbrida)
Reranking con Cross-Encoder
Evaluación automática de respuestas
Reducción de alucinaciones mediante grounding
🙋‍♂️ Autor
