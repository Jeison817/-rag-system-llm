# 🔍 Sistema RAG con LLM, FAISS y Evaluación

Proyecto que implementa un sistema completo de **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre documentos PDF utilizando modelos de lenguaje y recuperación semántica.

Este proyecto forma parte de un trabajo académico de IA Generativa, donde se integran técnicas modernas de NLP, búsqueda híbrida y evaluación automática.

---

## 🎯 Objetivo

Construir un sistema capaz de:

* Consultar documentos propios (PDF)
* Generar respuestas con un LLM
* Reducir alucinaciones mediante grounding
* Evaluar automáticamente la calidad de las respuestas

---

## ⚙️ Arquitectura del sistema

```
PDFs → Chunking → Embeddings → FAISS + BM25 → Reranker → LLM → Evaluación
```

---

## 🧠 Tecnologías utilizadas

* Python
* HuggingFace Transformers
* FAISS (búsqueda vectorial)
* BM25 (búsqueda por palabras clave)
* SentenceTransformers
* Qwen 1.5 (LLM cuantizado 4-bit)
* NLTK / ROUGE (evaluación)

---

## 🚀 Cómo ejecutarlo

```bash
git clone https://github.com/Jeison817/rag-system-llm.git
cd rag-system-llm
pip install -r requirements.txt
jupyter notebook notebook/ProyectoFinal_IAGenerativa.ipynb
```

---

## 🧪 Ejemplo de uso

```python
query = "¿Cuáles son las tareas básicas de un DBA?"

docs = retrieve(query, k=10)
top = rerank(query, docs, top_k=3)
context = "\n".join(c["text"] for c in top)

answer = ask_llm(context, query)
print(answer)
```

---

## 🖼️ Demo

![Pipeline](assets/pipeline_arquitectura.png)

---

## 📊 Evaluación

El sistema incluye métricas automáticas como:

* BLEU-1
* ROUGE-L
* Grounding Score (detección de alucinaciones)

---

## 📂 Estructura del proyecto

```
rag-system-llm/
├── notebook/
│   └── ProyectoFinal_IAGenerativa.ipynb
├── assets/
│   └── pipeline_arquitectura.png
├── requirements.txt
└── README.md
```

---

## 📌 Alcance del proyecto

Este proyecto es:

* ✔️ Un sistema funcional de RAG
* ✔️ Enfocado en aprendizaje avanzado
* ✔️ Aplicación de IA generativa moderna

No es:

* ❌ Un sistema en producción
* ❌ Optimizado para grandes volúmenes

---

## 📈 Lo que demuestra

* Integración de LLMs con recuperación de información
* Uso de FAISS y búsqueda híbrida
* Implementación de reranking
* Evaluación de modelos generativos
* Reducción de alucinaciones en IA

---

## 🙋‍♂️ Autor

**Jeison Contreras**
🎓 Matemática Pura + Ciencia de Datos
🔗 GitHub: https://github.com/Jeison817
