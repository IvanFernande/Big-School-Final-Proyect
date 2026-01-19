1. TF-IDF + OHE
2. Features del texto. 
3. modelo de embedding
4. Zero-shot (LLM)
5. deep-learning
5.1. Fine-tuning (HF)
5.2. SetFit (HF)


Metricas
1. precision
2. tiempo de ejecucion
3. precio dinero (coste modelos)

Departamentos
1. Embedding a partir de resumen de dep y tags

---

3 datas sets:
- train 70
- validation 20 seleccion del modelo en base a este data
- test 10 saco metrica del modelo final (simulacion en produccion)

CV o 3 sets, comparar y elegir uno con justificacion

porque LIME 

Metricas negocio: por cada prediccion mala, una penalizacion (dinero/tiempo)

En el repo para reproducirlo y como configurar