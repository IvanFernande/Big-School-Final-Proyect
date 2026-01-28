Grafico de caja, entender bien como se visualiza (mas o menos pero me puedo hacer guión)
 
Enterarme bien que es CV (que hace)
Por si tengo que repasar:
La validación cruzada (CV) es una técnica para estimar el rendimiento de un modelo de forma más robusta que un único split train/test. Se divide el conjunto de entrenamiento en K partes (folds) y se entrena K veces, dejando cada vez una parte para validar y las restantes para entrenar. Luego se promedian las métricas obtenidas. Así se reduce la dependencia del azar del split y se puede comparar modelos o ajustar hiperparámetros sin tocar el conjunto de test final, que se reserva para la evaluación definitiva.


Ver que aportan realmente:
1. ngram_range=(1,2)
2. min_df=2
3. sublinear_tf=True
4. smooth_idf=True

En TF‑IDF usamos ngram_range=(1,2) para capturar no solo palabras sueltas sino también frases cortas con más contexto. min_df=2 elimina términos muy raros que suelen ser ruido, y sublinear_tf=True reduce el peso de palabras demasiado repetidas. smooth_idf=True estabiliza el peso de términos poco frecuentes para evitar extremos.


que es n_hash

n_hash es la dimensión del vector en el método de hashing: cada categoría (o token) se transforma con una función hash y se asigna a una de n_hash posiciones. Con n_hash=2048, todas las categorías se proyectan en un vector de 2048 columnas; si hay pocas categorías, muchas columnas quedan vacías y no aporta ventaja. El hashing es útil cuando la cardinalidad es alta o cambia con el tiempo, porque evita crecer el número de columnas como en OHE, a costa de posibles colisiones.


que es estratificacion

La estratificación es una forma de hacer el split (train/test o CV) manteniendo la misma proporción de clases que en el dataset completo.
Así evitas que una clase quede infra‑representada en train o test, lo cual sesga las métricas. En clasificación con clases desbalanceadas, es clave para tener resultados fiables. Ejemplo: si Low es ~20% del total, en train y en test también será ~20%. No inventa datos ni balancea: solo preserva la proporción para que el modelo vea todas las clases y la evaluación sea justa.


Entender bien el parrafo de objetivo de embeddings

Queríamos no cambiar el modelo, solo cambiar la forma de convertir texto a números y ver si la semántica extra de los embeddings aporta una mejora real en macro‑F1 o recall High.


parametro C

C es el parámetro de regularización en SVM y Logistic Regression.
Controla el equilibrio entre ajustar bien los datos y mantener el modelo simple:

- C alto → poca regularización → el modelo intenta ajustar más el training (más riesgo de sobreajuste).
- C bajo → mucha regularización → el modelo es más “suave” (mejor generalización, pero puede infra‑ajustar).

se prueba C en varios valores para ver el mejor trade‑off entre macro‑F1 y recall High.


AP High:

“Ordenar por probabilidad” significa que el modelo asigna a cada ticket un score de ser High. Si los ordenas de mayor a menor score, los que están arriba son los que el modelo considera más críticos.

AP High mide qué tan buena es esa lista ordenada:
si al revisar los primeros tickets casi todos son realmente High, AP es alto; si se mezclan muchos que no lo son, AP baja.


Entender que es cada uno: iteraciones {5,10}, épocas {1,2}, batch {16,32} (GPU) / {8,16} (CPU), LR {2e‑5, 5e‑5}.

iteraciones = cuántas rondas de entrenamiento contrastivo (Entrenamiento contrastivo: el modelo aprende con pares; textos de la misma clase se acercan en el embedding y de clases distintas se alejan, mejorando la separabilidad con pocos datos.) haces; épocas = cuántas veces recorres el dataset completo.
Batch size = cuántos ejemplos entrenas a la vez (más grande = más estable/rápido si hay memoria).
LR = tamaño del paso de aprendizaje (alto aprende rápido pero puede ser inestable; bajo es más estable).

Entender bien que se hace cuando se hace downsample y cuando se hace oversample

Oversample: se aumenta artificialmente la clase minoritaria (repitiendo ejemplos o generando nuevos) para equilibrar las proporciones.
Ventaja: el modelo ve más ejemplos de la clase rara.
Riesgo: puede sobreajustar a ejemplos repetidos.

Downsample: se reduce la clase mayoritaria eliminando ejemplos para igualar tamaños.
Ventaja: entrenamiento más rápido y balanceado.
Riesgo: pierdes información útil de la clase mayoritaria.


**Entender bien que es cada cosa (TF-IDF,embeddings,SetFit)**

TF‑IDF y embeddings son formas de codificar texto.
SetFit no es otra codificación, sino un método de entrenamiento: parte de un encoder de embeddings y lo afina con pares contrastivos para adaptarlo al dominio, y luego entrena un clasificador encima.

TF‑IDF y embeddings son formas de codificar texto.
SetFit no es otra codificación, sino un método de entrenamiento: parte de un encoder de embeddings y lo afina con pares contrastivos para adaptarlo al dominio, y luego entrena un clasificador encima.
O sea, SetFit usa embeddings, pero los ajusta al dataset en lugar de usarlos “tal cual”.


Explicar porque los modelos lineales y porque usar tf-idf, embeddings etc, porque de cada cosa