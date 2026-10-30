import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName("EmpleoGEIH").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 1. Carga: datos simulados con la estructura de variables de la GEIH
np.random.seed(42)
n = 500
pdf = pd.DataFrame({
    "P6020": np.random.choice([1, 2], size=n, p=[0.52, 0.48]),
    "P6040": np.random.randint(18, 65, size=n),
    "OCI": np.random.choice([0, 1], size=n, p=[0.28, 0.72])
})
df = spark.createDataFrame(pdf)
print("Registros cargados:", df.count())

# 2. Limpieza
df = df.dropna().filter((col("P6040") >= 18) & (col("P6040") <= 64))

# 3. Transformacion
df = (df.withColumn("ocupacion", when(col("OCI") == 1, "Ocupado").otherwise("No Ocupado"))
        .withColumn("genero", when(col("P6020") == 1, "Hombre").otherwise("Mujer"))
        .withColumn("rango_edad",
            when(col("P6040") <= 28, "Jovenes (18-28)")
            .when(col("P6040") <= 45, "Adultos Jovenes (29-45)")
            .otherwise("Adultos (46-64)")))

# 4. EDA con Spark
print("Total tras limpieza:", df.count())
df.groupBy("ocupacion").count().show()
df.groupBy("rango_edad", "ocupacion").count().orderBy("rango_edad").show()
df.groupBy("genero", "ocupacion").count().orderBy("genero").show()

# 5. Visualizacion
edad = df.groupBy("rango_edad", "ocupacion").count().toPandas().pivot(index="rango_edad", columns="ocupacion", values="count")
gen = df.groupBy("genero", "ocupacion").count().toPandas().pivot(index="genero", columns="ocupacion", values="count")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
edad.plot(kind="bar", ax=ax[0], title="Ocupacion por rango de edad")
gen.plot(kind="bar", ax=ax[1], title="Ocupacion por genero")
fig.suptitle("Empleo y ocupacion - Datos simulados con estructura GEIH (DANE)")
plt.tight_layout()
plt.savefig("grafica_ocupacion.png")
print("Grafica guardada")
spark.stop()
