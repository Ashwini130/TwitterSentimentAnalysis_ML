import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer,StopWordsRemover,IDF}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession


object SentimentPredictionModel{

  def main(args:Array[String]):Unit={
    val spark = SparkSession.builder.appName("PredictSentiment").getOrCreate()

    val regexTokenizer = new RegexTokenizer().setInputCol("cleaned_tweet").setOutputCol("words").setPattern("\\W")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val hashingTF = new HashingTF().setInputCol("filtered_words").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,remover,hashingTF,idf, lr))

    val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.elasticNetParam, Array(0.8, 0.10)).addGrid(lr.maxIter,Array(10,100)).build()

    val tweet_df = spark.read.format("csv").option("inferSchema","true").option("header","true").load("/common4all/bigdatapgp/assignment12/tweets.csv")
    val tweetRDD = tweet_df.rdd.map {
      row => val tweetStr = row.getAs[String]("tweet")
        val cleaned_tweet = tweetStr.trim.replaceAll("[\\x01-\\x19,\\x21-\\x40,\\x7B-\\xFF,\\[,\\]]", " ").replaceAll(" +"," ").trim
        Row.fromSeq(row.toSeq.toList :+ cleaned_tweet)
    }

    val tweet_schema = StructType(tweet_df.schema.fields ++ Array(StructField("cleaned_tweet", StringType, true)))
    val Cleaned_tweetDF = spark.sqlContext.createDataFrame(tweetRDD,tweet_schema)
    val Array(train, test) = Cleaned_tweetDF.randomSplit(Array(0.7,0.3))


    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3).setParallelism(2)
    val cvModel = cv.fit(train)

    val test_transform = cvModel.transform(test)
    val pred_label = test_transform.withColumn("label",test_transform.col("label").cast(DoubleType)).select("prediction","label")
    val metrics = new MulticlassMetrics(pred_label.rdd.map(x=>(x.getAs[Double]("prediction"),x.getAs[Double]("label"))))
    println(metrics.accuracy)
    println(metrics.confusionMatrix)
  }
}