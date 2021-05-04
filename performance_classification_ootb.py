import pandas as pd

import modelop.monitors.performance as performance
import modelop.schema.infer as infer


# modelop.init
def init(job_json):
    
    # Extract input schema from job JSON
    input_schema_definition = infer.extract_input_schema(job_json)

    print("input_schema_definition", input_schema_definition, flush=True)
    
    # Get monitoring parameters from schema
    global MONITORING_PARAMETERS
    MONITORING_PARAMETERS = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )


# modelop.metrics
def metrics(dataframe):

    # Get identifier_columns from MONITORING_PARAMETERS
    identifier_columns = MONITORING_PARAMETERS["identifier_columns"]

    # Initialize Volumetrics monitor with 1st input DataFrame
    model_evaluator = performance.ModelEvaluator(
        dataframe=dataframe,
        score_column=MONITORING_PARAMETERS["score_column"],
        label_column=MONITORING_PARAMETERS["label_column"],
    )

    # Compare DataFrames on identifier_columns
    classification_metrics = model_evaluator.evaluate_performance(
        pre_defined_metric="classification_metric"
    )

    result = {
        # Boolean top-level metric
        "accuracy": classification_metrics["values"]["Accuracy"],
        "precision": classification_metrics["values"]["precision"],
        "recall": classification_metrics["values"]["recall"],
        "auc": classification_metrics["values"]["auc"],
        "f1_score": classification_metrics["values"]["f1_score"],
        
        # Vanilla ModelEvaluator output
        "performance": [
            classification_metrics
        ]
    }
    yield result