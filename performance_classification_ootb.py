import pandas as pd

import modelop.monitors.performance as performance
import modelop.schema.infer as infer


# modelop.init
def init(job_json):
    
    # Extract input schema from job JSON
    input_schema_definition = infer.extract_input_schema(job_json)
    
    # Get monitoring parameters from schema
    global MONITORING_PARAMETERS
    MONITORING_PARAMETERS = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )
    
    print("label_column: ", MONITORING_PARAMETERS["label_column"], flush=True)
    print("score_column: ", MONITORING_PARAMETERS["score_column"], flush=True)


# modelop.metrics
def metrics(dataframe):
    
    print(dataframe.columns, flush=True)

    # Initialize ModelEvaluator
    model_evaluator = performance.ModelEvaluator(
        dataframe=dataframe,
        score_column=MONITORING_PARAMETERS["score_column"],
        label_column=MONITORING_PARAMETERS["label_column"],
    )

    # Compute classification metrics
    classification_metrics = model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )

    result = {
        # Top-level metrics
        "accuracy": classification_metrics["values"]["accuracy"],
        "precision": classification_metrics["values"]["precision"],
        "recall": classification_metrics["values"]["recall"],
        "auc": classification_metrics["values"]["auc"],
        "f1_score": classification_metrics["values"]["f1"],
        
        # Vanilla ModelEvaluator output
        "performance": [
            classification_metrics
        ]
    }
    yield result
