import pandas as pd

import modelop.monitors.volumetrics as volumetrics
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
def metrics(df_1, df_2):

    # Get identifier_columns from MONITORING_PARAMETERS
    identifier_columns = MONITORING_PARAMETERS["identifier_columns"]

    # Initialize Volumetrics monitor with 1st input DataFrame
    volumetrics_monitor = volumetrics.VolumetricsMonitor(df_1)

    # Compare DataFrames on identifier_columns
    identifiers_comparison = volumetrics_monitor.identifiers_match_on_column(
        df_2, identifier_columns
    )

    result = {
        # Boolean top-level metric
        "identifiers_match": identifiers_comparison["identifiers_match"],
        
        # Vanilla VolumetricsMonitor output
        "volumetrics": [
            identifiers_comparison
        ]
    }
    yield result
