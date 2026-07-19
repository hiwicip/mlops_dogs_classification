# MLOps Dog Breed Classification Project Documentation

Documentation of core classes and functions used in the MLOps Dog Breed Classification Project.

## Data Preprocessing

::: dogs_classification.data.DogDataset

::: dogs_classification.data.download_data

::: dogs_classification.data.preprocess


## Model Training

::: dogs_classification.model.DogModel

::: dogs_classification.train.train

::: dogs_classification.create_onnx.create_onnx_model


## Model Evaluation

::: dogs_classification.evaluate.evaluate

::: dogs_classification.visualize.log_visualizations


## Model Deployment

::: dogs_classification.bentoml.DogBreedClassificationService

::: dogs_classification.bentoml.save_prediction

::: dogs_classification.frontend.get_backend_url

::: dogs_classification.frontend.main


## Drift Detection

::: dogs_classification.data_drift.download_predictions_from_gcs

::: dogs_classification.data_drift.build_reference_df

::: dogs_classification.data_drift.build_reference_report_df

::: dogs_classification.data_drift.load_or_build_reference_report_df

::: dogs_classification.data_drift.run_analysis

::: dogs_classification.data_drift.upload_report_to_gcs

::: dogs_classification.data_drift.main
