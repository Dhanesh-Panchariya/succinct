from Succinct.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from Succinct.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nX============X")
except Exception as e:
    logger.exception(e)
    raise e

