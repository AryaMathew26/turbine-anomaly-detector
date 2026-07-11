import os
import tomllib
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, get_run_logger, task

project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)


@task(name="run_monitoring_pipeline", log_prints=True)
def run_monitoring_pipeline(env: str = "local", pipeline_name: str = "monitoring"):
    """Prefect task to run the Kedro monitoring pipeline."""
    logger = get_run_logger()
    logger.info("Running monitoring pipeline")
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)

    logger.info("Monitoring pipeline completed")


@flow(name="monitoring_flow", log_prints=True)
def monitoring_flow():
    logger = get_run_logger()
    logger.info("Starting monitoring flow")
    run_monitoring_pipeline()


if __name__ == "__main__":
    monitoring_flow()
