import dagshub
from dagshub.data_engine import datasources
from typing import Any
from pydantic import BaseModel, Field
from data_loaders.loader_interface import DataLoaderInterface
from settings import settings


class DagsHubDataEngineLoader(DataLoaderInterface):
    """
    A concrete loader for DagsHub Data Engine.
    Returns the full dataset; splitting is handled by splitter modules.
    """

    class ConfigModel(BaseModel):
        repo: str = Field(
            default="ofekmarks/my-first-repo",
            description="DagsHub repository owner/repo",
        )
        datasource_name: str = Field(
            default="datasource", description="Name of the Data Engine datasource"
        )
        format_type: str = Field(
            default="dataframe",
            description="Return format (dataframe, pytorch, tensorflow)",
        )

    def __init__(
        self,
        repo: str,
        datasource_name: str,
        format_type: str = "dataframe",
        **flavor_kwargs,
    ):
        self.repo = repo
        self.datasource_name = datasource_name
        self.format_type = format_type
        self.flavor_kwargs = flavor_kwargs

        dagshub.auth.add_app_token(settings.dagshub_user_token)

        self.ds = datasources.get_datasource(self.repo, self.datasource_name)

    def _convert_query_to_format(self, query) -> Any:
        """Converts the DagsHub Data Engine query results into the requested format type."""
        if self.format_type == "dataframe":
            df = query.all().dataframe
            if df.empty:
                raise ValueError(
                    "No Data Engine records found for the requested query."
                )
            return df
        elif self.format_type == "pytorch":
            return query.all().as_ml_dataloader(flavor="torch", **self.flavor_kwargs)
        elif self.format_type == "tensorflow":
            return query.all().as_ml_dataloader(
                flavor="tensorboard", **self.flavor_kwargs
            )
        else:
            raise ValueError(
                f"Unsupported format_type: '{self.format_type}'. Expected dataframe, pytorch, or tensorflow."
            )

    def get_data(self) -> Any:
        """Fetch the complete dataset from the DagsHub datasource."""
        return self._convert_query_to_format(self.ds)
