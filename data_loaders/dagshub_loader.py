import dagshub
from dagshub.data_engine import datasources
from typing import Any, Optional
from data_loaders.loader_interface import DataLoaderInterface
from settings import settings


class DagsHubDataEngineLoader(DataLoaderInterface):
    """
    A concrete loader for DagsHub Data Engine.
    It can return data in various formats (DataFrames, PyTorch Dataloaders, TensorFlow Datasets)
    based on configuration, natively handling CV, sequential, or tabular data.
    """

    def __init__(
        self,
        repo: str,
        datasource_name: str,
        format_type: str = "dataframe",
        **flavor_kwargs,
    ):
        """
        Args:
            repo: DagsHub repo (e.g., 'ofekmarks/my-first-repo')
            datasource_name: Name of the Data Engine datasource.
            format_type: 'dataframe', 'pytorch', or 'tensorflow'.
            flavor_kwargs: Extra kwargs for PyTorch/TF (e.g., batch_size, transform wrappers).
        """
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
                    f"No Data Engine records found for the requested split."
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

    def get_train_data(self) -> Any:
        """Query and return the 'train' split."""
        return self._convert_query_to_format(self.ds[self.ds["split"] == "train"])

    def get_test_data(self) -> Any:
        """Query and return the 'test' split."""
        return self._convert_query_to_format(self.ds[self.ds["split"] == "test"])
