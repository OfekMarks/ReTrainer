import dagshub
from dagshub.data_engine import datasources
from settings import settings


def load_dagshub_dataset(repo: str, datasource_name: str):
    """
    Load a dataset using the DagsHub Data Engine API.

    Args:
        repo: The repository name in the format "owner/repo" (e.g., "ofekmarks/my-first-repo")
        datasource_name: The name of the datasource to load.

    Returns:
        The fetched datasource object which can be converted to pandas/arrow/dataloader.
    """
    dagshub.auth.add_app_token(settings.dagshub_user_token)
    ds = datasources.get_datasource(repo, datasource_name)

    return ds
