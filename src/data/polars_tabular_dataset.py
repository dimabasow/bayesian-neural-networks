from src.data.abstract_tabular_dataset import AbstractTabularDataset
from src.data.types import PolarsTableItem


class PolarsTabularDataset(AbstractTabularDataset):
    def make_data(self, data: PolarsTableItem) -> PolarsTableItem:
        return data
