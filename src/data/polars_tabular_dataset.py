from src.data.types import PolarsTableItem
from src.data.abstract_tabular_dataset import AbstractTabularDataset


class PolarsTabularDataset(AbstractTabularDataset):
    def make_data(self, data: PolarsTableItem) -> PolarsTableItem:
        return data
