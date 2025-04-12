from src.data.abstract_tabular_dataset import AbstractTabularDataset, PolarsTableItem


class PolarsTabularDataset(AbstractTabularDataset):
    def prepare_data(self, data: PolarsTableItem) -> PolarsTableItem:
        return data
