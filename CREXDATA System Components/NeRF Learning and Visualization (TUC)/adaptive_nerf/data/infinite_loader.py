import itertools
from torch.utils.data import DataLoader


class InfiniteDataLoader:
    """
    Wrap a DataLoader to yield batches indefinitely, restarting on each epoch end.

    Automatically handles shuffling, worker reuse, and StopIteration.
    """

    def __init__(self, dataloader: DataLoader):
        if not isinstance(dataloader, DataLoader):
            raise TypeError("Expected a torch.utils.data.DataLoader instance.")
        self.dataloader = dataloader
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            # Reset the iterator when the underlying loader is exhausted.
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
        return batch

    def next(self):
        """Return next batch (alias for `next(loader)`)."""
        return self.__next__()

    def reset(self):
        """Manually reset the underlying iterator (e.g., after changing dataset)."""
        self._iterator = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
