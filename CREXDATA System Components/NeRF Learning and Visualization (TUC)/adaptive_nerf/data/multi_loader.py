from typing import Dict, List
from torch.utils.data import DataLoader


class MultiLoader:
    """
    Iterate over multiple DataLoaders in lockstep, yielding {cell_id:=region_id: batch} forever.

    Each loader is cycled on exhaustion; empty loaders are filtered out on init.
    """

    def __init__(self, loaders: List[DataLoader]):
        # keep only non-empty loaders
        self.loaders = [dl for dl in loaders if len(dl) > 0]
        if not self.loaders:
            raise ValueError("MultiLoader received no non-empty DataLoaders.")

        def get_cid(dl: DataLoader) -> int:
            ds = getattr(dl, "dataset", None)
            cid = getattr(ds, "cell_id", None)
            if cid is None:
                raise ValueError(
                    "Each dataset used with MultiLoader must expose .cell_id."
                )
            return int(cid)

        self.cids = [get_cid(dl) for dl in self.loaders]

    def __iter__(self):
        iters = [iter(dl) for dl in self.loaders]
        while True:
            group: Dict[int, dict] = {}
            for i, (dl, it) in enumerate(zip(self.loaders, iters)):
                try:
                    batch = next(it)
                except StopIteration:
                    iters[i] = iter(dl)
                    batch = next(iters[i])
                group[self.cids[i]] = batch
            yield group
