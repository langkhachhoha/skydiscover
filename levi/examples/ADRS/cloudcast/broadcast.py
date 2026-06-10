import os
from typing import Dict, List

# Verbose diagnostic output is opt-in (set CLOUDCAST_VERBOSE=1).
VERBOSE = os.environ.get("CLOUDCAST_VERBOSE", "").lower() in ("1", "true", "yes")


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            # Partition ids are stored as strings ("0", "1", ...) to match how
            # they are looked up everywhere else (append_dst_partition_path /
            # set_dst_partition_paths str()-ify the partition, and the
            # simulator indexes self.paths[dst][str(partition_id)]). Using
            # range(num_partitions) here produced int keys and a KeyError.
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        vprint(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)
