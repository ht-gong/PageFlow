
from typing import List
from variable import Variable

class CombinedColumnRep:
    reps: List[Variable]

    def __init__(self, reps=List[Variable], tag: str | None = None, identity=None):
        self.reps = reps
        self.tag = tag
        self.identity = identity  # Optional ColumnEntry for dataflow provenance

    def get_raw(self):
        """Return the RAW (uncompressed) representations."""
        return [r for r in self.reps if not r.is_compressed]

    def get_compressed(self):
        """Return the compressed representations."""
        return [r for r in self.reps if r.is_compressed]
    
    def orig_elem_size(self):
        return self.reps[0].orig_elem_size()
    
    def col_width(self):
        return self.reps[0].col_width()

    def col_length(self):
        return self.reps[0].col_length()
    
    def is_floating_point(self):
        return self.reps[0].tensor_type == "float"

    @property
    def selectivity(self):
        """The selectivity a mask column recorded (None: unknown)."""
        return self.reps[0].selectivity
    
    def get_sliced_rep(self, slice_in: slice):
        return CombinedColumnRep([self.reps[i].get_sliced_rep(slice_in) for i in range(len(self.reps))],
                                 tag=self.tag)
    
    def free_underlying_mem(self, cpu_pool, gpu_pool):
        for rep in self.reps:
            rep.free_underlying_mem(cpu_pool=cpu_pool, gpu_pool=gpu_pool)

    def get_type_torch(self):
        return self.reps[0].get_type_torch()
    
    def tensor_type(self):
        return self.reps[0].tensor_type

