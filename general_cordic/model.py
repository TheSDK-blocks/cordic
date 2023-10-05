import os
import sys
from BitVector import BitVector

if not (os.path.abspath("../cordic_common") in sys.path):
    sys.path.append(os.path.abspath("../cordic_common"))

import cordic_common.cordic_types as cordic_types


class cordic_model:
    def __init__(self):
        pass

    def set_mode(self, mode: cordic_types.cordic_mode):
        raise NotImplementedError

    def set_type(self, cordic_type: cordic_types.rotation_type):
        raise NotImplementedError

    def set_inputs(
        self,
        x_in: BitVector,
        y_in: BitVector,
        z_in: BitVector,
    ):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
