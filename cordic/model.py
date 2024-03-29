import os
import sys
from BitVector import BitVector

if not (os.path.abspath("../cordic_common") in sys.path):
    sys.path.append(os.path.abspath("../cordic_common"))

import cordic.cordic_common.cordic_types as cordic_types


class cordic_model:
    def __init__(self, mantissa_bits: int, frac_bits: int, iterations: int, repr: str):
        """
        Args:
            mantissa_bits (int): How many mantissa bits are used
            frac_bits (int): How many fractional bits are used
            iterations (int): How many CORDIC iterations are run
            repr (str): number representation: fixed-point or pi (values range from -pi to pi)
        """
        # Config parameters
        self.mb = mantissa_bits
        self.fb = frac_bits
        self.iters = iterations
        self.repr = repr

        # IOs
        self.d_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.rs1_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.rs2_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.rs3_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.op: cordic_types.trigonometric_function = (
            cordic_types.trigonometric_function.SIN
        )
        self.d_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.rs1_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.rs2_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self.rs3_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))

    def run(self):
        raise NotImplementedError
