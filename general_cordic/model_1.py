from BitVector import BitVector

# from general_cordic import rotation_type, cordic_mode


class model_1:
    def __init__(
        self,
        mantissa_bits: int,
        frac_bits: int,
        iterations: int,
    ):
        self.mb = mantissa_bits
        self.fb = frac_bits
        self.iters = iterations
        self.x_in: (BitVector, BitVector) = (
            BitVector(intVal=0, size=self.mb),
            BitVector(intVal=0, size=self.fb),
        )
        self.y_in: (BitVector, BitVector) = (
            BitVector(intVal=0, size=self.mb),
            BitVector(intVal=0, size=self.fb),
        )
        self.z_in: (BitVector, BitVector) = (
            BitVector(intVal=0, size=self.mb),
            BitVector(intVal=0, size=self.fb),
        )
        self.type = None
        self.mode = None

    def set_inputs(
        self,
        x_in: (BitVector, BitVector),
        y_in: (BitVector, BitVector),
        z_in: (BitVector, BitVector),
    ):
        self.x_in = x_in
        self.y_in = y_in
        self.z_in = z_in

    def run(self):
        self.x_out = self.x_in
        self.y_out = self.y_in
        self.z_out = self.z_in
