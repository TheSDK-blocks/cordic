from BitVector import BitVector
from math import pow

# from general_cordic import rotation_type, cordic_mode

atan_lut = [
    0.7853981634,
    0.463647609,
    0.2449786631,
    0.1243549945,
    0.06241881,
    0.03123983343,
    0.01562372862,
    0.00781234106,
    0.003906230132,
]

two_pow_lut = [
    pow(2, 0),
    pow(2, -1),
    pow(2, -2),
    pow(2, -3),
    pow(2, -4),
    pow(2, -5),
    pow(2, -6),
    pow(2, -7),
    pow(2, -8),
]


class ripple_carry_adder:
    def __init__(self, bits: int, endianness: str = 'big'):
        self.bits = bits
        assert endianness in ['big', 'little'], \
            "Endianness must be either 'big' or 'little'"
        self.endianness = endianness

    def add(self, input_a: BitVector, input_b: BitVector):
        if self.endianness == 'big':
            dir = -1
            offs = 1
        else:
            dir = 1
            offs = 0

        S_vec = self.bits * [0]
        C_vec = (self.bits + 1) * [0]
        for i in range(0, self.bits):
            S_vec[i] = input_a[dir*i - offs] ^ input_b[dir*i - offs] ^ C_vec[i]
            C_vec[i+1] = (input_a[dir*i - offs] & input_b[dir*i - offs]) + \
                (C_vec[i] & (input_a[dir*i - offs] ^ input_b[dir*i - offs]))
        if self.endianness == 'big':
            S_vec.reverse()
        return BitVector(bitlist=S_vec)


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

        self.signbit = 0

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
        x_vec = (self.iters + 1) * [BitVector]
        y_vec = (self.iters + 1) * [BitVector]
        z_vec = (self.iters + 1) * [BitVector]

        x_vec[0] = self.x_in[0] + self.x_in[1]
        y_vec[0] = self.y_in[0] + self.y_in[1]
        z_vec[0] = self.z_in[0] + self.z_in[1]

        for i in range(0, self.iters):
            if z_vec[i][self.signbit] == 0:
                x_vec[i+1] = x_vec[i] - (y_vec[i] >> i)
                y_vec[i+1] = y_vec[i] + (x_vec[i] >> i)
                z_vec[i+1] = z_vec[i] - atan_lut[i]
            else:
                x_vec[i+1] = x_vec[i] + (y_vec[i] >> i)
                y_vec[i+1] = y_vec[i] - (x_vec[i] >> i)
                z_vec[i+1] = z_vec[i] + atan_lut[i]

        import pdb; pdb.set_trace()

        self.x_out = self.x_in
        self.y_out = self.y_in
        self.z_out = self.z_in


if __name__ == "__main__":
    rca = ripple_carry_adder(16)
    a = BitVector(intVal=0, size=16)
    b = BitVector(intVal=2, size=16)
    ans = rca.add(a, b)
    print(f"a: {a.int_val()}, b: {b.int_val()}, ans: {ans.int_val()}")
