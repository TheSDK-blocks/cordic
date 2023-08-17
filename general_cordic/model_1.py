from BitVector import BitVector
from math import pow, modf, floor

# from general_cordic import rotation_type, cordic_mode

lut_size = 9
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


class methods:
    def to_fixed_point(value: float, mantissa_bits: int, float_bits: int):
        (frac, integer) = modf(value)
        frac_bits = floor((1 << float_bits) * frac)
        return (
            BitVector(intVal=int(integer), size=mantissa_bits),
            BitVector(intVal=frac_bits, size=float_bits),
        )

    def to_double(bit_vector: (BitVector, BitVector), float_bits: int):
        integer = bit_vector[0].int_val()
        frac = bit_vector[1].int_val() / (1 << float_bits)
        return integer + frac

    def to_double_single(bit_vector: BitVector, mantissa_bits: int, float_bits: int):
        if bit_vector[0]:
            bt_vec = BitVector(intVal=((~bit_vector).int_val() + 1), size=(mantissa_bits + float_bits))
            sign = -1
        else:
            bt_vec = bit_vector
            sign = 1
        integer = bt_vec[0:mantissa_bits].int_val()
        frac = bt_vec[mantissa_bits:(mantissa_bits + float_bits)].int_val() / (1 << float_bits)
        return sign * (integer + frac)


class ripple_carry_adder:
    def __init__(self, bits: int, endianness: str = "big"):
        self.bits = bits
        assert endianness in [
            "big",
            "little",
        ], "Endianness must be either 'big' or 'little'"
        self.endianness = endianness

    def add(self, input_a: BitVector, input_b: BitVector):
        if self.endianness == "big":
            dir = -1
            offs = 1
        else:
            dir = 1
            offs = 0

        S_vec = self.bits * [0]
        C_vec = (self.bits + 1) * [0]
        for i in range(0, self.bits):
            S_vec[i] = input_a[dir * i - offs] ^ input_b[dir * i - offs] ^ C_vec[i]
            C_vec[i + 1] = (input_a[dir * i - offs] & input_b[dir * i - offs]) + (
                C_vec[i] & (input_a[dir * i - offs] ^ input_b[dir * i - offs])
            )
        if self.endianness == "big":
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
        self.adder = ripple_carry_adder(bits=self.mb + self.fb)

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
        x_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        y_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        z_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        x_c_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        y_c_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        z_c_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        x_shift_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        y_shift_vec = (self.iters + 1) * [BitVector(intVal=0, size=(self.mb + self.fb))]
        one_vec = BitVector(intVal=1, size=(self.mb + self.fb))
        mant_one_vec = BitVector(intVal=1, size=(self.mb + self.fb)) << self.fb

        x_vec[0] = self.x_in[0] + self.x_in[1]
        y_vec[0] = self.y_in[0] + self.y_in[1]
        z_vec[0] = self.z_in[0] + self.z_in[1]

        x_c_vec[0] = self.adder.add(x_vec[0], one_vec)
        y_c_vec[0] = self.adder.add(y_vec[0], one_vec)
        z_c_vec[0] = self.adder.add(z_vec[0], one_vec)

        atan_lut_fp = []
        for i in range(0, lut_size):
            (man, flo) = methods.to_fixed_point(atan_lut[i], self.mb, self.fb)
            atan_lut_fp.append(man + flo)

        for i in range(0, self.iters):
            # After enough iterations, atan is approximated as 2^(-1)
            if i < lut_size:
                atan_val = atan_lut_fp[i]
            else:
                atan_val = mant_one_vec.deep_copy().shift_right(i)

            x_shift_vec[i] = x_vec[i].deep_copy().shift_right(i)
            y_shift_vec[i] = y_vec[i].deep_copy().shift_right(i)

            x_c_vec[i] = self.adder.add(~x_shift_vec[i], one_vec)
            y_c_vec[i] = self.adder.add(~y_shift_vec[i], one_vec)
            z_c_vec[i] = self.adder.add(~atan_val, one_vec)

            if z_vec[i][self.signbit] == 0:
                x_vec[i + 1] = self.adder.add(x_vec[i], y_c_vec[i])
                y_vec[i + 1] = self.adder.add(y_vec[i], x_shift_vec[i])
                z_vec[i + 1] = self.adder.add(z_vec[i], z_c_vec[i])
            else:
                x_vec[i + 1] = self.adder.add(x_vec[i], y_shift_vec[i])
                y_vec[i + 1] = self.adder.add(y_vec[i], x_c_vec[i])
                z_vec[i + 1] = self.adder.add(z_vec[i], atan_val)
        import pdb

        pdb.set_trace()

        self.x_out = self.x_in
        self.y_out = self.y_in
        self.z_out = self.z_in


if __name__ == "__main__":
    rca = ripple_carry_adder(16)
    a = BitVector(intVal=0, size=16)
    b = BitVector(intVal=2, size=16)
    ans = rca.add(a, b)
    print(f"a: {a.int_val()}, b: {b.int_val()}, ans: {ans.int_val()}")
