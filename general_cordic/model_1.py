from BitVector import BitVector
import methods
import cordic_types
from cordic_constants import atan_lut, atanh_lut, hyperbolic_repeat_indices

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
    def __init__(self, mantissa_bits: int, frac_bits: int, iterations: int):
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
        self.set_mode(cordic_types.cordic_mode.ROTATION)
        self.set_type(cordic_types.rotation_type.CIRCULAR)

        self.signbit = 0
        self.adder = ripple_carry_adder(bits=self.mb + self.fb)

    def set_mode(self, mode: cordic_types.cordic_mode):
        self._mode = mode

    def set_type(self, cordic_type: cordic_types.rotation_type):
        self._type = cordic_type

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

        # Calculate number of repeats
        # Can usually be 2 or 3 depending whether there are less or
        # more than 40 iterations
        n_repeats = 0
        for i in range(0, self.iters):
            if i in hyperbolic_repeat_indices:
                n_repeats += 1

        # Values from all iterations are saved in a vector
        # This eases debugging and is more hardware-like
        x_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        y_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        z_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        x_c_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        y_c_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        z_c_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        x_shift_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        y_shift_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        atan_val_vec = (self.iters + n_repeats + 1) * [
            BitVector(intVal=0, size=(self.mb + self.fb))
        ]
        lut_index_vec = (self.iters + n_repeats + 1) * [0]

        one_vec = BitVector(intVal=1, size=(self.mb + self.fb))
        mant_one_vec = BitVector(intVal=1, size=(self.mb + self.fb)) << self.fb

        # Concatenate mantissa and fraction
        x_vec[0] = self.x_in[0] + self.x_in[1]
        y_vec[0] = self.y_in[0] + self.y_in[1]
        z_vec[0] = self.z_in[0] + self.z_in[1]

        # Convert floats into fixed-point representation
        atan_lut_fp = []
        atanh_lut_fp = []
        for i in range(0, len(atan_lut)):
            (man, flo) = methods.to_fixed_point(atan_lut[i], self.mb, self.fb)
            atan_lut_fp.append(man + flo)
            if i != 0:
                (man2, flo2) = methods.to_fixed_point(atanh_lut[i], self.mb, self.fb)
                atanh_lut_fp.append(man2 + flo2)
            else:
                atanh_lut_fp.append(BitVector(intVal=0, size=(self.mb + self.fb)))

        lut_index_vec[0] = 0
        repeats = 0
        repeat = False

        for i in range(0, self.iters + n_repeats):

            bypass = False

            if (i - repeats) in hyperbolic_repeat_indices and not repeat:
                repeat = True
                repeats += 1
            else:
                repeat = False

            if (i == 0) and (self._type == cordic_types.rotation_type.HYPERBOLIC):
                bypass = True
            elif repeat and (self._type == cordic_types.rotation_type.CIRCULAR):
                bypass = True

            # Values are simply moved forward, if:
            # - it is the 0th iteration of hyperbolic mode or
            # - it is a repeat index and circular mode (is this necessary?)
            if bypass:
                x_vec[i + 1] = x_vec[i]
                y_vec[i + 1] = y_vec[i]
                z_vec[i + 1] = z_vec[i]
                lut_index_vec[i + 1] = lut_index_vec[i]
                continue

            # After enough iterations, atan and atanh is approximated as 2^(-i)
            if lut_index_vec[i] < len(atan_lut):
                if self._type == cordic_types.rotation_type.CIRCULAR:
                    atan_val_vec[i] = atan_lut_fp[lut_index_vec[i]]
                elif self._type == cordic_types.rotation_type.HYPERBOLIC:
                    atan_val_vec[i] = atanh_lut_fp[lut_index_vec[i]]
            else:
                atan_val_vec[i] = mant_one_vec.deep_copy().shift_right(lut_index_vec[i])

            # if the shifted value is negative,
            # the shifted-in bits must be 1's instead of 0's
            # Here we create a bitmask for OR'ing them into 1's if needed
            sign_ext_vec = (self.mb + self.fb) * [0]
            for j in range(0, self.mb + self.fb):
                if j < i:
                    sign_ext_vec[j] = 1
            sign_ext_bit_vec = BitVector(bitlist=sign_ext_vec)

            # Shifted versions
            x_shift_vec[i] = x_vec[i].deep_copy().shift_right(lut_index_vec[i])
            y_shift_vec[i] = y_vec[i].deep_copy().shift_right(lut_index_vec[i])

            # Shifted bits are 0's by default, so here we OR
            # them to be 1's if the value was negative
            if x_vec[i][self.signbit] == 1:
                x_shift_vec[i] |= sign_ext_bit_vec
            if y_vec[i][self.signbit] == 1:
                y_shift_vec[i] |= sign_ext_bit_vec

            # Two's complement
            x_c_vec[i] = self.adder.add(~x_shift_vec[i], one_vec)
            y_c_vec[i] = self.adder.add(~y_shift_vec[i], one_vec)
            z_c_vec[i] = self.adder.add(~atan_val_vec[i], one_vec)

            # sigma = True means sigma = 1
            # sigma = False means sigma = -1
            if self._mode == cordic_types.cordic_mode.ROTATION:
                sigma = z_vec[i][self.signbit] == 0
            elif self._mode == cordic_types.cordic_mode.VECTORING:
                sigma = y_vec[i][self.signbit] == 1

            if self._type == cordic_types.rotation_type.CIRCULAR:
                m = True
            elif self._type == cordic_types.rotation_type.HYPERBOLIC:
                m = False

            if sigma ^ m:
                x_vec[i + 1] = self.adder.add(x_vec[i], y_shift_vec[i])
            else:
                x_vec[i + 1] = self.adder.add(x_vec[i], y_c_vec[i])

            if sigma:
                y_vec[i + 1] = self.adder.add(y_vec[i], x_shift_vec[i])
                z_vec[i + 1] = self.adder.add(z_vec[i], z_c_vec[i])
            else:
                y_vec[i + 1] = self.adder.add(y_vec[i], x_c_vec[i])
                z_vec[i + 1] = self.adder.add(z_vec[i], atan_val_vec[i])

            if not repeat:
                lut_index_vec[i + 1] = lut_index_vec[i] + 1
            else:
                lut_index_vec[i + 1] = lut_index_vec[i]

        # import pdb; pdb.set_trace()
        self.x_out = (x_vec[-1][0 : self.mb], x_vec[-1][self.mb : (self.mb + self.fb)])
        self.y_out = (y_vec[-1][0 : self.mb], y_vec[-1][self.mb : (self.mb + self.fb)])
        self.z_out = (z_vec[-1][0 : self.mb], z_vec[-1][self.mb : (self.mb + self.fb)])


if __name__ == "__main__":
    rca = ripple_carry_adder(16)
    a = BitVector(intVal=0, size=16)
    b = BitVector(intVal=2, size=16)
    ans = rca.add(a, b)
    print(f"a: {a.int_val()}, b: {b.int_val()}, ans: {ans.int_val()}")
