import os
import sys
from BitVector import BitVector
from numpy import pi

if not (os.path.abspath("../cordic_common") in sys.path):
    sys.path.append(os.path.abspath("../cordic_common"))

from model import cordic_model
import cordic_common.methods as methods
import cordic_common.cordic_types as cordic_types
from cordic_common.adder_subtractor import adder_subtractor
from cordic_common.cordic_constants import (
    atan_lut,
    atanh_lut,
    hyperbolic_repeat_indices,
)


class model_2(cordic_model):
    def __init__(self, mantissa_bits: int, frac_bits: int, iterations: int):
        """model_2 improves from model_1 by:
        - using adder-subtractor instead of converting to two's complement
        - extending sine/cosine range from -+ pi/2 to -+ pi
        - multiplying log result by two (bit shift)

        Args:
            mantissa_bits (int): How many mantissa bits are used
            frac_bits (int): How many fractional bits are used
            iterations (int): How many CORDIC iterations are run
        """
        super().__init__(mantissa_bits, frac_bits, iterations)

        # Private members
        self._x_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self._y_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self._z_in: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self._x_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self._y_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self._z_out: BitVector = BitVector(intVal=0, size=(self.mb + self.fb))
        self._type = None
        self._mode = None
        self._signbit = 0
        self._invert_res = False
        self._mul_by_2 = False
        self._adder = adder_subtractor(bits=self.mb + self.fb)

    def preprocess(self):
        mant_one_vec = methods.to_fixed_point(1.0, self.mb, self.fb)
        zero_vec = BitVector(intVal=0, size=(self.mb + self.fb))
        K_vec = methods.to_fixed_point(
            1 / methods.calc_k(self.iters, cordic_types.rotation_type.CIRCULAR),
            self.mb,
            self.fb,
        )
        Kh_vec = methods.to_fixed_point(
            1 / methods.calc_k(self.iters, cordic_types.rotation_type.HYPERBOLIC),
            self.mb,
            self.fb,
        )
        pi_vec = methods.to_fixed_point(
            pi,
            self.mb,
            self.fb,
        )

        sincos_addsub = 0
        self._invert_res = False
        self._mul_by_2 = False

        if (
            self.op == "Sine"
            or self.op == "Cosine"
        ):
            # TODO: what is the efficient way to do this in hardware??
            if methods.to_double_single(self.d_in, self.mb, self.fb) > pi/2:
                addee = pi_vec
                sincos_addsub = 1
                self._invert_res = True
            elif methods.to_double_single(self.d_in, self.mb, self.fb) < -pi/2:
                addee = pi_vec
                self._invert_res = True
            else:
                addee = zero_vec
        elif self.op == "Log":
            addee = mant_one_vec
            self._mul_by_2 = True
        else:
            addee = zero_vec

        d_in_add = self._adder.add(self.d_in, addee)
        d_in_sub = self._adder.sub(self.d_in, addee)

        if (
            self.op == "Sine"
            or self.op == "Cosine"
        ):
            self._mode = cordic_types.cordic_mode.ROTATION
            self._type = cordic_types.rotation_type.CIRCULAR
            self._x_in = K_vec
            self._y_in = zero_vec
            self._z_in = d_in_add if sincos_addsub == 0 else d_in_sub
        elif (
            self.op == "Arctan"
            or self.op == "Arctanh"
        ):
            self._mode = cordic_types.cordic_mode.VECTORING
            self._type = (
                cordic_types.rotation_type.CIRCULAR
                if self.op == "Arctan"
                else cordic_types.rotation_type.HYPERBOLIC
            )
            self._x_in = mant_one_vec
            self._y_in = d_in_add
            self._z_in = zero_vec
        elif (
            self.op == "Sinh"
            or self.op == "Cosh"
        ):
            self._mode = cordic_types.cordic_mode.ROTATION
            self._type = cordic_types.rotation_type.HYPERBOLIC
            self._x_in = Kh_vec
            self._y_in = zero_vec
            self._z_in = d_in_add
        elif self.op == "Exponential":
            self._mode = cordic_types.cordic_mode.ROTATION
            self._type = cordic_types.rotation_type.HYPERBOLIC
            self._x_in = Kh_vec
            self._y_in = Kh_vec
            self._z_in = d_in_add
        elif self.op == "Log":
            self._mode = cordic_types.cordic_mode.VECTORING
            self._type = cordic_types.rotation_type.HYPERBOLIC
            self._x_in = d_in_add
            self._y_in = d_in_sub
            self._z_in = zero_vec

    def postprocess(self):
        if self.op == "Sine":
            d_out = self._y_out
        elif self.op == "Cosine":
            d_out = self._x_out
        elif self.op == "Arctan":
            d_out = self._z_out
        elif self.op == "Sinh":
            d_out = self._y_out
        elif self.op == "Cosh":
            d_out = self._x_out
        elif self.op == "Arctanh":
            d_out = self._z_out
        elif self.op == "Exponential":
            d_out = self._x_out
        elif self.op == "Log":
            d_out = self._z_out
        d_out_c = self._adder.add(~d_out, BitVector(intVal=1, size=self.mb + self.fb))
        d_out_shifted = d_out.deep_copy() << 1
        if self._invert_res:
            self.d_out = d_out_c
        elif self._mul_by_2:
            self.d_out = d_out_shifted
        else:
            self.d_out = d_out

    def run(self):
        self.preprocess()

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

        mant_one_vec = BitVector(intVal=1, size=(self.mb + self.fb)) << self.fb

        # Concatenate mantissa and fraction
        x_vec[0] = self._x_in
        y_vec[0] = self._y_in
        z_vec[0] = self._z_in

        # Convert floats into fixed-point representation
        atan_lut_fp = []
        atanh_lut_fp = []
        for i in range(0, len(atan_lut)):
            fx_pnt = methods.to_fixed_point(atan_lut[i], self.mb, self.fb)
            atan_lut_fp.append(fx_pnt)
            if i != 0:
                fx_pnt2 = methods.to_fixed_point(atanh_lut[i], self.mb, self.fb)
                atanh_lut_fp.append(fx_pnt2)
            else:
                atanh_lut_fp.append(BitVector(intVal=0, size=(self.mb + self.fb)))

        if self._type == cordic_types.rotation_type.CIRCULAR:
            lut_index_vec[0] = 0
        elif self._type == cordic_types.rotation_type.HYPERBOLIC:
            lut_index_vec[0] = 1
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

            # Values are simply moved forward, if:
            # - it is the 0th iteration of hyperbolic mode
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
            if x_vec[i][self._signbit] == 1:
                x_shift_vec[i] |= sign_ext_bit_vec
            if y_vec[i][self._signbit] == 1:
                y_shift_vec[i] |= sign_ext_bit_vec

            # sigma = True means sigma = 1
            # sigma = False means sigma = -1
            if self._mode == cordic_types.cordic_mode.ROTATION:
                sigma = z_vec[i][self._signbit] == 0
            elif self._mode == cordic_types.cordic_mode.VECTORING:
                sigma = y_vec[i][self._signbit] == 1

            # m = True means m = 1
            # m = False means m = -1
            if self._type == cordic_types.rotation_type.CIRCULAR:
                m = True
            elif self._type == cordic_types.rotation_type.HYPERBOLIC:
                m = False

            if sigma ^ m:
                x_vec[i + 1] = self._adder.add(x_vec[i], y_shift_vec[i])
            else:
                x_vec[i + 1] = self._adder.sub(x_vec[i], y_shift_vec[i])

            if sigma:
                y_vec[i + 1] = self._adder.add(y_vec[i], x_shift_vec[i])
                z_vec[i + 1] = self._adder.sub(z_vec[i], atan_val_vec[i])
            else:
                y_vec[i + 1] = self._adder.sub(y_vec[i], x_shift_vec[i])
                z_vec[i + 1] = self._adder.add(z_vec[i], atan_val_vec[i])

            if not repeat:
                lut_index_vec[i + 1] = lut_index_vec[i] + 1
            else:
                lut_index_vec[i + 1] = lut_index_vec[i]

        self._x_out = x_vec[-1]
        self._y_out = y_vec[-1]
        self._z_out = z_vec[-1]
        self.postprocess()
