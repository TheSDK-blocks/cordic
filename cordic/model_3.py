import os
import sys
import numpy as np

if not (os.path.abspath("../cordic_common") in sys.path):
    sys.path.append(os.path.abspath("../cordic_common"))

from cordic.model import cordic_model
import cordic.cordic_common.methods as methods
import cordic.cordic_common.cordic_types as cordic_types
from cordic.cordic_common.adder_subtractor import adder_subtractor
from cordic.cordic_common.cordic_constants import (
    atan_lut,
    atanh_lut,
    hyperbolic_repeat_indices,
)


class model_3():
    def __init__(self,
                 mantissa_bits: int,
                 frac_bits: int,
                 iterations: int,
                 repr: str,
                 preprocessor_class: str,
                 postprocessor_class: str,
                 use_phase_accum: bool = False,
                 phase_accum_width: int = 32
                 ):
        """model_3 is meant to be fast. 
        model_2 with BitVector was agonizingly slow.
        Thus, we now use fixed 32 bits numpy values.

        Parameters
        ----------
            mantissa_bits (int): How many mantissa bits are used
            frac_bits (int): How many fractional bits are used
            iterations (int): How many CORDIC iterations are run
        """
        self.d_in = np.int32(0)
        self.rs1_in = np.int32(0)
        self.rs2_in = np.int32(0)
        self.rs3_in = np.int32(0)
        self.d_out = np.int32(0)
        self.rs1_out = np.int32(0)
        self.rs2_out = np.int32(0)
        self.rs3_out = np.int32(0)
        self._x_in  = np.int32(0)
        self._y_in  = np.int32(0)
        self._z_in  = np.int32(0)
        self._x_out = np.int32(0)
        self._y_out = np.int32(0)
        self._z_out = np.int32(0)
        self.mb = mantissa_bits
        self.fb = frac_bits
        self.iters = iterations
        self.repr = repr
        self.op = 0
        self._type = None
        self._mode = None
        self.preproc_class = preprocessor_class
        self.postproc_class = postprocessor_class
        self.use_phase_accum = use_phase_accum
        self.phase_accum_width = phase_accum_width

    def trigfunc_preprocess(self):
        K_vec = methods.to_fixed_point(
            1 / methods.calc_k(self.iters, cordic_types.rotation_type.CIRCULAR),
            self.mb,
            self.fb,
            self.repr,
            ret_type="numpy"
        )
        Kh_vec = methods.to_fixed_point(
            1 / methods.calc_k(self.iters, cordic_types.rotation_type.HYPERBOLIC),
            self.mb,
            self.fb,
            self.repr,
            ret_type="numpy"
        )

        if self.op == 0 or self.op == 1:
            # SINE or COSINE
            # TODO: pre-rotation
            self._mode = cordic_types.cordic_mode.ROTATION
            self._type = cordic_types.rotation_type.CIRCULAR
            self._x_in = K_vec
            self._y_in = 0
            self._z_in = self.d_in
        elif self.op == 2:
            # ARCTAN
            self._mode = cordic_types.cordic_mode.VECTORING
            self._type = cordic_types.rotation_type.CIRCULAR
            self._x_in = methods.to_fixed_point(1.0, self.mb, self.fb,
                                                self.repr, ret_type="numpy")
            self._y_in = self.d_in
            self._z_in = 0
        elif self.op == 3 or self.op == 4:
            # SINH or COSH
            self._mode = cordic_types.cordic_mode.ROTATION
            self._type = cordic_types.rotation_type.HYPERBOLIC
            self._x_in = Kh_vec
            self._y_in = 0
            self._z_in = self.d_in
        elif self.op == 5:
            # ARCTANH
            self._mode = cordic_types.cordic_mode.VECTORING
            self._type = cordic_types.rotation_type.HYPERBOLIC
            self._x_in = methods.to_fixed_point(1.0, self.mb, self.fb,
                                                self.repr, ret_type="numpy")
            self._y_in = self.d_in
            self._z_in = 0
        elif self.op == 6:
            # EXPONENTIAL
            self._mode = cordic_types.cordic_mode.ROTATION
            self._type = cordic_types.rotation_type.HYPERBOLIC
            self._x_in = Kh_vec
            self._y_in = Kh_vec
            self._z_in = self.d_in
        elif self.op == 7:
            # LOG
            self._mode = cordic_types.cordic_mode.VECTORING
            self._type = cordic_types.rotation_type.HYPERBOLIC
            one = methods.to_fixed_point(1.0, self.mb, self.fb,
                                         self.repr, ret_type="numpy")
            self._x_in = self.d_in + one
            self._y_in = self.d_in - one
            self._z_in = 0
        else:
            raise ValueError(f"Unidentified operation in preprocessor: {self.op}")

    def basic_preprocess(self):
        self._x_in = self.rs1_in
        self._y_in = self.rs2_in
        self._z_in = self.rs3_in
        if self.op & 1:
            self._type = cordic_types.rotation_type.CIRCULAR
        else:
            self._type = cordic_types.rotation_type.HYPERBOLIC
        if self.op & (1 << 1):
            self._mode = cordic_types.cordic_mode.ROTATION
        else:
            self._mode = cordic_types.cordic_mode.VECTORING

    def upconvert_preprocess(self):
        x = self.rs1_in
        y = self.rs2_in
        z = self.rs3_in
        self._mode = cordic_types.cordic_mode.ROTATION
        self._type = cordic_types.rotation_type.CIRCULAR
        pi_over_2 = np.int32(methods.to_fixed_point(np.pi/2, self.mb, self.fb, self.repr, ret_type="numpy"))
        # Apply initial rotation of 90 degrees if needed
        # This is because CORDIC cannot rotate more than around 100 degrees
        if self.rs3_in > pi_over_2:
            z = self.rs3_in - pi_over_2
            x = -self.rs2_in
            y = self.rs1_in
        elif z < -pi_over_2:
            z = z + pi_over_2
            x = self.rs2_in
            y = -self.rs1_in
        self._x_in = np.int32(x)
        self._y_in = np.int32(y)
        self._z_in = np.int32(z)

    
    def trigfunc_postprocess(self):
        self.rs1_out = self._x_out
        self.rs2_out = self._y_out
        self.rs3_out = self._z_out
        if self.op == 0:
            # SINE
            d_out = self._y_out
        elif self.op == 1:
            # COSINE
            d_out = self._x_out
        elif self.op == 2:
            # ARCTAN
            d_out = self._z_out
        elif self.op == 3:
            # SINH
            d_out = self._y_out
        elif self.op == 4:
            # COSH
            d_out = self._x_out
        elif self.op == 5:
            # ARCTANH
            d_out = self._z_out
        elif self.op == 6:
            # EXPONENTIAL
            d_out = self._x_out
        elif self.op == 7:
            # LOG
            d_out = np.int32(self._z_out << 1)
        else:
            raise ValueError(f"Unidentified operation in postprocessor: {self.op}")

        self.d_out = d_out

    def basic_postprocess(self):
        self.rs1_out = self._x_out
        self.rs2_out = self._y_out
        self.rs3_out = self._z_out
        self.d_out   = np.int32(0)


    def truncate(self):
        """Truncate output to better mimick a CORDIC with given resolution"""
        self.rs1_out = np.int32(self.rs1_out & (((1 << (self.mb + self.fb)) - 1)) << (32 - (self.mb + self.fb)))
        self.rs2_out = np.int32(self.rs2_out & (((1 << (self.mb + self.fb)) - 1)) << (32 - (self.mb + self.fb)))
        self.rs3_out = np.int32(self.rs3_out & (((1 << (self.mb + self.fb)) - 1)) << (32 - (self.mb + self.fb)))
        self.d_out   = np.int32(self.d_out   & (((1 << (self.mb + self.fb)) - 1)) << (32 - (self.mb + self.fb)))


    def run(self):
        if self.preproc_class == "TrigFunc":
            self.trigfunc_preprocess()
        elif self.preproc_class == "Basic":
            self.basic_preprocess()
        elif self.preproc_class == "UpConvert":
            self.upconvert_preprocess()
        else:
            raise ValueError(f"Unidentified preprocessor class: {self.preproc_class}")

        # Calculate number of repeats
        # Can usually be 2 or 3 depending whether there are less or
        # more than 40 iterations
        n_repeats = 0
        if self._type == cordic_types.rotation_type.HYPERBOLIC:
            for i in range(0, self.iters):
                if i in hyperbolic_repeat_indices:
                    n_repeats += 1

        atan_lut_fp = []
        atanh_lut_fp = []
        for i in range(0, len(atan_lut)):
            fx_pnt = methods.to_fixed_point(atan_lut[i], self.mb, self.fb, self.repr, ret_type="numpy")
            atan_lut_fp.append(np.int32(fx_pnt))
            if i != 0:
                fx_pnt2 = methods.to_fixed_point(atanh_lut[i], self.mb, self.fb, self.repr, ret_type="numpy")
                atanh_lut_fp.append(np.int32(fx_pnt2))
            else:
                atanh_lut_fp.append(np.int32(0))

        if self._type == cordic_types.rotation_type.CIRCULAR:
            lut_idx = 0
        elif self._type == cordic_types.rotation_type.HYPERBOLIC:
            lut_idx = 1
        repeats = 0
        repeat = False

        x_vec = np.int32(self._x_in)
        y_vec = np.int32(self._y_in)
        z_vec = np.int32(self._z_in)
        x_shift_vec = np.int32(0)
        y_shift_vec = np.int32(0)
        atan_val_vec = np.int32(0)

        for i in range(0, self.iters + n_repeats):
            bypass = False

            if (i - repeats) in hyperbolic_repeat_indices and not repeat \
                and self._type == cordic_types.rotation_type.HYPERBOLIC:
                repeat = True
                repeats += 1
            else:
                repeat = False

            if (i == 0) and (self._type == cordic_types.rotation_type.HYPERBOLIC):
                bypass = True

            # Values are simply moved forward, if:
            # - it is the 0th iteration of hyperbolic mode
            if bypass:
                x_vec = x_vec
                y_vec = y_vec
                z_vec = z_vec
                lut_idx = lut_idx
                continue

            if lut_idx < len(atan_lut):
                if self._type == cordic_types.rotation_type.CIRCULAR:
                    atan_val_vec = atan_lut_fp[lut_idx]
                elif self._type == cordic_types.rotation_type.HYPERBOLIC:
                    atan_val_vec = atanh_lut_fp[lut_idx]
            else:
                atan_val_vec = 1 << lut_idx

            x_shift_vec = np.int32(x_vec >> lut_idx)
            y_shift_vec = np.int32(y_vec >> lut_idx)

            # sigma = True means sigma = 1
            # sigma = False means sigma = -1
            if self._mode == cordic_types.cordic_mode.ROTATION:
                sigma = z_vec > 0
            elif self._mode == cordic_types.cordic_mode.VECTORING:
                sigma = y_vec < 0

            # m = True means m = 1
            # m = False means m = -1
            if self._type == cordic_types.rotation_type.CIRCULAR:
                m = True
            elif self._type == cordic_types.rotation_type.HYPERBOLIC:
                m = False

            if sigma ^ m:
                x_vec = x_vec + y_shift_vec
            else:
                x_vec = x_vec - y_shift_vec

            if sigma:
                y_vec = y_vec + x_shift_vec
                z_vec = z_vec - atan_val_vec
            else:
                y_vec = y_vec - x_shift_vec
                z_vec = z_vec + atan_val_vec

            if not repeat:
                lut_idx = lut_idx + 1
            else:
                lut_idx = lut_idx

        self._x_out = x_vec
        self._y_out = y_vec
        self._z_out = z_vec

        if self.postproc_class == "TrigFunc":
            self.trigfunc_postprocess()
        elif self.postproc_class == "Basic":
            self.basic_postprocess()
        elif self.postproc_class == "UpConvert":
            self.basic_postprocess()
        else:
            raise ValueError(f"Unidentified postprocessor class: {self.postproc_class}")
        
        self.truncate()
