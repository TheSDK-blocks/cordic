import os
import sys

if not (os.path.abspath("../../thesdk") in sys.path):
    sys.path.append(os.path.abspath("../../thesdk"))

from thesdk import IO, thesdk
from rtl import rtl, rtl_iofile, rtl_connector_bundle
from rtl.module import verilog_module
from spice import spice

import numpy as np
import cocotb
from cocotb.runner import get_runner

# from model_1 import model_1
from model_2 import model_2
from BitVector import BitVector
import cordic_common.methods as methods
import cordic_common.cordic_types as cordic_types
from cordic import Cordic
import matplotlib.pyplot as plt

import plot_format
plot_format.set_style('ieeetran')

class CordicTestbench(thesdk):
    """Testbench for CORDIC module

    Parameters
    ----------
    mantissa_bits : int
        Number of mantissa bits in fixed-point representation 
    fraction_bits : int
        Number of fraction bits in fixed-point representation
    iterations : int
        Number of CORDIC iterations

    """
    def __init__(
        self,
        **kwargs
    ):
        self.print_log(type="I", msg="Initializing %s" % (__name__))
        self.model = "py"  # Can be set externalouly, but is not propagated

        self.mantissa_bits = kwargs["mantissa_bits"]
        self.fraction_bits = kwargs["fraction_bits"]
        self.iterations = kwargs["iterations"]

    def all_to_fp(self, tdata):
        """Convert np array of floats to np array fixed point
        """
        return np.array(
            [
                methods.to_fixed_point(
                    data_point, self.mantissa_bits, self.fraction_bits
                ).int_val()
                for data_point in tdata[:, 0]
            ]
        ).reshape(-1, 1)

class NRTestbench(CordicTestbench):
    """Testbench for running 5g signals

    Parameters
    ----------
    mantissa_bits : int
        Number of mantissa bits in fixed-point representation 
    fraction_bits : int
        Number of fraction bits in fixed-point representation
    iterations : int
        Number of CORDIC iterations
    
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        from URC_toolkit import URC_toolkit
        from URC import URC
        self.URC_tk = URC_toolkit()
        self.sig_gen = None
        self.urc = URC()
        self.urc.model = "py"
        self.urc.mode = 16
        self.dut = Cordic(mantissa_bits=self.mantissa_bits,
                          fraction_bits=self.fraction_bits,
                          iterations=self.iterations)

    # Default path for 5G stimuli yaml
    dsp_test_include_dir = "Entities/ACoreTests/build/tests/programs/dsp-tests/rv32im/sw-build/include"

    def gen_5G_stimuli(self, 
                       file=os.path.join(os.path.abspath(thesdk.HOME), dsp_test_include_dir, "iq-vecs_sigparams.yml"),
                       Rs_bb=0, resolution=16, buffer_len=151):
        """Generate 5G stimuli from YAML to be fed to the URC.
        
        Parameters
        ----------
        file : str
            Path to YAML file
        Rs_bb : int
            Baseband sampling rate
        resolution : int
            Resolution of the signal in hardware
        buffer_len : int
            Buffer length

        Returns
        -------
        Complex numpy array as a column vector in the form I + jQ
        """
        if not os.path.exists(file):
            raise FileNotFoundError(
                "Sig gen YAML file not found: You need to compile dsp-tests in ACoreTests to generate it or provide your own."
                )
        QAM, osr, BWP, BW, in_bits, _ = self.URC_tk.load_sig_gen_yaml(
            file
        )
        self.sig_gen, I_sig, Q_sig, _ = self.URC_tk.init_NR_siggen(
            [], QAM, osr, BWP, BW, in_bits, Rs_bb, resolution, buffer_len
        )
        return np.insert((I_sig + 1j * Q_sig), 0, np.zeros(100)).reshape(-1, 1)

    def gen_cordic_stimuli(self, Fs, Fs_divisor, vec_len):
        """Generate stimuli for CORDIC.
        
        Parameters
        ----------
        Fs : int
            Sampling rate of Z signal
        Fs_divisor : int
            To generate new center frequency, by how much is the sampling rate divided with

        Returns
        -------
        rotation_vector : numpy array
            Rotation vector as column vector
        center_freq : int
            Center frequency calculated from Fs and Fs_divisor
        """
        center_freq = Fs / Fs_divisor
        # Generate rotation vector
        rot_vec = np.linspace(np.pi, -np.pi, Fs_divisor, endpoint=False)
        # Check how many times it manages to rotate during the length of the input signal
        # Truncate output
        repeats = vec_len // len(rot_vec)
        # How many samples remain to be filled of a partial circle
        remainder = vec_len % len(rot_vec)
        # Concatenate repeated rotation and the remainder vector
        rot_vec_extended = np.concatenate(
                (np.tile(rot_vec, repeats), rot_vec[:remainder])
                ).reshape(-1, 1)

        return rot_vec_extended, center_freq

    def rotate(self, I, Q, angle_vec):
        """Ideal rotation using sine and cosine

        Parameters
        ----------
        I : numpy array
            I vector
        Q : numpy array
            Q vector
        anlge_vec: numpy array
            Vector containing rotation angles
        
        Returns
        -------
        new_I : numpy array
            Rotated I as column vector
        new_Q : numpy array
            Rotated Q as column vector
        """
        
        new_I = (I * np.cos(angle_vec) - \
                Q * np.sin(angle_vec)).reshape(-1, 1)
        new_Q = (I * np.sin(angle_vec) + \
                Q * np.cos(angle_vec)).reshape(-1, 1)
        return new_I, new_Q
    
    def plot(self, I, Q, center_freq):
        self.URC_tk.plot_5G_output(["I","Q"], "interp", [ 16 ], self.sig_gen, [[ I[:,0], Q[:,0] ]], Fc=center_freq)

    
    def run(self, **kwargs):
        self.urc.IOS.Members["iptr_A"].Data = self.gen_5G_stimuli()
        self.urc.run()
        self.dut.IOS.Members["io_in_bits_rs1"].Data = self.urc.IOS.Members["Z"].Data
        rotation_vec, Fc = self.gen_cordic_stimuli(Fs=(self.sig_gen.s_struct['Fs'] * 16), Fs_divisor=5,
                                                   vec_len=len(self.urc.IOS.Members["Z"].Data))
        new_I, new_Q = self.rotate(self.urc.IOS.Members["Z"].Data.real, self.urc.IOS.Members["Z"].Data.imag,
                                   rotation_vec) 
        self.plot(new_I, new_Q, Fc)


class TrigFuncTestbench(CordicTestbench):
    """Testbench for using CORDIC for trigonometric ops

    Parameters
    ----------
    mantissa_bits : int
        Number of mantissa bits in fixed-point representation 
    fraction_bits : int
        Number of fraction bits in fixed-point representation
    iterations : int
        Number of CORDIC iterations
    
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dut = Cordic(mantissa_bits=self.mantissa_bits,
                          fraction_bits=self.fraction_bits,
                          iterations=self.iterations)

    def generate_stimuli(self, cordic_op: str, index: int):
        if cordic_op == "Sine" or cordic_op == "Cosine":
            self.dut.IOS.Members["io_in_bits_rs1"].Data = self.all_to_fp(
                np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
            )
            size = np.size(self.dut.IOS.Members["io_in_bits_rs1"].Data)
            self.dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            self.dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
        elif cordic_op == "Arctan":
            self.dut.IOS.Members["io_in_bits_rs1"].Data = self.all_to_fp(
                np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
            )
            size = np.size(self.dut.IOS.Members["io_in_bits_rs1"].Data)
            self.dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            self.dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
        elif cordic_op == "Cosh" or cordic_op == "Sinh":
            self.dut.IOS.Members["io_in_bits_rs1"].Data = self.all_to_fp(
                np.arange(-1.1, 1.1, 0.01, dtype=float).reshape(-1, 1)
            )
            size = np.size(self.dut.IOS.Members["io_in_bits_rs1"].Data)
            self.dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            self.dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
        elif cordic_op == "Arctanh":
            self.dut.IOS.Members["io_in_bits_rs1"].Data = self.all_to_fp(
                np.arange(-0.8, 0.8, 0.01, dtype=float).reshape(-1, 1)
            )
            size = np.size(self.dut.IOS.Members["io_in_bits_rs1"].Data)
            self.dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            self.dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
        elif cordic_op == "Exponential":
            self.dut.IOS.Members["io_in_bits_rs1"].Data = self.all_to_fp(
                np.arange(-1.1, 1.1, 0.01, dtype=float).reshape(-1, 1)
            )
            size = np.size(self.dut.IOS.Members["io_in_bits_rs1"].Data)
            self.dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            self.dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
        elif cordic_op == "Log":
            self.dut.IOS.Members["io_in_bits_rs1"].Data = self.all_to_fp(
                np.arange(0.15, 3.0, 0.01, dtype=float).reshape(-1, 1)
            )
            size = np.size(self.dut.IOS.Members["io_in_bits_rs1"].Data)
            self.dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            self.dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
        else:
            self.print_log(type="E", msg="No supported CORDIC operation was given!")

        self.dut.IOS.Members["io_in_bits_op"].Data = np.full(
            self.dut.IOS.Members["io_in_bits_rs1"].Data.size, index
        ).reshape(-1, 1)


    def plot(self, cordic_op: str):
        hfont = {"fontname": "Sans"}
        fig, ax = plt.subplots()
        test_data = np.array(
            [
                methods.to_double_single(
                    methods.to_fixed_point(data_point, self.dut.mb, self.dut.fb), self.dut.mb, self.dut.fb
                )
                for data_point in self.dut.IOS.Members["io_in_bits_rs1"].Data[:, 0]
            ]
        ).reshape(-1, 1)
        output = np.array(
            [
                methods.to_double_single(
                    methods.to_fixed_point(data_point, self.dut.mb, self.dut.fb), self.dut.mb, self.dut.fb
                )
                for data_point in self.dut.IOS.Members["io_out_bits_dOut"].Data[:, 0]
            ]
        ).reshape(-1, 1)
        bits_info = f" mb={self.dut.mb}, fb={self.dut.fb}"
        ax.set_title(f"{self.model} {cordic_op}" + bits_info)
        if cordic_op == "Sine":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\sin(\theta)$")
            reference = np.sin(test_data)
        elif cordic_op == "Cosine":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\cos(\theta)$")
            reference = np.cos(test_data)
        elif cordic_op == "Arctan":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\arctan(\theta)$")
            reference = np.arctan(test_data)
        elif cordic_op == "Sinh":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\sinh(\theta)$")
            reference = np.sinh(test_data)
        elif cordic_op == "Cosh":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\cosh(\theta)$")
            reference = np.cosh(test_data)
        elif cordic_op == "Arctanh":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$arctanh(\theta)$")
            reference = np.arctanh(test_data)
        elif cordic_op == "Exponential":
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$e^{\theta}$")
            reference = np.exp(test_data)
        elif cordic_op == "Log":
            ax.set_xlabel(r"a")
            ax.set_ylabel(r"ln (a)")
            reference = np.log(test_data)
        error = abs(output - reference)
        ax.plot(test_data, reference, label="reference")
        ax.plot(test_data, output, label="cordic")
        ax2 = ax.twinx()
        ax2.set_ylabel("|error|")
        ax2.plot(test_data, error, color="red", label="error")
        ax.legend(loc=2)
        ax2.legend(loc=1)
        fig.tight_layout()
        plt.draw()
        plt.show(block=False)


    def run(self, **kwargs):
        """Run simulation
        
        Parameters
        ----------
        cordic_op : str
            Cordic operation to perform (e.g. "Sine", "Cosine")
        index : int
            Index corresponding to the given cordic_op (only for model='sv')
        show : bool
            Show plots at the end of the simulation
            
        """
        cordic_op = kwargs["cordic_op"]
        index = kwargs.get("index", 0)
        self.generate_stimuli(cordic_op, index)
        self.dut.model = self.model
        self.dut.run()
        show = kwargs.get("show", True)
        if show:
            self.plot(cordic_op)


if __name__ == "__main__":
    import testbench
    #tb = testbench.TrigFuncTestbench(mantissa_bits=4, fraction_bits=12, iterations=16)
    #tb.model = "sv"
    #for idx, op in enumerate(["Sine", "Cosine"]):
    #    tb.run(cordic_op=op, index=idx)
    tb = testbench.NRTestbench(mantissa_bits=4, fraction_bits=12, iterations=16)
    tb.model = "py"
    tb.run()
    input("Press enter to close figures and exit.")
