"""
========
Inverter
========

Inverter model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

For reference of the markup syntax
https://docutils.sourceforge.io/docs/user/rst/quickref.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this
entity may be outdated and incomplete. Regardless of that, every day
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.


Role of section 'if __name__=="__main__"'
--------------------------------------------

This section is for self testing and interfacing of this class. The content of
it is fully up to designer. You may use it for example to test the
functionality of the class by calling it as ``pyhon3 __init__.py`` or you may
define how it handles the arguments passed during the invocation. In this
example it is used as a complete self test script for all the simulation models
defined for the inverter.

"""

import os
import sys

if not (os.path.abspath("../../thesdk") in sys.path):
    sys.path.append(os.path.abspath("../../thesdk"))

if not (os.path.abspath("../cordic_common") in sys.path):
    sys.path.append(os.path.abspath("../cordic_common"))

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


class CordicAccelerator(rtl, spice, thesdk):
    def __init__(
        self,
        *arg,
        mantissa_bits=12,
        fractional_bits=4,
        iterations=16,
        function="Sine",
        mode=cordic_types.cordic_mode.ROTATION,
        rot_type=cordic_types.rotation_type.CIRCULAR,
    ):
        """CordicAccelerator parameters and attributes
        Parameters
        ----------
            *arg :
            If any arguments are defined, the first one should be the
            parent instance

            mantissa_bits :
            How many mantissa bits are used in the fixed-point repr.

            fractional_bits :
            How many fractional bits are used in the fixed-point repr.

            iterations :
            How many iterations the CORDIC is supposed to have

            function :
            Which operation the CORDIC is calculating

        """
        self.print_log(type="I", msg="Initializing %s" % (__name__))

        self.IOS.Members["io_in_valid"] = IO()
        self.IOS.Members["io_in_bits_rs1"] = IO()
        self.IOS.Members["io_in_bits_rs2"] = IO()
        self.IOS.Members["io_in_bits_rs3"] = IO()
        self.IOS.Members["io_in_bits_op"] = IO()
        self.IOS.Members["io_out_bits_dOut"] = IO()
        self.IOS.Members["io_out_bits_cordic_x"] = IO()
        self.IOS.Members["io_out_bits_cordic_y"] = IO()
        self.IOS.Members["io_out_bits_cordic_z"] = IO()
        self.IOS.Members["io_out_valid"] = IO()

        self.IOS.Members["clock"] = IO()
        self.IOS.Members["reset"] = IO()

        self.model = "py"  # Can be set externalouly, but is not propagated

        # Model related properties
        self.mb = mantissa_bits
        self.fb = fractional_bits
        self.iters = iterations
        self.function = function
        self.mode = mode
        self.type = rot_type

        # Function index for hardware implementation
        self.function_idx = 0

        # Simulation related properties
        self.waves = False

    def main(self):
        """The main python description of the operation. Contents fully up to
        designer, however, the
        IO's should be handled bu following this guideline:

        To isolate the internal processing from IO connection assigments,
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        """
        d_in: np.ndarray = self.IOS.Members["io_in_bits_rs1"].Data
        ops: np.ndarray = self.IOS.Members["io_in_bits_op"].Data

        assert d_in.size == ops.size, "Input vectors must be same size!"

        self.IOS.Members["io_out_bits_dOut"].Data = np.zeros(d_in.size)

        dut = model_2(self.mb, self.fb, self.iters)

        for i in range(0, d_in.size):
            dut.d_in = methods.to_fixed_point(d_in[i][0], self.mb, self.fb)
            dut.op = ops[i]
            dut.run()
            self.IOS.Members["io_out_bits_dOut"].Data[i] = methods.to_double_single(
                dut.d_out, self.mb, self.fb
            )

    def run(self, *arg):
        """The default name of the method to be executed. This means:
        parameters and attributes
        control what is executed if run method is executed.
        By this we aim to avoid the need of
        documenting what is the execution method. It is always self.run.

        Parameters
        ----------
        *arg :
            The first argument is assumed to be the queue for the parallel
            processing defined in the parent,
            and it is assigned to self.queue and self.par is set to True.

        """
        if self.model == "py":
            self.main()
        else:
            sim = os.getenv("SIM", "icarus")
            clock_name = "clock"
            reset_name = "reset"
            in_ios = {
                "rs1": "io_in_bits_rs1",
                "rs2": "io_in_bits_rs2",
                "rs3": "io_in_bits_rs3",
                "op": "io_in_bits_op",
            }
            out_ios = {
                "x": "io_out_bits_cordic_x",
                "y": "io_out_bits_cordic_y",
                "z": "io_out_bits_cordic_z",
                "dOut": "io_out_bits_dOut",
            }
            in_iofiles = []
            in_ionames = []
            out_iofiles = []
            out_ionames = []
            out_iofileinsts = []
            for key, value in in_ios.items():
                file = rtl_iofile(
                    self,
                    name=key,
                    dir="in",
                    iotype="sample",
                    ionames=[value],
                    datatype="sint",
                )
                file.file = self.simpath + "/" + file.name + ".txt"
                data = self.IOS.Members[value].Data
                if data is None:
                    data = np.empty((1, 1))
                file.write(data=data)
                in_iofiles.append(file.file)
                in_ionames.append(value)
            for key, value in out_ios.items():
                file = rtl_iofile(
                    self,
                    name=key,
                    dir="out",
                    iotype="sample",
                    ionames=[value],
                    datatype="sint",
                )
                file.file = self.simpath + "/" + file.name + ".txt"
                out_iofiles.append(file.file)
                out_ionames.append(value)
                out_iofileinsts.append(file)

            runner = get_runner(sim)
            runner.build(
                verilog_sources=[
                    self.vlogsrc,
                    self.vlogsrcpath + "/cocotb_iverilog_dump.v",
                ],
                hdl_toplevel=self.name,
                always=True,
            )
            runner.test(
                hdl_toplevel=self.name,
                test_module=f"test_{self.name}",
                plusargs=[
                    f"+in_iofiles={','.join(in_iofiles)}",
                    f"+in_ionames={','.join(in_ionames)}",
                    f"+out_iofiles={','.join(out_iofiles)}",
                    f"+out_ionames={','.join(out_ionames)}",
                    f"+clock={clock_name}",
                    f"+reset={reset_name}",
                    f"+op={self.function_idx}",
                ],
                waves=self.waves,
            )

            # Read iofiles into Python datatype
            for i, iofile in enumerate(out_iofileinsts):
                iofile.read(dtype="int")
                self.IOS.Members[out_ionames[i]].Data = iofile.Data


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    def comma_separated_type(value):
        return value.split(",")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # Implement argument parser
    parser = argparse.ArgumentParser(description="Parse selectors")
    parser.add_argument(
        "--show",
        dest="show",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Show figures on screen",
    )
    parser.add_argument(
        "--models",
        help="Models to run",
        choices=["py", "sv"],
        nargs="+",
    )
    parser.add_argument(
        "--mantissa-bits",
        help="Mantissa bits",
        type=int,
    )
    parser.add_argument(
        "--fraction-bits",
        help="Fraction bits",
        type=int,
    )
    parser.add_argument(
        "--iterations",
        help=("Number of iterations to run"),
        type=int,
    )
    parser.add_argument(
        "--cordic-ops",
        help="Cordic operations enabled",
        type=comma_separated_type,
    )
    parser.add_argument(
        "--waves",
        help="Dump waveform file",
        type=str2bool,
    )
    args = parser.parse_args()

    models = args.models
    duts = []

    mantissa_bits = 4
    fractional_bits = 12
    iterations = 14

    n_values = 10
    # test_data = (np.random.random(size=n_values) * max_value).reshape(-1, 1)
    clk = np.array([0 if i % 2 == 0 else 1 for i in range(2 * n_values)]).reshape(-1, 1)

    for model in models:
        for i, function_name in enumerate(args.cordic_ops):
            dut = CordicAccelerator(
                mantissa_bits=10,  # placeholder
                fractional_bits=10,  # placeholder
                iterations=iterations,
            )
            dut.model = model
            dut.function = function_name
            dut.function_idx = i
            dut.waves = args.waves

            if model == "py":
                function = function_name
            elif model == "sv":
                function = i

            if function_name == "Sine" or function_name == "Cosine":
                test_data = np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
            elif function_name == "Arctan":
                test_data = np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
            elif function_name == "Cosh" or function_name == "Sinh":
                test_data = np.arange(-1.1, 1.1, 0.01, dtype=float).reshape(-1, 1)
            elif function_name == "Arctanh":
                test_data = np.arange(-0.8, 0.8, 0.01, dtype=float).reshape(-1, 1)
            elif function_name == "Exponential":
                test_data = np.arange(-1.1, 1.1, 0.01, dtype=float).reshape(-1, 1)
            elif function_name == "Log":
                test_data = np.arange(0.15, 3.0, 0.01, dtype=float).reshape(-1, 1)

            dut.IOS.Members["io_in_bits_rs1"].Data = np.array(
                [
                    methods.to_fixed_point(
                        data_point, mantissa_bits, fractional_bits
                    ).int_val()
                    for data_point in test_data[:, 0]
                ]
            ).reshape(-1, 1)

            dut.IOS.Members["io_in_bits_op"].Data = np.full(
                test_data.size, function
            ).reshape(-1, 1)

            dut.mb = mantissa_bits
            dut.fb = fractional_bits

            if model == "sv":
                dut.IOS.Members["io_in_bits_rs1"].Data = np.array(
                    [
                        methods.to_fixed_point(
                            x, mantissa_bits, fractional_bits
                        ).int_val()
                        for x in dut.IOS.Members["io_in_bits_rs1"].Data
                    ]
                ).reshape(-1, 1)

            dut.IOS.Members["clock"].Data = clk
            duts.append(dut)

    for dut in duts:
        dut.run()

        hfont = {"fontname": "Sans"}
        fig, ax1 = plt.subplots()

        bits_info = f" mb={dut.mb}, fb={dut.fb}"

        test_data = np.array(
            [
                methods.to_double_single(
                    methods.to_fixed_point(data_point, dut.mb, dut.fb), dut.mb, dut.fb
                )
                for data_point in dut.IOS.Members["io_in_bits_rs1"].Data[:, 0]
            ]
        ).reshape(-1, 1)
        output = dut.IOS.Members["io_out_bits_dOut"].Data.reshape(-1, 1)
        # import pdb; pdb.set_trace()
        if dut.model == "sv":
            output = np.array(
                [
                    methods.to_double_single(
                        BitVector(intVal=x, size=(dut.mb + dut.fb)), dut.mb, dut.fb
                    )
                    for x in output[:, 0]
                ]
            ).reshape(-1, 1)
        ax1.set_title(f"{dut.model} {dut.function}" + bits_info)

        if dut.function == "Sine":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\sin(\theta)$")
            reference = np.sin(test_data)
        elif dut.function == "Cosine":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\cos(\theta)$")
            reference = np.cos(test_data)
        elif dut.function == "Arctan":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\arctan(\theta)$")
            reference = np.arctan(test_data)
        elif dut.function == "Sinh":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\sinh(\theta)$")
            reference = np.sinh(test_data)
        elif dut.function == "Cosh":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\cosh(\theta)$")
            reference = np.cosh(test_data)
        elif dut.function == "Arctanh":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$arctanh(\theta)$")
            reference = np.arctanh(test_data)
        elif dut.function == "Exponential":
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$e^{\theta}$")
            reference = np.exp(test_data)
        elif dut.function == "Log":
            ax1.set_xlabel(r"a")
            ax1.set_ylabel(r"ln (a)")
            reference = np.log(test_data)

        error = abs(output - reference)
        ax1.plot(test_data, reference, label="reference")
        ax1.plot(test_data, output, label="cordic")
        ax2 = ax1.twinx()
        ax2.set_ylabel("|error|")
        ax2.plot(test_data, error, color="red", label="error")
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        fig.tight_layout()
        plt.draw()
    plt.show()
