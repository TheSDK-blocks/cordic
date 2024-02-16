"""
========
Cordic
========
"""

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
from cordic.model_2 import model_2
from BitVector import BitVector
import cordic.cordic_common.methods as methods
import cordic.cordic_common.cordic_types as cordic_types


class cordic(rtl, spice, thesdk):
    def __init__(
        self,
        *arg,
        mantissa_bits=12,
        fraction_bits=4,
        iterations=16,
        function="Sine",
    ):
        """Cordic parameters and attributes
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
        self.IOS.Members["io_in_bits_control"] = IO()
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
        self.fb = fraction_bits
        self.iters = iterations
        self.function = function

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
        rs1: np.ndarray = self.IOS.Members["io_in_bits_rs1"].Data
        rs2: np.ndarray = self.IOS.Members["io_in_bits_rs2"].Data
        rs3: np.ndarray = self.IOS.Members["io_in_bits_rs3"].Data
        ops: np.ndarray = self.IOS.Members["io_in_bits_control"].Data

        d_out = np.zeros(d_in.size, dtype=np.float32)
        rs1_out = np.zeros(d_in.size, dtype=np.float32)
        rs2_out = np.zeros(d_in.size, dtype=np.float32)
        rs3_out = np.zeros(d_in.size, dtype=np.float32)

        dut = model_2(self.mb, self.fb, self.iters)

        for i in range(0, d_in.size):
            dut.d_in   = methods.to_fixed_point(d_in[i][0], self.mb, self.fb)
            dut.rs1_in = methods.to_fixed_point(rs1[i][0], self.mb, self.fb)
            dut.rs2_in = methods.to_fixed_point(rs2[i][0], self.mb, self.fb)
            dut.rs3_in = methods.to_fixed_point(rs3[i][0], self.mb, self.fb)
            dut.op = ops[i]
            dut.run()
            d_out[i] = methods.to_double_single(dut.d_out, self.mb, self.fb)
            rs1_out[i] = methods.to_double_single(dut.rs1_out, self.mb, self.fb)
            rs2_out[i] = methods.to_double_single(dut.rs2_out, self.mb, self.fb)
            rs3_out[i] = methods.to_double_single(dut.rs3_out, self.mb, self.fb)
        self.IOS.Members["io_out_bits_dOut"].Data = d_out.reshape(-1, 1)
        self.IOS.Members["io_out_bits_cordic_x"].Data = rs1_out.reshape(-1, 1)
        self.IOS.Members["io_out_bits_cordic_y"].Data = rs2_out.reshape(-1, 1)
        self.IOS.Members["io_out_bits_cordic_z"].Data = rs3_out.reshape(-1, 1)


    def control_string_to_int(self, string):
        """
        Convert function name in string to an int.
        These are defined in the chisel model (e.g. TrigFuncPreProcessor)
        """
        if string == "Sine":
            return 0
        elif string == "Cosine":
            return 1
        elif string == "Arctan":
            return 2
        elif string == "Sinh":
            return 3
        elif string == "Cosh":
            return 4
        elif string == "Arctanh":
            return 5
        elif string == "Exponential":
            return 6
        elif string == "Log":
            return 7


    def convert_inputs(self):
        # TODO: restructure and replace with inlist
        for i in range(0, len(self.IOS.Members["io_in_bits_rs1"].Data)):
            self.IOS.Members["io_in_bits_rs1"].Data[i] = \
                methods.to_fixed_point(self.IOS.Members["io_in_bits_rs1"].Data[i][0], self.mb, self.fb).int_val()
        for i in range(0, len(self.IOS.Members["io_in_bits_rs2"].Data)):
            self.IOS.Members["io_in_bits_rs2"].Data[i] = \
                methods.to_fixed_point(self.IOS.Members["io_in_bits_rs2"].Data[i][0], self.mb, self.fb).int_val()
        for i in range(0, len(self.IOS.Members["io_in_bits_rs3"].Data)):
            self.IOS.Members["io_in_bits_rs3"].Data[i] = \
                methods.to_fixed_point(self.IOS.Members["io_in_bits_rs3"].Data[i][0], self.mb, self.fb).int_val()
        for i in range(0, len(self.IOS.Members["io_in_bits_control"].Data)):
            self.IOS.Members["io_in_bits_control"].Data[i] = \
                self.control_string_to_int(self.IOS.Members["io_in_bits_control"].Data[i][0])

    def convert_outputs(self):
        # TODO: replace with outlist
        converted = [
            self.IOS.Members["io_out_bits_dOut"],
            self.IOS.Members["io_out_bits_cordic_x"],
            self.IOS.Members["io_out_bits_cordic_y"],
            self.IOS.Members["io_out_bits_cordic_z"],
        ]
        for ios in converted:
            new_arr = np.empty(len(ios.Data), dtype='float32')
            for i in range(0, len(ios.Data)):
                new_arr[i] = methods.to_double_single(BitVector(intVal=ios.Data[i][0], size=self.mb+self.fb), self.mb, self.fb)
            ios.Data = new_arr.reshape(-1, 1)


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
            self.convert_inputs()
            sim = os.getenv("SIM", "icarus")
            clock_name = "clock"
            reset_name = "reset"
            in_ios = {
                "rs1": "io_in_bits_rs1",
                "rs2": "io_in_bits_rs2",
                "rs3": "io_in_bits_rs3",
                "control": "io_in_bits_control",
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
                    self.vlogsrcpath + "/CordicTop.v",
                    self.vlogsrcpath + "/cocotb_iverilog_dump.v",
                ],
                hdl_toplevel="CordicTop",
                always=True,
            )
            runner.test(
                hdl_toplevel="CordicTop",
                test_module="test_cordic",
                test_dir=os.path.join(thesdk.HOME, "Entities/cordic/cordic"),
                plusargs=[
                    f"+in_iofiles={','.join(in_iofiles)}",
                    f"+in_ionames={','.join(in_ionames)}",
                    f"+out_iofiles={','.join(out_iofiles)}",
                    f"+out_ionames={','.join(out_ionames)}",
                    f"+clock={clock_name}",
                    f"+reset={reset_name}",
                ],
                waves=self.waves,
            )

            # Read iofiles into Python datatype
            for i, iofile in enumerate(out_iofileinsts):
                iofile.read(dtype="int")
                self.IOS.Members[out_ionames[i]].Data = iofile.Data
            self.convert_outputs()

if __name__ == "__main__":
    """Quick and dirty self test"""
    import numpy as np
    import matplotlib.pyplot as plt
    dut = cordic(mantissa_bits=4, fraction_bits=12, iterations=16, function="Sine")

    test_data = \
        np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
    size = np.size(test_data)
    dut.model = "sv"
    dut.IOS.Members["io_in_bits_rs1"].Data = np.copy(test_data)
    dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
    dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
    dut.IOS.Members["io_in_bits_control"].Data = np.full(
        dut.IOS.Members["io_in_bits_rs1"].Data.size, dut.function
    ).reshape(-1, 1)

    dut.run()
    output = np.array(
        [
            data_point for data_point in dut.IOS.Members["io_out_bits_dOut"].Data[:, 0]
        ]
    ).reshape(-1, 1)
    fig, ax = plt.subplots()
    bits_info = f" mb={dut.mb}, fb={dut.fb}"
    ax.set_title(f"{dut.model} {dut.function}" + bits_info)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\sin(\theta)$")
    reference = np.sin(test_data)
    error = abs(output - reference)
    ax.plot(test_data, reference, label="reference")
    ax.plot(test_data, output, label="cordic", color="green")
    ax2 = ax.twinx()
    ax2.set_ylabel("|error|")
    ax2.plot(test_data, error, color="red", label="error")
    ax.legend(loc=2)
    ax2.legend(loc=1)
    fig.tight_layout()
    plt.draw()
    plt.show()

