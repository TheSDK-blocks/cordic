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
        self.fb = fraction_bits
        self.iters = iterations
        self.function = function

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
        rs1: np.ndarray = self.IOS.Members["io_in_bits_rs1"].Data
        rs2: np.ndarray = self.IOS.Members["io_in_bits_rs2"].Data
        rs3: np.ndarray = self.IOS.Members["io_in_bits_rs3"].Data
        ops: np.ndarray = self.IOS.Members["io_in_bits_op"].Data

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
                    f"+op={self.function_idx}",
                ],
                waves=self.waves,
            )

            # Read iofiles into Python datatype
            for i, iofile in enumerate(out_iofileinsts):
                iofile.read(dtype="int")
                self.IOS.Members[out_ionames[i]].Data = iofile.Data
            self.convert_outputs()

    def gen_5G_stimuli(self):
        from URC_toolkit import URC_toolkit

        include_dir = os.path.join(
            os.path.abspath(thesdk.HOME),
            "Entities/ACoreTests/build/tests/programs/dsp-tests/rv32im/sw-build/include/",
        )
        URC_tk = URC_toolkit()
        QAM, osr, BWP, BW, in_bits, vec_len = URC_tk.load_sig_gen_yaml(
            include_dir + "iq-vecs_sigparams.yml"
        )
        signal_gen, I_sig, Q_sig, _ = URC_tk.init_NR_siggen(
            ["I"], QAM, osr, BWP, BW, in_bits, 0, 16, 0
        )
        interp_sig = URC_tk.interpolate_sig(signal_gen.IOS.Members["out"].Data, 16)
        I = signal_gen.IOS.Members["out"].Data[:, 0]
        Q = signal_gen.IOS.Members["out"].Data[:, 1]
        return I, Q, signal_gen, URC_tk


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
        type=str2bool,
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
    functions = args.cordic_ops
    duts = []

    mantissa_bits = args.mantissa_bits
    fractional_bits = args.fraction_bits
    iterations = args.iterations

    n_values = 10
    # test_data = (np.random.random(size=n_values) * max_value).reshape(-1, 1)
    clk = np.array([0 if i % 2 == 0 else 1 for i in range(2 * n_values)]).reshape(-1, 1)

    for model in models:
        for i, function_name in enumerate(functions):
            dut = cordic(
                mantissa_bits=10,  # placeholder
                fraction_bits=10,  # placeholder
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

            def all_to_fp(tdata):
                # Convert np array of floats to np array fixed point
                return np.array(
                    [
                        methods.to_fixed_point(
                            data_point, mantissa_bits, fractional_bits
                        ).int_val()
                        for data_point in tdata[:, 0]
                    ]
                ).reshape(-1, 1)

            if function_name == "Sine" or function_name == "Cosine":
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(
                    np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
                )
                size = np.size(dut.IOS.Members["io_in_bits_rs1"].Data)
                dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
                dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            elif function_name == "Arctan":
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(
                    np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
                )
                size = np.size(dut.IOS.Members["io_in_bits_rs1"].Data)
                dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
                dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            elif function_name == "Cosh" or function_name == "Sinh":
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(
                    np.arange(-1.1, 1.1, 0.01, dtype=float).reshape(-1, 1)
                )
                size = np.size(dut.IOS.Members["io_in_bits_rs1"].Data)
                dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
                dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            elif function_name == "Arctanh":
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(
                    np.arange(-0.8, 0.8, 0.01, dtype=float).reshape(-1, 1)
                )
                size = np.size(dut.IOS.Members["io_in_bits_rs1"].Data)
                dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
                dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            elif function_name == "Exponential":
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(
                    np.arange(-1.1, 1.1, 0.01, dtype=float).reshape(-1, 1)
                )
                size = np.size(dut.IOS.Members["io_in_bits_rs1"].Data)
                dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
                dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            elif function_name == "Log":
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(
                    np.arange(0.15, 3.0, 0.01, dtype=float).reshape(-1, 1)
                )
                size = np.size(dut.IOS.Members["io_in_bits_rs1"].Data)
                dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
                dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int64).reshape(-1, 1)
            elif function_name == "Upconvert":
                I, Q, signal_gen, URC_tk = dut.gen_5G_stimuli()
                dut.IOS.Members["io_in_bits_rs1"].Data = all_to_fp(I.reshape(-1, 1))
                
                dut.IOS.Members["io_in_bits_rs2"].Data = all_to_fp(Q.reshape(-1, 1))
                # Create a vector that rotates 360 degrees continuously
                Fs = signal_gen.s_struct['Fs']
                center_freq = 5000000
                rot_vec = np.linspace(np.pi/2, -np.pi/2, round(Fs/center_freq), endpoint=False)
                #rot_vec = np.zeros(round(signal_gen.s_struct['Fs']/1e6))
                # Check how many times it manages to rotate during the length of the input signal
                # Truncate output
                repeats = len(dut.IOS.Members["io_in_bits_rs1"].Data) // len(rot_vec)
                # How many samples remain to be filled of a partial circle
                remainder = len(dut.IOS.Members["io_in_bits_rs1"].Data) % len(rot_vec)
                # Concatenate repeated rotation and the remainder vector
                rot_vec_extended = np.concatenate(
                        (np.tile(rot_vec, repeats), rot_vec[:remainder])
                        ).reshape(-1, 1)
                new_I = (I * np.cos(rot_vec_extended).reshape(1,-1) - \
                        Q * np.sin(rot_vec_extended).reshape(1,-1)).reshape(-1, 1)
                new_Q = (I * np.sin(rot_vec_extended).reshape(1,-1) + \
                        Q * np.cos(rot_vec_extended).reshape(1,-1)).reshape(-1, 1)
                #dut.IOS.Members["io_in_bits_rs3"].Data = all_to_fp(rot_vec_extended)
                dut.signal_gen = signal_gen
                dut.URC_tk = URC_tk
                import pdb; pdb.set_trace()
                dut.URC_tk.plot_5G_output(["I","Q"], "interp", [ 16 ], dut.signal_gen, [[ new_I[:,0], new_Q[:,0] ]])
                input("Ess Prenter e toxit")
                exit()
                

            dut.IOS.Members["io_in_bits_op"].Data = np.full(
                dut.IOS.Members["io_in_bits_rs1"].Data.size, function
            ).reshape(-1, 1)

            dut.mb = mantissa_bits
            dut.fb = fractional_bits

            dut.IOS.Members["clock"].Data = clk
            duts.append(dut)

    # Prepare figures
    plot_list = []
    # Indexing breaks if we have only one plot,
    # so we create a dummy plot if n_models == 1
    n_models = max(len(models), 2)
    for i, function_name in enumerate(functions):
        fig, ax1 = plt.subplots(n_models, 1)
        plot_list.append((fig, ax1))

    for dut in duts:
        dut.run()

        hfont = {"fontname": "Sans"}

        def plot_trigonometric():
            fig, ax1 = plot_list[dut.function_idx]
            ax_idx = models.index(dut.model)

            bits_info = f" mb={dut.mb}, fb={dut.fb}"

            test_data = np.array(
                [
                    methods.to_double_single(
                        methods.to_fixed_point(data_point, dut.mb, dut.fb), dut.mb, dut.fb
                    )
                    for data_point in dut.IOS.Members["io_in_bits_rs1"].Data[:, 0]
                ]
            ).reshape(-1, 1)
            output = np.array(
                [
                    methods.to_double_single(
                        methods.to_fixed_point(data_point, dut.mb, dut.fb), dut.mb, dut.fb
                    )
                    for data_point in dut.IOS.Members["io_out_bits_dOut"].Data[:, 0]
                ]
            ).reshape(-1, 1)
            ax1[ax_idx].set_title(f"{dut.model} {dut.function}" + bits_info)

            if dut.function == "Sine":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$\sin(\theta)$")
                reference = np.sin(test_data)
            elif dut.function == "Cosine":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$\cos(\theta)$")
                reference = np.cos(test_data)
            elif dut.function == "Arctan":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$\arctan(\theta)$")
                reference = np.arctan(test_data)
            elif dut.function == "Sinh":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$\sinh(\theta)$")
                reference = np.sinh(test_data)
            elif dut.function == "Cosh":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$\cosh(\theta)$")
                reference = np.cosh(test_data)
            elif dut.function == "Arctanh":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$arctanh(\theta)$")
                reference = np.arctanh(test_data)
            elif dut.function == "Exponential":
                ax1[ax_idx].set_xlabel(r"$\theta$")
                ax1[ax_idx].set_ylabel(r"$e^{\theta}$")
                reference = np.exp(test_data)
            elif dut.function == "Log":
                ax1[ax_idx].set_xlabel(r"a")
                ax1[ax_idx].set_ylabel(r"ln (a)")
                reference = np.log(test_data)

            error = abs(output - reference)
            ax1[ax_idx].plot(test_data, reference, label="reference")
            ax1[ax_idx].plot(test_data, output, color="green", label="cordic")
            ax2 = ax1[ax_idx].twinx()
            ax2.set_ylabel("|error|")
            ax2.plot(test_data, error, color="red", label="error")
            ax1[ax_idx].legend(loc=2)
            ax2.legend(loc=1)
            fig.tight_layout()
            plt.draw()

        if dut.function == "Upconvert":
            I_vec = dut.IOS.Members["io_out_bits_cordic_x"].Data[:, 0]
            Q_vec = dut.IOS.Members["io_out_bits_cordic_y"].Data[:, 0]
            import pdb; pdb.set_trace()
            dut.URC_tk.plot_5G_output(["I"], "interp_decim", [ 16 ], dut.signal_gen, [[ np.array(I_vec).astype("int16"), np.array(Q_vec).astype("int16") ]])
        else:
            plot_trigonometric()
    if args.show:
        if dut.function == "Upconvert":
            input("Press enter to exit")
        else:
            plt.show()
