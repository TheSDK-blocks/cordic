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

from thesdk import thesdk, IO
from rtl import rtl, rtl_iofile
from spice import spice

import numpy as np

from model_1 import model_1
import methods
import cordic_types


class trigonometric_function:
    SIN = 0
    COS = 1
    ARCSIN = 2
    ARCCOS = 3
    ARCTAN = 4
    SINH = 5
    COSH = 6
    ARCTANH = 7
    EXPONENTIAL = 8
    LOG = 9


class general_cordic(rtl, spice, thesdk):
    def __init__(
        self,
        *arg,
        mantissa_bits=12,
        fractional_bits=4,
        iterations=16,
        function=trigonometric_function.SIN,
        mode=cordic_types.cordic_mode.ROTATION,
        rot_type=cordic_types.rotation_type.CIRCULAR,
    ):
        """Inverter parameters and attributes
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

        Attributes
        ----------
        proplist : array_like
            List of strings containing the names of attributes whose
            values are to be copied
            from the parent

        Rs : float
            Sampling rate [Hz] of which the input values are assumed to
            change. Default: 100.0e6

        vdd : float
            Supply voltage [V] for inverter analog simulation. Default 1.0.

        IOS : Bundle
            Members of this bundle are the IO's of the entity.
            See documentation of thsdk package.
            Default members defined as

            self.IOS.Members['A']=IO() # Pointer for input data
            self.IOS.Members['Z']= IO() # pointer for oputput data
            self.IOS.Members['control_write']= IO() # Piter for control IO
              for rtl simulations

        model : string
            Default 'py' for Python. See documentation of thsdk package
            for more details.

        """
        self.print_log(type="I", msg="Initializing %s" % (__name__))
        self.proplist = ["Rs", "vdd"]
        self.Rs = 100e6  # Sampling frequency
        self.vdd = 1.0

        self.IOS.Members["X_IN"] = IO()
        self.IOS.Members["Y_IN"] = IO()
        self.IOS.Members["Z_IN"] = IO()
        self.IOS.Members["X_OUT"] = IO()
        self.IOS.Members["Y_OUT"] = IO()
        self.IOS.Members["Z_OUT"] = IO()

        self.IOS.Members["CLK"] = IO()
        self.IOS.Members["RST"] = IO()

        self.model = "py"  # Can be set externalouly, but is not propagated

        self.mb = mantissa_bits
        self.fb = fractional_bits
        self.iters = iterations
        self.function = function
        self.mode = mode
        self.type = rot_type

        # this copies the parameter values from the parent based
        # on self.proplist
        if len(arg) >= 1:
            parent = arg[0]
            self.copy_propval(parent, self.proplist)
            self.parent = parent

        self.init()

    def init(self):
        """Method to re-initialize the structure if the attribute values are
        changed after creation.

        """
        pass

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
        x_in: np.ndarray = self.IOS.Members["X_IN"].Data
        y_in: np.ndarray = self.IOS.Members["Y_IN"].Data
        z_in: np.ndarray = self.IOS.Members["Z_IN"].Data

        assert x_in.size == y_in.size, "Input vectors must be same size!"
        assert x_in.size == z_in.size, "Input vectors must be same size!"

        self.IOS.Members["X_OUT"].Data = np.zeros(x_in.size)
        self.IOS.Members["Y_OUT"].Data = np.zeros(x_in.size)
        self.IOS.Members["Z_OUT"].Data = np.zeros(x_in.size)

        dut = model_1(self.mb, self.fb, self.iters)
        dut.set_mode(self.mode)
        dut.set_type(self.type)

        for i in range(0, x_in.size):
            dut.set_inputs(
                methods.to_fixed_point(x_in[i][0], self.mb, self.fb),
                methods.to_fixed_point(y_in[i][0], self.mb, self.fb),
                methods.to_fixed_point(z_in[i][0], self.mb, self.fb),
            )
            dut.run()
            self.IOS.Members["X_OUT"].Data[i] = methods.to_double(dut.x_out)
            self.IOS.Members["Y_OUT"].Data[i] = methods.to_double(dut.y_out)
            self.IOS.Members["Z_OUT"].Data[i] = methods.to_double(dut.z_out)

        # if self.par:
        #     self.queue.put(out)

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
            # This defines contents of modelsim control file executed
            # when interactive_rtl = True
            # Interactive control files
            if self.model == "icarus" or self.model == "ghdl":
                self.interactive_control_contents = """
                    set io_facs [list]
                    lappend io_facs "tb_inverter.A"
                    lappend io_facs "tb_inverter.Z"
                    lappend io_facs "tb_inverter.clock"
                    gtkwave::addSignalsFromList $io_facs
                    gtkwave::/Time/Zoom/Zoom_Full
                """
            else:
                self.interactive_control_contents = """
                    add wave \\
                    sim/:tb_inverter:A \\
                    sim/:tb_inverter:initdone \\
                    sim/:tb_inverter:clock \\
                    sim/:tb_inverter:Z
                    run -all
                    wave zoom full
                """

            if self.model == "ghdl":
                # With this structure you can control the signals
                # to be dumped to VCD pass
                self.simulator_control_contents = (
                    "version = 1.1  # Optional\n"
                    + "/tb_inverter/A\n"
                    + "/tb_inverter/Z\n"
                    + "/tb_inverter/clock\n"
                )

            if self.model in ["sv", "icarus"]:
                # Verilog simulation options here
                _ = rtl_iofile(
                    self,
                    name="A",
                    dir="in",
                    iotype="sample",
                    ionames=["A"],
                    datatype="sint",
                )
                f = rtl_iofile(
                    self,
                    name="Z",
                    dir="out",
                    iotype="sample",
                    ionames=["Z"],
                    datatype="sint",
                )
                # This is to avoid sampling time confusion with Icarus
                if self.lang == "sv":
                    f.rtl_io_sync = "@(negedge clock)"
                elif self.lang == "vhdl":
                    f.rtl_io_sync = "falling_edge(clock)"

                # Defines the sample rate
                self.rtlparameters = dict(
                    [
                        ("g_Rs", ("real", self.Rs)),
                    ]
                )
                self.run_rtl()
                self.IOS.Members["Z"].Data = (
                    self.IOS.Members["Z"].Data[:, 0].astype(int).reshape(-1, 1)
                )
            elif self.model == "vhdl" or self.model == "ghdl":
                # VHDL simulation options here
                _ = rtl_iofile(
                    self, name="A", dir="in", iotype="sample", ionames=["A"]
                )  # IO file for input A
                f = rtl_iofile(
                    self,
                    name="Z",
                    dir="out",
                    iotype="sample",
                    ionames=["Z"],
                    datatype="int",
                )
                if self.lang == "sv":
                    f.rtl_io_sync = "@(negedge clock)"
                elif self.lang == "vhdl":
                    f.rtl_io_sync = "falling_edge(clock)"
                # Defines the sample rate
                self.rtlparameters = dict(
                    [
                        ("g_Rs", ("real", self.Rs)),
                    ]
                )
                self.run_rtl()
                self.IOS.Members["Z"].Data = (
                    self.IOS.Members["Z"].Data.astype(int).reshape(-1, 1)
                )

            if self.par:
                self.queue.put(self.IOS.Members)

    def define_io_conditions(self):
        """This overloads the method called by run_rtl method. It defines
        the read/write conditions for the files

        """
        if self.lang == "sv":
            # Input A is read to verilog simulation after 'initdone' is set to
            # 1 by controller
            self.iofile_bundle.Members["A"].rtl_io_condition = "initdone"
            # Output is read to verilog simulation when all of the outputs are
            # valid, and after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members["Z"].rtl_io_condition_append(cond="&& initdone")
        elif self.lang == "vhdl":
            self.iofile_bundle.Members["A"].rtl_io_condition = "(initdone = '1')"
            # Output is read to verilog simulation when all of the outputs
            # are valid, and after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members["Z"].rtl_io_condition_append(
                cond="and initdone = '1'"
            )


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    # import matplotlib.pyplot as plt

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
    args = parser.parse_args()

    models = ["py"]
    duts = []
    functions = [
        trigonometric_function.SIN,
        trigonometric_function.COS,
        trigonometric_function.ARCTAN,
        trigonometric_function.SINH,
        trigonometric_function.COSH,
        trigonometric_function.ARCTANH
    ]

    mantissa_bits = 4
    frac_bits = 12
    iterations = 14

    max_value = 1
    n_values = 10
    # test_data = (np.random.random(size=n_values) * max_value).reshape(-1, 1)
    clk = np.array([0 if i % 2 == 0 else 1 for i in range(2 * n_values)]).reshape(-1, 1)
    print(methods.calc_k(iterations))

    for model in models:
        for function in functions:
            dut = general_cordic(
                mantissa_bits=mantissa_bits,
                fractional_bits=frac_bits,
                iterations=iterations,
            )
            dut.model = model
            dut.function = function

            if (
                function == trigonometric_function.SIN
                or function == trigonometric_function.COS
            ):
                test_data = np.arange(0.0, np.pi / 2, 0.01, dtype=float).reshape(-1, 1)
                dut.IOS.Members["X_IN"].Data = np.full(
                    test_data.size, 1 / methods.calc_k(iterations)
                ).reshape(-1, 1)
                dut.IOS.Members["Y_IN"].Data = np.full(test_data.size, 0).reshape(-1, 1)
                dut.IOS.Members["Z_IN"].Data = test_data
                dut.mode = cordic_types.cordic_mode.ROTATION
                dut.type = cordic_types.rotation_type.CIRCULAR
            elif function == trigonometric_function.ARCTAN:
                test_data = np.arange(0.0, 1.1, 0.01, dtype=float).reshape(-1, 1)
                dut.IOS.Members["X_IN"].Data = np.full(test_data.size, 1.0).reshape(
                    -1, 1
                )
                dut.IOS.Members["Y_IN"].Data = test_data
                dut.IOS.Members["Z_IN"].Data = np.full(test_data.size, 0).reshape(-1, 1)
                dut.mode = cordic_types.cordic_mode.VECTORING
                dut.type = cordic_types.rotation_type.CIRCULAR
            elif (
                function == trigonometric_function.COSH
                or function == trigonometric_function.SINH
            ):
                test_data = np.arange(0.0, 1.1, 0.01, dtype=float).reshape(-1, 1)
                dut.IOS.Members["X_IN"].Data = np.full(
                    test_data.size, 1 / 0.82816  # methods.calc_k(iterations)
                ).reshape(-1, 1)
                dut.IOS.Members["Y_IN"].Data = np.full(test_data.size, 0).reshape(-1, 1)
                dut.IOS.Members["Z_IN"].Data = test_data
                dut.mode = cordic_types.cordic_mode.ROTATION
                dut.type = cordic_types.rotation_type.HYPERBOLIC
            elif function == trigonometric_function.ARCTANH:
                test_data = np.arange(0.0, 0.8, 0.01, dtype=float).reshape(-1, 1)
                dut.IOS.Members["X_IN"].Data = np.full(test_data.size, 1.0).reshape(
                    -1, 1
                )
                dut.IOS.Members["Y_IN"].Data = test_data
                dut.IOS.Members["Z_IN"].Data = np.full(test_data.size, 0).reshape(-1, 1)
                dut.mode = cordic_types.cordic_mode.VECTORING
                dut.type = cordic_types.rotation_type.HYPERBOLIC

            dut.IOS.Members["CLK"] = clk
            duts.append(dut)

    for dut in duts:
        dut.init()
        dut.run()

        hfont = {"fontname": "Sans"}
        fig, ax1 = plt.subplots()

        if dut.function == trigonometric_function.SIN:
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\sin(\theta)$")
            ax1.set_title(f"{dut.model} sin")
            test_data = dut.IOS.Members["Z_IN"].Data
            reference = np.sin(test_data)
            output = dut.IOS.Members["Y_OUT"].Data.reshape(-1, 1)
        elif dut.function == trigonometric_function.COS:
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\cos(\theta)$")
            ax1.set_title(f"{dut.model} cos")
            test_data = dut.IOS.Members["Z_IN"].Data
            reference = np.cos(test_data)
            output = dut.IOS.Members["X_OUT"].Data.reshape(-1, 1)
        elif dut.function == trigonometric_function.ARCTAN:
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\arctan(\theta)$")
            ax1.set_title(f"{dut.model} arctan")
            test_data = dut.IOS.Members["Y_IN"].Data
            reference = np.arctan(test_data)
            output = dut.IOS.Members["Z_OUT"].Data.reshape(-1, 1)
        elif dut.function == trigonometric_function.SINH:
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\sinh(\theta)$")
            ax1.set_title(f"{dut.model} sinh")
            test_data = dut.IOS.Members["Z_IN"].Data
            reference = np.sinh(test_data)
            output = dut.IOS.Members["Y_OUT"].Data.reshape(-1, 1)
        elif dut.function == trigonometric_function.COSH:
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$\cosh(\theta)$")
            ax1.set_title(f"{dut.model} cosh")
            test_data = dut.IOS.Members["Z_IN"].Data
            reference = np.cosh(test_data)
            output = dut.IOS.Members["X_OUT"].Data.reshape(-1, 1)
        elif dut.function == trigonometric_function.ARCTANH:
            ax1.set_xlabel(r"$\theta$")
            ax1.set_ylabel(r"$arctanh(\theta)$")
            ax1.set_title(f"{dut.model} arctanh")
            test_data = dut.IOS.Members["Y_IN"].Data
            reference = np.arctanh(test_data)
            output = dut.IOS.Members["Z_OUT"].Data.reshape(-1, 1)

        error = abs(output - reference)
        ax1.plot(test_data, reference)
        ax1.plot(test_data, output)
        ax2 = ax1.twinx()
        ax2.set_ylabel("|error|")
        ax2.plot(test_data, error, color="red")
        fig.tight_layout()
        plt.draw()
    plt.show()
