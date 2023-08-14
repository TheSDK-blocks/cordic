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
from BitVector import BitVector

from model_1 import model_1


class rotation_type:
    CIRCULAR = 0
    LINEAR = 1
    HYPERBOLIC = 2


class cordic_mode:
    ROTATION = 0
    VECTORING = 1


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
    def __init__(self, *arg):
        """Inverter parameters and attributes
        Parameters
        ----------
            *arg :
            If any arguments are defined, the first one should be the
            parent instance

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

    def to_fixed_point(self, value, mantissa_bits, fractional_bits):
        # TODO: fractional support
        return (
            BitVector(intVal=value, size=mantissa_bits),
            BitVector(intVal=0, size=fractional_bits),
        )

    def to_double(
        self, bit_vector: (BitVector, BitVector), fractional_bits
    ):
        return bit_vector[0].int_val()

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

        mantissa_bits = 12
        frac_bits = 4
        iterations = 16

        dut = model_1(mantissa_bits, frac_bits, iterations)

        for i in range(0, x_in.size):
            dut.set_inputs(
                self.to_fixed_point(x_in[i][0], mantissa_bits, frac_bits),
                self.to_fixed_point(y_in[i][0], mantissa_bits, frac_bits),
                self.to_fixed_point(z_in[i][0], mantissa_bits, frac_bits),
            )
            dut.run()
            self.IOS.Members["X_OUT"].Data[i] = self.to_double(dut.x_out, frac_bits)
            self.IOS.Members["Y_OUT"].Data[i] = self.to_double(dut.y_out, frac_bits)
            self.IOS.Members["Z_OUT"].Data[i] = self.to_double(dut.z_out, frac_bits)

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

    max_value = 128
    test_data = np.random.randint(2, size=max_value).reshape(-1, 1)
    clk = np.array([0 if i % 2 == 0 else 1 for i in range(2 * len(test_data))]).reshape(
        -1, 1
    )

    for model in models:
        dut = general_cordic()
        dut.model = model
        print("Input:\n")
        print(test_data)
        dut.IOS.Members["X_IN"].Data = test_data
        dut.IOS.Members["Y_IN"].Data = test_data
        dut.IOS.Members["Z_IN"].Data = test_data
        dut.IOS.Members["CLK"] = clk
        duts.append(dut)

    for dut in duts:
        dut.init()
        dut.run()
        print("Output:\n")
        print(dut.IOS.Members["X_OUT"].Data)

    # length=2**8
    # rs=100e6
    # lang='sv'
    # #Testbench vhdl
    # #lang='vhdl'
    # controller=inverter_controller(lang=lang)
    # controller.Rs=rs
    # #controller.reset()
    # #controller.step_time()
    # controller.start_datafeed()
    # #models=['py','sv','icarus', 'ghdl', 'vhdl','eldo','spectre', 'ngspice']
    # #By default, we set only open souce simulators
    # models=['py', 'icarus', 'ghdl', 'ngspice']
    # # Here we instantiate the signal source
    # duts=[]
    # plotters=[]
    # #Here we construct the 'testbench'
    # s_source=signal_source()
    # for model in models:
    #     # Create an inverter
    #     d=inverter()
    #     duts.append(d)
    #     d.model=model
    #     if model == 'ghdl':
    #         d.lang='vhdl'
    #     else:
    #         d.lang=lang
    #     d.Rs=rs
    #     #d.preserve_rtlfiles = True
    #     # Enable debug messages
    #     #d.DEBUG = True
    #     # Run simulations in interactive modes to monitor progress/results
    #     #d.interactive_spice=True
    #     #d.interactive_rtl=True
    #     # Preserve the IO files or simulator files for debugging purposes
    #     #d.preserve_iofiles = True
    #     #d.preserve_spicefiles = True
    #     # Save the entity state after simulation
    #     #d.save_state = True
    #     #d.save_database = True
    #     # Optionally load the state of the most recent simulation
    #     #d.load_state = 'latest'
    #     # This connects the input to the output of the signal source
    #     d.IOS.Members['A']=s_source.IOS.Members['data']
    #     # This connects the clock to the output of the signal source
    #     d.IOS.Members['CLK']=s_source.IOS.Members['clk']
    #     d.IOS.Members['control_write']=controller.IOS.Members['control_write']
    #     ## Add plotters
    #     p=signal_plotter()
    #     plotters.append(p)
    #     p.plotmodel=d.model
    #     p.plotvdd=d.vdd
    #     p.Rs = rs
    #     p.IOS.Members['A']=d.IOS.Members['A']
    #     p.IOS.Members['Z']=d.IOS.Members['Z']
    #     p.IOS.Members['A_OUT']=d.IOS.Members['A_OUT']
    #     p.IOS.Members['A_DIG']=d.IOS.Members['A_DIG']
    #     p.IOS.Members['Z_ANA']=d.IOS.Members['Z_ANA']
    #     p.IOS.Members['Z_RISE']=d.IOS.Members['Z_RISE']
    #

    # # Here we run the instances
    # s_source.run() # Creates the data to the output
    # for d in duts:
    #     d.init()
    #     d.run()
    # for p in plotters:
    #     p.init()
    #     p.run()

    #  #This is here to keep the images visible
    #  #For batch execution, you should comment the following line
    # if args.show:
    #    input()
    # #This is to have exit status for succesfuulexecution
    # sys.exit(0)
