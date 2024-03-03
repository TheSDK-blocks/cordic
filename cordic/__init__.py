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
import yaml

# from model_1 import model_1
from cordic.model_3 import model_3
#from BitVector import BitVector
import cordic.cordic_common.methods as methods
import cordic.cordic_common.cordic_types as cordic_types


class cordic(rtl, spice, thesdk):
    """Cordic parameters and attributes
    Parameters
    ----------
        *arg :
            If any arguments are defined, the first one should be the
            parent instance

        config_file : str
            Path to CORDIC config file
        
        model : str
            model (py or sv)

    """
    def __init__(
        self,
        *arg,
        config_file,
        model
    ):
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

        self.IOS.Members['control_write']= IO() 
        self.IOS.Members["clock"] = IO()
        self.IOS.Members["reset"] = IO()

        self.model = model

        self.Rs = 100e6
        self.lang = 'sv'
        self.vlogext = '.v'

        with open(config_file, 'r') as cfile:
            cordic_config = yaml.safe_load(cfile)
            self.mb = cordic_config['mantissa-bits']
            self.fb = cordic_config['fraction-bits']
            self.iters = cordic_config['iterations']
            self.repr = cordic_config.get('number-repr', "fixed-point")
            self.enable_circular = cordic_config.get('enable-circular', False)
            self.enable_hyperbolic = cordic_config.get('enable-hyperbolic', False)
            self.enable_rotational = cordic_config.get('enable-rotational', False)
            self.enable_vectoring = cordic_config.get('enable-vectoring', False)
            self.preprocessor_class = cordic_config.get('preprocessor-class', "Basic")
            self.postprocessor_class = cordic_config.get('postprocessor-class', "Basic")
            self.use_phase_accum = None
            self.phase_accum_width = None
            if cordic_config.get('up-convert-config'):
                self.use_phase_accum = cordic_config["up-convert-config"]["use-phase-accum"]
                self.phase_accum_width = cordic_config["up-convert-config"]["phase-accum-width"]

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

        dut = model_3(self.mb,
                      self.fb,
                      self.iters,
                      self.repr,
                      self.preprocessor_class,
                      self.postprocessor_class,
                      self.use_phase_accum,
                      self.phase_accum_width)
        for i in range(0, d_in.size):
            dut.d_in   = methods.to_fixed_point(d_in[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            dut.rs1_in = methods.to_fixed_point(rs1[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            dut.rs2_in = methods.to_fixed_point(rs2[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            dut.rs3_in = methods.to_fixed_point(rs3[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            dut.op = ops[i]
            dut.run()
            d_out[i] = methods.to_double_single(dut.d_out, self.mb, self.fb, self.repr)
            rs1_out[i] = methods.to_double_single(dut.rs1_out, self.mb, self.fb, self.repr)
            rs2_out[i] = methods.to_double_single(dut.rs2_out, self.mb, self.fb, self.repr)
            rs3_out[i] = methods.to_double_single(dut.rs3_out, self.mb, self.fb, self.repr)

        self.IOS.Members["io_out_bits_dOut"].Data = d_out.reshape(-1, 1)
        self.IOS.Members["io_out_bits_cordic_x"].Data = rs1_out.reshape(-1, 1)
        self.IOS.Members["io_out_bits_cordic_y"].Data = rs2_out.reshape(-1, 1)
        self.IOS.Members["io_out_bits_cordic_z"].Data = rs3_out.reshape(-1, 1)
        self.check_for_overflow()


    def control_string_to_int(self, string):
        """
        Used with trig config.
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
        else:
            return string

    def calc_control_word(self, Rs, Fc):
        """
        Parameters
        ----------
        Rs : int
            Sampling rate
        Fc : int
            Center frequency

        Returns
        -------
        control_word : int
            Integer control word to achieve given Fc
        """
        phase_accum_width = self.phase_accum_width
        return 2**phase_accum_width * Fc // Rs


    def convert_inputs(self):
        # TODO: restructure and replace with inlist
        new_arr = np.empty(len(self.IOS.Members["io_in_bits_rs1"].Data), dtype="int32")
        for i in range(0, len(self.IOS.Members["io_in_bits_rs1"].Data)):
            len32 = methods.to_fixed_point(self.IOS.Members["io_in_bits_rs1"].Data[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            len32_bin = np.binary_repr(len32, width=32)
            wanted_bits = len32_bin[0:self.mb+self.fb]
            new_arr[i] = \
                np.int32(len32)
        self.IOS.Members["io_in_bits_rs1"].Data = new_arr.reshape(-1, 1)
        new_arr = np.empty(len(self.IOS.Members["io_in_bits_rs2"].Data), dtype="int32")
        for i in range(0, len(self.IOS.Members["io_in_bits_rs2"].Data)):
            len32 = methods.to_fixed_point(self.IOS.Members["io_in_bits_rs2"].Data[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            len32_bin = np.binary_repr(len32, width=32)
            wanted_bits = len32_bin[0:self.mb+self.fb]
            new_arr[i] = \
                np.int32(len32)
        self.IOS.Members["io_in_bits_rs2"].Data = new_arr.reshape(-1, 1)
        new_arr = np.empty(len(self.IOS.Members["io_in_bits_rs3"].Data), dtype="int32")
        for i in range(0, len(self.IOS.Members["io_in_bits_rs3"].Data)):
            len32 = methods.to_fixed_point(self.IOS.Members["io_in_bits_rs3"].Data[i][0], self.mb, self.fb, self.repr, ret_type="numpy")
            len32_bin = np.binary_repr(len32, width=32)
            wanted_bits = len32_bin[0:self.mb+self.fb]
            new_arr[i] = \
                np.int32(len32)
        self.IOS.Members["io_in_bits_rs3"].Data = new_arr.reshape(-1, 1)
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
                # Shift to 32 bit
                data = ios.Data.astype('int32')[i][0]
                new_arr[i] = methods.to_double_single(
                    data, 
                    self.mb, self.fb, self.repr)
            ios.Data = new_arr.reshape(-1, 1)


    def check_for_overflow(self):
        max_val = 2**(self.mb + self.fb - 1) - 1
        min_val = -2**(self.mb + self.fb - 1)
        for val in [self.IOS.Members["io_out_bits_cordic_x"].Data,
                    self.IOS.Members["io_out_bits_cordic_y"].Data,
                    self.IOS.Members["io_out_bits_cordic_z"].Data,
                    self.IOS.Members["io_out_bits_dOut"].Data,]:
            for elem in val:
                if (elem > max_val ) or (elem < min_val):
                    self.print_log(type="W", msg=f"Overflow detected: {val}")

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
            _=rtl_iofile(self, name='io_in_valid', dir='in', iotype='sample', datatype='sint', ionames=['io_in_valid'])
            _=rtl_iofile(self, name='io_in_bits_rs1', dir='in', iotype='sample', datatype='sint', ionames=['io_in_bits_rs1'])
            _=rtl_iofile(self, name='io_in_bits_rs2', dir='in', iotype='sample', datatype='sint', ionames=['io_in_bits_rs2'])
            _=rtl_iofile(self, name='io_in_bits_rs3', dir='in', iotype='sample', datatype='sint', ionames=['io_in_bits_rs3'])
            _=rtl_iofile(self, name='io_in_bits_control', dir='in', iotype='sample', datatype='int', ionames=['io_in_bits_control'])

            _=rtl_iofile(self, name='io_out_bits_dOut', dir='out', iotype='sample', datatype='sint', ionames=['io_out_bits_dOut'])
            _=rtl_iofile(self, name='io_out_bits_cordic_x', dir='out', iotype='sample', datatype='int', ionames=['io_out_bits_cordic_x'])
            _=rtl_iofile(self, name='io_out_bits_cordic_y', dir='out', iotype='sample', datatype='int', ionames=['io_out_bits_cordic_y'])
            _=rtl_iofile(self, name='io_out_bits_cordic_z', dir='out', iotype='sample', datatype='int', ionames=['io_out_bits_cordic_z'])
            _=rtl_iofile(self, name='io_out_valid', dir='out', iotype='sample', datatype='int', ionames=['io_out_valid'])

            self.rtlparameters=dict([ ('g_Rs', (float, self.Rs)), ]) #Freq for sim clock

            self.run_rtl()
            self.convert_outputs()

    def define_io_conditions(self):
        self.iofile_bundle.Members["io_in_valid"].rtl_io_condition='initdone'
        self.iofile_bundle.Members["io_in_bits_rs1"].rtl_io_condition='initdone'
        self.iofile_bundle.Members["io_in_bits_rs2"].rtl_io_condition='initdone'
        self.iofile_bundle.Members["io_in_bits_rs3"].rtl_io_condition='initdone'
        self.iofile_bundle.Members["io_in_bits_control"].rtl_io_condition='initdone'
        self.iofile_bundle.Members["io_out_bits_dOut"].rtl_io_condition='io_out_valid'
        self.iofile_bundle.Members["io_out_bits_cordic_x"].rtl_io_condition='io_out_valid'
        self.iofile_bundle.Members["io_out_bits_cordic_y"].rtl_io_condition='io_out_valid'
        self.iofile_bundle.Members["io_out_bits_cordic_z"].rtl_io_condition='io_out_valid'
        self.iofile_bundle.Members["io_out_valid"].rtl_io_condition='initdone'

    def run_cocotb(self):
        self.convert_inputs()
        sim = os.getenv("SIM", "icarus") # use verilator for faster simulation (version > 5)
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
            "io_out_bits_dOut": "io_out_bits_dOut",
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
        if sim == "verilator":
            build_args = ["--trace"]
        else:
            build_args = []

        runner.build(
            verilog_sources=[
                self.vlogsrcpath + "/cordic.v",
                self.vlogsrcpath + "/cocotb_iverilog_dump.v",
            ],
            hdl_toplevel="cordic",
            always=True,
            build_args=build_args
        )
        runner.test(
            hdl_toplevel="cordic",
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
            iofile.read(dtype="int32")
            self.IOS.Members[out_ionames[i]].Data = iofile.Data
        self.convert_outputs()

if __name__ == "__main__":
    """Quick and dirty self test"""
    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
    from cordic.controller import controller
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str
    )
    args = parser.parse_args()
    dut = cordic(config_file=args.config_file, model="sv")
    dut.preserve_iofiles = True
    dut.preserve_rtlfiles = True
    dut.interactive_rtl = False
    dut.repr = "fixed-point"
    function = "Sine"
    test_data = \
        np.arange(-np.pi, np.pi, 0.01, dtype=float).reshape(-1, 1)
    size = np.size(test_data)

    cordic_controller = controller()
    cordic_controller.Rs = dut.Rs
    cordic_controller.reset()
    cordic_controller.step_time()
    cordic_controller.start_datafeed()
    cordic_controller.step_time(step=20000000)
    cordic_controller.set_simdone()

    if dut.model == "sv":
        dut.print_log(type="I", msg="Note: this test requires building CORDIC with trig config.")
    dut.IOS.Members["io_in_bits_rs1"].Data = np.copy(test_data)
    dut.IOS.Members["io_in_bits_rs2"].Data = np.zeros(size, dtype=np.int32).reshape(-1, 1)
    dut.IOS.Members["io_in_bits_rs3"].Data = np.zeros(size, dtype=np.int32).reshape(-1, 1)
    dut.IOS.Members["io_in_bits_control"].Data = np.full(
        dut.IOS.Members["io_in_bits_rs1"].Data.size, dut.control_string_to_int(function)
    ).reshape(-1, 1)
    dut.IOS.Members["io_in_valid"].Data = np.ones(size+1, dtype=np.int32).reshape(-1, 1)
    dut.IOS.Members["io_in_valid"].Data[-1] = 0
    dut.IOS.Members["control_write"] = cordic_controller.IOS.Members["control_write"]
    dut.run()
    output = np.array(
        [
            data_point for data_point in dut.IOS.Members["io_out_bits_dOut"].Data[:, 0]
        ]
    ).reshape(-1, 1)
    fig, ax = plt.subplots()
    bits_info = f" mb={dut.mb}, fb={dut.fb}"
    ax.set_title(f"{dut.model} {function}" + bits_info)
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

