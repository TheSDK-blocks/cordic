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
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
from rtl import *
from spice import *

import numpy as np

class general_cordic(rtl,spice,thesdk):

    def __init__(self,*arg): 
        """ Inverter parameters and attributes
            Parameters
            ----------
                *arg : 
                If any arguments are defined, the first one should be the parent instance

            Attributes
            ----------
            proplist : array_like
                List of strings containing the names of attributes whose values are to be copied 
                from the parent

            Rs : float
                Sampling rate [Hz] of which the input values are assumed to change. Default: 100.0e6

            vdd : float
                Supply voltage [V] for inverter analog simulation. Default 1.0.

            IOS : Bundle
                Members of this bundle are the IO's of the entity. See documentation of thsdk package.
                Default members defined as

                self.IOS.Members['A']=IO() # Pointer for input data
                self.IOS.Members['Z']= IO() # pointer for oputput data
                self.IOS.Members['control_write']= IO() # Piter for control IO for rtl simulations

            model : string
                Default 'py' for Python. See documentation of thsdk package for more details.

        """
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = ['Rs', 'vdd'] # Properties that can be propagated from parent
        self.Rs = 100e6 # Sampling frequency
        self.vdd = 1.0
        self.IOS.Members['A'] = IO() # Pointer for input data
        self.IOS.Members['Z'] = IO()
        self.IOS.Members['CLK'] = IO() # Test clock for spice simulations
        self.IOS.Members['A_OUT'] = IO() # Test output for the input A
        ##Analog output for inverter for analog simulation
        self.IOS.Members['Z_ANA'] = IO()
        ## For Extracting rising edges from the output waveform
        self.IOS.Members['Z_RISE'] = IO()
        ## Extracting values of A and Z at falling edges of CLK in decimal format (integer, in this case 0 or 1)
        ## The clock signal can be any node voltage in the simulation
        self.IOS.Members['A_DIG'] = IO()

        self.IOS.Members['control_write'] = IO() # File for control is created in controller
        self.model = 'py' # Can be set externally, but is not propagated

        # this copies the parameter values from the parent based on self.proplist
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent=parent

        self.init()

    def init(self):
        """ Method to re-initialize the structure if the attribute values are changed after creation.

        """
        pass #Currently nothing to add

    def main(self):
        ''' The main python description of the operation. Contents fully up to designer, however, the 
        IO's should be handled bu following this guideline:
        
        To isolate the internal processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        inval=self.IOS.Members['A'].Data
        out=np.array(1-inval)
        if self.par:
            self.queue.put(out)
        self.IOS.Members['Z'].Data=out

    def run(self,*arg):
        ''' The default name of the method to be executed. This means: parameters and attributes 
            control what is executed if run method is executed. By this we aim to avoid the need of 
            documenting what is the execution method. It is always self.run. 

            Parameters
            ----------
            *arg :
                The first argument is assumed to be the queue for the parallel processing defined in the parent, 
                and it is assigned to self.queue and self.par is set to True. 
        
        '''
        if self.model=='py':
            self.main()
        else: 
            # This defines contents of modelsim control file executed when interactive_rtl = True
            # Interactive control files
            if self.model == 'icarus' or self.model == 'ghdl':
                self.interactive_control_contents="""
                    set io_facs [list] 
                    lappend io_facs "tb_inverter.A"
                    lappend io_facs "tb_inverter.Z" 
                    lappend io_facs "tb_inverter.clock"
                    gtkwave::addSignalsFromList $io_facs 
                    gtkwave::/Time/Zoom/Zoom_Full
                """
            else:
                self.interactive_control_contents="""
                    add wave \\
                    sim/:tb_inverter:A \\
                    sim/:tb_inverter:initdone \\
                    sim/:tb_inverter:clock \\
                    sim/:tb_inverter:Z
                    run -all
                    wave zoom full
                """

            if self.model == 'ghdl':
                # With this structure you can control the signals to be dumped to VCD 
                #pass
                self.simulator_control_contents=("version = 1.1  # Optional\n"
                + "/tb_inverter/A\n"
                + "/tb_inverter/Z\n"
                + "/tb_inverter/clock\n"
                                                 )

            if self.model in ['sv', 'icarus']:
                # Verilog simulation options here
                _=rtl_iofile(self, name='A', dir='in', iotype='sample', ionames=['A'], datatype='sint') # IO file for input A
                f=rtl_iofile(self, name='Z', dir='out', iotype='sample', ionames=['Z'], datatype='sint')
                # This is to avoid sampling time confusion with Icarus
                if self.lang == 'sv':
                    f.rtl_io_sync='@(negedge clock)'
                elif self.lang == 'vhdl':
                    f.rtl_io_sync='falling_edge(clock)'

                self.rtlparameters=dict([ ('g_Rs',('real',self.Rs)),]) # Defines the sample rate
                self.run_rtl()
                self.IOS.Members['Z'].Data=self.IOS.Members['Z'].Data[:,0].astype(int).reshape(-1,1)
            elif self.model=='vhdl' or self.model == 'ghdl':
                # VHDL simulation options here
                _=rtl_iofile(self, name='A', dir='in', iotype='sample', ionames=['A']) # IO file for input A
                f=rtl_iofile(self, name='Z', dir='out', iotype='sample', ionames=['Z'], datatype='int')
                if self.lang == 'sv':
                    f.rtl_io_sync='@(negedge clock)'
                elif self.lang == 'vhdl':
                    f.rtl_io_sync='falling_edge(clock)'
                self.rtlparameters=dict([ ('g_Rs',('real',self.Rs)),]) # Defines the sample rate
                self.run_rtl()
                self.IOS.Members['Z'].Data=self.IOS.Members['Z'].Data.astype(int).reshape(-1,1)
            elif self.model in ['eldo','spectre','ngspice']:

                # Creating a clock signal, which is used for testing the sample output features
                _=spice_iofile(self, name='CLK', dir='in', iotype='sample', ionames='CLK', rs=2*self.Rs, \
                               vhi=self.vdd, trise=1/(self.Rs*8), tfall=1/(self.Rs*8))
                # Sample type input
                _=spice_iofile(self, name='A', dir='in', iotype='sample', ionames='A', rs=self.Rs, \
                               vhi=self.vdd, trise=1/(self.Rs*4), tfall=1/(self.Rs*4))

                # These are helper IOS for analog simulation
                _=spice_iofile(self, name='Z_ANA', dir='out', iotype='event', sourcetype='V', ionames='Z')
                
                # Sample type output
                # Clock is used to sample the waveform in analog simulation
                _=spice_iofile(self, name='Z', dir='out', iotype='sample', ionames='Z', trigger='CLK', \
                               vth=self.vdd/2,edgetype='rising',ioformat='dec')
                

                # Saving the analog waveform of the input as well
                _=spice_iofile(self, name='A_OUT', dir='out', iotype='event', sourcetype='V', ionames='A')

                # For Extracting rising edges from the output waveform
                _=spice_iofile(self, name='Z_RISE', dir='out', iotype='time', sourcetype='V', ionames='Z', \
                               edgetype='rising',vth=self.vdd/2)


                ## Extracting values of A and Z at falling edges of CLK in decimal format (integer, in this case 0 or 1)
                ## The clock signal can be any node voltage in the simulation
                _=spice_iofile(self, name='A_DIG', dir='out', iotype='sample', ionames='A', trigger='CLK', \
                               vth=self.vdd/2,edgetype='rising',ioformat='dec')

                # Multithreading, options and parameters
                self.nproc = 2
                self.spiceoptions = {
                            'eps': '1e-6'
                        }
                self.spiceparameters = {
                            'exampleparam': '0'
                        }

                # Defining library options
                # Path to model libraries needs to be defined in TheSDK.config as
                # either ELDOLIBFILE or SPECTRELIBFILE. In this case, no model libraries
                # will be included (assuming these variables are not defined). The
                # temperature will be set regardless.
                self.spicecorner = {
                            'corner': 'top_tt',
                            'temp': 27,
                        }

                # Example of defining supplies (not used here because the example inverter has no supplies)
                _=spice_dcsource(self,name='supply',value=self.vdd,pos='VDD',neg='VSS',extract=True)
                _=spice_dcsource(self,name='ground',value=0,pos='VSS',neg='0')

                # Adding a resistor between VDD and VSS to demonstrate power consumption extraction
                # This also demonstrates how to inject manual commands in to the testbench
                if self.model=='spectre':
                    self.spicemisc.append('simulator lang=spice')
                self.spicemisc.append('Rtest VDD VSS 2000')
                if self.model=='spectre':
                    self.spicemisc.append('simulator lang=spectre')
                
                # Plotting nodes for interactive waveform viewing.
                # Spectre also supported, but without 'v()' specifiers.
                # i.e. plotlist = ['A','Z']
                if self.model == 'eldo':
                    plotlist = ['v(A)','v(Z)']
                elif self.model == 'spectre':
                    plotlist = ['A','Z']
                else:
                    plotlist = []

                # Simulation command
                _=spice_simcmd(self,sim='tran',plotlist=plotlist)
                self.run_spice()

            if self.par:
                self.queue.put(self.IOS.Members)

    def define_io_conditions(self):
        '''This overloads the method called by run_rtl method. It defines the read/write conditions for the files

        '''
        if self.lang == 'sv':
            # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members['A'].rtl_io_condition='initdone'
            # Output is read to verilog simulation when all of the outputs are valid, 
            # and after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members['Z'].rtl_io_condition_append(cond='&& initdone')
        elif self.lang == 'vhdl':
            self.iofile_bundle.Members['A'].rtl_io_condition='(initdone = \'1\')'
            # Output is read to verilog simulation when all of the outputs are valid, 
            # and after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members['Z'].rtl_io_condition_append(cond='and initdone = \'1\'')

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    # from inverter import *
    # from inverter.controller import controller as inverter_controller
    # from inverter.signal_source import signal_source
    # from inverter.signal_plotter import signal_plotter
    # import pdb

    # # Implement argument parser
    # parser = argparse.ArgumentParser(description='Parse selectors')
    # parser.add_argument('--show', dest='show', type=bool, nargs='?', const = True, 
    #         default=False,help='Show figures on screen')
    # args=parser.parse_args()

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

