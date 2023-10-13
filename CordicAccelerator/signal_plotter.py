"""
==============
Signal plotter 
==============

Signal plotter to be used fo inverter simulation examples

This class is for simulation methodology purposes only, 
implemented here to avoid dependencies of the template on 
external entities.

If you really need dedicated signal generators, it is strongly 
advisable to implement them as dedicated Entities.

See: https://github.com/TheSDK-blocks/signal_analyzer

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2022.
"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *

import numpy as np
import matplotlib.pyplot as plt
import pdb

class signal_plotter(thesdk):
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        """ Signal source parameters and attributes
            Parameters
            ----------
                *arg : 
                If any arguments are defined, the first one should be the parent instance

            Attributes
            ----------
            proplist : array_like
                List of strings containing the names of attributes whose values are to be copied 
                from the parent

            IOS : Bundle
                Members of this bundle are the IO's of the entity. See documentation of thesdk package.
                Default members defined as

                self.IOS.Members['A'] = IO() 
                self.IOS.Members['A_OUT'] = IO() 
                self.IOS.Members['A_DIG'] = IO() 
                self.IOS.Members['Z_ANA'] = IO() 
                self.IOS.Members['Z_RISE'] = IO() 
            
        plotmodel : string
            Default 'py' for Python.

        plotprefix : string
            Default  ''
            Plotfiles are named as inv_<plotprefix><plotmodel>.eps

        plotvdd : float
            Supply voltage of analog signals. Defines the range for plots


        """
        #self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = ['Rs'] # Properties that can be propagated from parent
        self.Rs = 100e6

        self.IOS.Members['A'] = IO() 
        self.IOS.Members['A_OUT'] = IO() 
        self.IOS.Members['A_DIG'] = IO() 
        self.IOS.Members['Z_ANA'] = IO() 
        self.IOS.Members['Z_RISE'] = IO() 
        self.model = 'py'
        self.plotmodel = 'py'
        self.plotprefix = ''
        self.plotvdd = 1.0

        # this copies the parameter values from the parent based on self.proplist
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent=parent

        #self.init()

    def init(self):
        """ Method to re-initialize the structure if the attribute values are changed after creation.

        """
        pass #Currently nothing to add

    def main(self):
        ''' Creates the plots 
        '''
        hfont = {'fontname':'Sans'}
        nsamp = 20
        x = np.arange(nsamp).reshape(-1,1)
        if self.plotmodel == 'eldo' or self.plotmodel=='spectre' or self.plotmodel=='ngspice':
            figure,axes = plt.subplots(2,2,sharex='col',tight_layout=True)
            axes[0,0].stem(x,self.IOS.Members['A_DIG'].Data[:nsamp,0])
            axes[0,0].set_ylabel('Input', **hfont,fontsize=18)
            axes[0,0].grid(True)
            axes[1,0].stem(x,self.IOS.Members['Z'].Data[:nsamp,0])
            axes[1,0].set_xlim(0,nsamp-1)
            axes[1,0].set_ylabel('Output', **hfont,fontsize=18)
            axes[1,0].set_xlabel('Sample', **hfont,fontsize=18)
            axes[1,0].grid(True)
            axes[0,1].plot(self.IOS.Members['A_OUT'].Data[:,0],self.IOS.Members['A_OUT'].Data[:,1],label='Input')
            axes[0,1].grid(True)
            axes[1,1].plot(self.IOS.Members['Z_ANA'].Data[:,0],self.IOS.Members['Z_ANA'].Data[:,1],label='Output')
            axes[1,1].plot(self.IOS.Members['Z_RISE'].Data[:,0],np.ones(self.IOS.Members['Z_RISE'].Data[:,0].shape)*self.plotvdd/2,\
                           ls='None',marker='o',label='Rising edges')
            axes[1,1].set_xlabel('Time (s)', **hfont,fontsize=18)
            axes[1,1].set_xlim(0,(nsamp-1)/self.Rs)
            axes[1,1].grid(True)
        else:
            if self.plotmodel in [ 'placeholder' ]:
                latency=1
            else:
                latency=0
            figure,axes=plt.subplots(2,1,sharex=True)
            axes[0].stem(x,self.IOS.Members['A'].Data[:nsamp,0])
            axes[0].set_ylim(0, 1.1)
            axes[0].set_xlim((np.amin(x), np.amax(x)))
            axes[0].set_ylabel('Input', **hfont,fontsize=18)
            axes[0].grid(True)
            axes[1].stem(x,self.IOS.Members['Z'].Data[latency:nsamp+latency,0])
            axes[1].set_ylim(0, 1.1)
            axes[1].set_xlim((np.amin(x), np.amax(x)))
            axes[1].set_ylabel('Output', **hfont,fontsize=18)
            axes[1].set_xlabel('Sample (n)', **hfont,fontsize=18)
            axes[1].grid(True)
        titlestr = "Inverter model %s" %(self.plotmodel) 
        plt.suptitle(titlestr,fontsize=20)
        plt.grid(True)
        plt.show(block=False)
        printstr="../inv_%s%s.eps" %(self.plotprefix,self.plotmodel)
        figure.savefig(printstr, format='eps', dpi=300)

    def run(self,*arg):
        if self.model=='py':
            self.main()
        else:
            self.print_log(type='E', msg='Model %s not supported' %(self.model))

