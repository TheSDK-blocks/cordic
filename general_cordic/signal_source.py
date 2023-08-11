"""
=============
Signal source 
=============

Signal source to be used fo inverter simulation examples

This class is for simulation methodology purposes only, implemented here to avoid dependencies of the template on external entities.

If you really need dedicated signal generators, it is strongly 
advisable to implement them as dedicated Entities.

See: https://github.com/TheSDK-blocks/signal_generator

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2022.
"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *

import numpy as np

class signal_source(thesdk):
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

                self.IOS.Members['data']=IO() # Pointer for output data
                self.IOS.Members['clk']= IO() # pointer for oputput clock
            
        model : string
            Default 'py' for Python. See documentation of thsdk package for more details.

        length : int
            The length of the data. Default 2**8


        """
        #self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = ['Rs'] # Properties that can be propagated from parent
        self.length=2**8 # Length of the data.

        self.IOS.Members['data'] = IO() # Pointer for clock output
        self.IOS.Members['clk'] = IO() # Pointer for clock output
        self.model = 'py' # Can be set externally, but is not propagated

        # this copies the parameter values from the parent based on self.proplist
        #if len(arg)>=1:
        #    parent=arg[0]
        #    self.copy_propval(parent,self.proplist)
        #    self.parent=parent

        #self.init()

    def init(self):
        """ Method to re-initialize the structure if the attribute values are changed after creation.

        """
        pass #Currently nothing to add

    def main(self):
        ''' Creates the signals and assigns them to output 
        '''
        indata=np.random.randint(2,size=self.length).reshape(-1,1)
        clk=np.array([0 if i%2==0 else 1 for i in range(2*len(indata))]).reshape(-1,1)
        self.IOS.Members['data'].Data = indata 
        self.IOS.Members['clk'].Data = clk 

    def run(self,*arg):
        if self.model=='py':
            self.main()
        else:
            self.print_log(type='E', msg='Model %s not supported' %(self.model))

