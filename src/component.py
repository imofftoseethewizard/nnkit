'''
This module contains a single tiny class -- LayerComponent -- which abstracts containment within a
Layer instance.  Typically, LayerComponents (Dendrite, Synapse, etc) are provided to the Layer
constructor, which then attaches itself to them, allowing subsequent method calls without
requiring the containing layer as an argument.
'''

__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

class LayerComponent(object):
    '''
    This is an abstract base class for objects that will be contained within a Layer instance.
    Dendrite, Synapse, UpdateRule, and Objective are LayerComponent subclasses.
    '''
    def attach(self, layer):
        '''
        This method is typically called by one of the Layer constructors, which see.
        '''
        self.layer = layer
        return self


    def get_parameters(self):
        '''
        Returns a dictionary which contains a selection of the component's properties.  It is not
        intended to be used to support persistance, but as a source of diagnostic information.
        '''
        return { '__class__':  self.__class__.__name__ }


