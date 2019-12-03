#!/usr/bin/env python
'''Simple functions and variables for easily accessing common files and choices
of parameters.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import os

import stained_glass.config as sg_config

########################################################################
########################################################################


class FileManager( object ):

    def __init__( self, project=None ):
        self.project = project

        parameters_name = '{}_PARAMETERS'.format(
            sg_config.ACTIVE_SYSTEM.upper()
        )
        self.system_parameters = getattr( sg_config, parameters_name )

        if project is not None:
            self.project_parameters = \
                self.system_parameters['project'][self.project]

    ########################################################################
    ########################################################################

    def get_sim_subdir( self, sim_name, physics=None, resolution=None, ):

        if physics is None:
            name_mapping = {
                '': 'core',
                '_md': 'metal_diffusion',
            }
            physics = name_mapping[sim_name[4:]]

        return os.path.join(
            physics,
            '{}_res{}'.format(
                sim_name[:4],
                sg_config.DEFAULT_SIM_RESOLUTIONS[sim_name],
            )
        )

    ########################################################################

    def get_sim_dir( self, sim_name ):

        return os.path.join(
            self.system_parameters['simulation_data_dir'],
            self.get_sim_subdir( sim_name ),
            'output',
        )

    ########################################################################

    def get_metafile_dir( self, sim_name ):

        return os.path.join(
            self.system_parameters['simulation_data_dir'],
            self.get_sim_subdir( sim_name ),
        )

    ########################################################################

    def get_halo_dir( self, sim_name ):

        return os.path.join(
            self.system_parameters['halo_data_dir'],
            self.get_sim_subdir( sim_name ),
            'halo',
        )

    ########################################################################

    def get_stained_glass_dir( self, sim_name, subdir='data', ):
        
        return os.path.join(
            self.system_parameters['stained_glass_data_dir'],
            self.get_sim_subdir( sim_name ),
            subdir,
        )

    ########################################################################

    def get_project_figure_dir( self ):

        return os.path.join(
            self.project_parameters['project_dir'],
            'figures',
        )

    ########################################################################

    def get_project_presentation_dir( self ):

        return self.project_parameters['presentation_dir']

    ########################################################################
    ########################################################################

