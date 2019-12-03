#!/usr/bin/env python
'''Simple functions and variables for easily accessing common files and choices
of parameters.
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

        if resolution is None:
            resolution = sg_config.DEFAULT_SIM_RESOLUTIONS[sim_name]

        return os.path.join(
            physics,
            '{}_res{}'.format(
                sim_name[:4],
                resolution,
            )
        )

    ########################################################################

    def get_sim_dir( self, sim_name, **kwargs ):

        return os.path.join(
            self.system_parameters['simulation_data_dir'],
            self.get_sim_subdir( sim_name, **kwargs ),
            'output',
        )

    ########################################################################

    def get_yt_filepath( self, sim_name, snum, **kwargs ):

        formatted_snum = '{:03d}'.format( snum )

        # Assume there are subdirectories
        filepath = os.path.join(
            self.get_sim_dir( sim_name, **kwargs ),
            'snapdir_{}'.format( formatted_snum ),
            'snapshot_{}.0.hdf5'.format( formatted_snum )
        )

        # If there are not subdirectories change the filepath
        if not os.path.isfile( filepath ):
            filepath = os.path.join(
                self.get_sim_dir( sim_name, **kwargs ),
                'snapshot_{}.hdf5'.format( formatted_snum )
            )

        return filepath

    ########################################################################

    def get_ray_filepath( self, sim_name, snum, **kwargs ):

        data_dir = self.get_stained_glass_dir( sim_name, **kwargs )

        # Choose the next empty ray name
        empty_filename = 'ray_{}.h5'
        i = 0
        while True:
            
            filename = empty_filename.format( i )
            filepath = os.path.join( data_dir, filename )

            if not os.path.isfile( filepath ):
                break
            
            i += 1

        return filepath

    ########################################################################

    def get_metafile_dir( self, sim_name, **kwargs ):

        return os.path.join(
            self.system_parameters['simulation_data_dir'],
            self.get_sim_subdir( sim_name, **kwargs ),
        )

    ########################################################################

    def get_halo_dir( self, sim_name, **kwargs ):

        return os.path.join(
            self.system_parameters['halo_data_dir'],
            self.get_sim_subdir( sim_name, **kwargs ),
            'halo',
        )

    ########################################################################

    def get_stained_glass_dir( self, sim_name, subdir='data', **kwargs ):
        
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

