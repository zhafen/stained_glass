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

    def get_sim_subdir(
        self,
        sim_name,
        physics = None,
        resolution = None,
        subsubdir = None,
    ):

        if physics is None:
            name_mapping = {
                '': 'core',
                '_md': 'metal_diffusion',
            }
            physics = name_mapping[sim_name[4:]]

        if resolution is None:
            resolution = sg_config.DEFAULT_SIM_RESOLUTIONS[sim_name]

        sim_subdir = os.path.join(
            physics,
            '{}_res{}'.format(
                sim_name[:4],
                resolution,
            )
        )

        if subsubdir is not None:
            sim_subdir = os.path.join(
                sim_subdir,
                subsubdir,
            )

        return sim_subdir

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

    def get_sg_filepath(
        self,
        sim_name,
        snum,
        data_type = 'ray',
        extension = 'h5',
        i = 'first_available',
        **kwargs
    ):

        full_data_dir = self.get_stained_glass_dir(
            sim_name,
            snum = snum,
            **kwargs
        )

        # Get the filename, except for the ID
        if i is None:
            id_str = ''
        else:
            id_str = '_{:03d}'
        empty_filename = '{}_snum{}{}.{}'.format(
            data_type,
            snum,
            id_str,
            extension
        )
        empty_filepath = os.path.join( full_data_dir, empty_filename )

        # Get the file ID
        if i == 'first_available':
            i = 0
            while True:
                
                filepath = empty_filepath.format( i )

                if not os.path.isfile( filepath ):
                    break
                
                i += 1
        else:
            filepath = empty_filepath.format( i )

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

    def get_stained_glass_dir(
        self,
        sim_name,
        subdir = '',
        snum = None,
        makedirs = False,
        **kwargs
    ):
        
        sg_dir = os.path.join(
            self.system_parameters['stained_glass_data_dir'],
            self.get_sim_subdir( sim_name, **kwargs ),
            subdir,
        )

        if snum is not None:
            sg_dir = os.path.join(
                sg_dir,
                'snum{:03d}'.format( snum ),
            )

        if makedirs:
            os.makedirs( sg_dir, exist_ok=True )

        return sg_dir

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

