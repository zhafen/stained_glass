#!/usr/bin/env python
'''Non-physical constants for use in analyzing data.
For example, this contains the default fill values for invalid data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import os

########################################################################
# Global Default Parameters
########################################################################
# I do it this way that way I can access these global defaults from plotting
# scripts as well,
# and also have all the most important parameters shown in one place.

########################################################################
# Miscellanious Values
########################################################################

# What fill values to use throughout?
INT_FILL_VALUE = -99999
FLOAT_FILL_VALUE = np.nan

# What integer particle type does each component correspond to?
PTYPE_GAS = 0
PTYPE_DM = 1
PTYPE_LOWRES_DM = 2
PTYPE_STAR = 4

# What colorscheme to use
# COLORSCHEME = p_constants.CLASSIFICATION_COLORS_B

########################################################################
# Simulation Information
########################################################################

# Each simulation has a default resolution run that's analyzed
# (usually it's the highest resolution)
DEFAULT_SIM_RESOLUTIONS = {
    'm10q': 250,
    'm10v': 250,
    'm10y': 250,
    'm10z': 250,
    'm11a': 2100,
    'm11b': 2100,
    'm11q': 7100,
    'm11v': 7100,
    'm11c': 2100,
    'm11d': 7100,
    'm11e': 7100,
    'm11h': 7100,
    'm11i': 7100,
    'm12b': 7100,
    'm12c': 7100,
    'm12f': 7100,
    'm12i': 7100,
    'm12m': 7100,
    'm12r': 7100,
    'm12w': 7100,
    'm12z': 4200,
    'm10q_md': 250,
    'm10v_md': 250,
    'm10y_md': 250,
    'm10z_md': 250,
    'm11a_md': 2100,
    'm11b_md': 2100,
    'm11q_md': 7100,
    'm11v_md': 7100,
    'm11c_md': 2100,
    'm11d_md': 7100,
    'm11e_md': 7100,
    'm11h_md': 7100,
    'm11i_md': 7100,
    'm12b_md': 7100,
    'm12c_md': 7100,
    'm12f_md': 7100,
    'm12i_md': 7100,
    'm12m_md': 7100,
    'm12r_md': 7100,
    'm12w_md': 7100,
    'm12z_md': 4200,
}

# Running AHF (including merger tracing) on a simulation will result in the N
# most massive halos being saved as separate files.
# The most massive halo traced is usually the main halo, but *not always*.
# This dictionary just describes which halo is the main halo.
MAIN_MT_HALO_ID = {
    'm10q': 0,
    'm10v': 2,
    'm10y': 0,
    'm10z': 0,
    'm11a': 0,
    'm11b': 0,
    'm11q': 0,
    'm11v': 0,
    'm11c': 0,
    'm12i': 0,
    'm12f': 0,
    'm12m': 0,
    'm11d_md': 0,
    'm11e_md': 0,
    'm11h_md': 0,
    'm11i_md': 0,
    'm10q_md': 0,
    'm11q_md': 0,
    'm12b_md': 0,
    'm12c_md': 0,
    'm12i_md': 0,
    'm12r_md': 0,
    'm12w_md': 0,
    'm12z_md': 0,
    'm12iF1': 0,
}

# Different zoom-in simulations focus on different halo masses.
# These are the corresponding halo mass bins for a given simulation, where
# m12 == M_h ~ 1e12 M_sun
MASS_BINS = {
    'm10q': 'm10',
    'm10v': 'm10',
    'm10y': 'm10',
    'm10z': 'm10',
    'm11a': 'm10',
    'm11b': 'm10',
    'm11q': 'm11',
    'm11v': 'm11',
    'm11c': 'm11',
    'm12i': 'm12',
    'm12f': 'm12',
    'm12m': 'm12',
    'm10q_md': 'm10',
    'm11i_md': 'm10',
    'm11d_md': 'm11',
    'm11e_md': 'm11',
    'm11h_md': 'm11',
    'm11q_md': 'm11',
    'm12b_md': 'm12',
    'm12c_md': 'm12',
    'm12i_md': 'm12',
    'm12r_md': 'm12',
    'm12w_md': 'm12',
    'm12z_md': 'm12',
    'm12iF1': 'm12',
}

########################################################################
# System Information
########################################################################

ACTIVE_SYSTEM = 'Stampede2'
'''This variable should be the name of the cluster that's currently in use.
Capitilization doesn't matter.
'''

EXAMPLESYSTEM_PARAMETERS = {
    'simulation_data_dir': '/path/to/the/raw/simulation/data',
    'stained_glass_data_dir': '/path/to/overall/dir/for/tracking/output',
    'halo_data_dir': '/path/to/halo/files/location/e.g./AHF/data',
}
'''An example setup. Copy this and change accordingly.
'''

# BIN_PATH = '~/.local/bin'
BIN_PATH = '/work/03057/zhafen/stampede2/miniconda3/bin'
'''Location of executables. First is for default pip install options,
bottom is an example of a Conda setup.
TODO: Improve this further for users.
'''

# These two commands relate to the Jug executables.
JUG_PATH = os.path.join( BIN_PATH, 'jug' )
'''Where the jug executable is located.
In most cases this doesn't need to be changed.
'''
JUG_EXEC_PATH = os.path.join( BIN_PATH, 'jug-execute' )
'''Where the executable for "jug execute" is located.
In most cases this doesn't need to be changed.
'''

########################################################################
# Some custom choices are immediately below.
# You can largely ignore these, but may use them as reference.

QUEST_PARAMETERS = {
    'simulation_data_dir': '/projects/b1026/zhafen',
    'stained_glass_data_dir': '/projects/b1026/zhafen/stained_glass_data',
    'halo_data_dir': '/projects/b1026/zhafen',
}

STAMPEDE2_PARAMETERS = {
    'simulation_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE',
    'stained_glass_data_dir': '/scratch/03057/zhafen/stained_glass_data',
    'halo_data_dir': '/scratch/03057/zhafen',

    'project' : {
        'mock_obs' : {
            'project_dir' : '/home1/03057/zhafen/papers/cluster_sightlines_paper',
            'presentation_dir' : '/work/03057/zhafen/presentation_plots',
            'extras_dir' : '/work/03057/zhafen/extra_plots/mock_obs',
            'output_data_dir' : '/work/03057/zhafen/stampede2/nbs/mock_CGM_observations/data',
        },
        'obs_to_cycle' : {
            'project_dir' : '/home1/03057/zhafen/papers/Observable-to-Cycle-Mapping',
            'presentation_dir' : '/work/03057/zhafen/presentation_plots',
            'extras_dir' : '/work/03057/zhafen/extra_plots/mock_obs',
            'output_data_dir' : '/work/03057/zhafen/stampede2/nbs/mock_CGM_observations/data',
        },
    },
}
