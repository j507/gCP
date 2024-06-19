# Set the source files to be compiled
SET(TARGET_SRC
    boundary_conditions.cc
    constitutive_laws.cc
    crystal_data.cc
    fe_field.cc
    line_search.cc
    postprocessing.cc
    run_time_parameters.cc
    utilities.cc
    gradient_crystal_plasticity/active_set.cc
    gradient_crystal_plasticity/assembly.cc
    gradient_crystal_plasticity/assembly_data.cc
    gradient_crystal_plasticity/gradient_crystal_plasticity_solver.cc
    gradient_crystal_plasticity/quadrature_point_history.cc
    gradient_crystal_plasticity/setup.cc
    gradient_crystal_plasticity/solve.cc
)
ADD_LIBRARY(gCP SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(gCP)