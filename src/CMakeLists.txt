# Set the source files to be compiled
SET(TARGET_SRC
    constitutive_equations.cc
    #crystal_data.cc
    #postprocessing.cc 
    #run_time_parameters.cc
    gradient_crystal_plasticity/assembly_data.cc
    #gradient_crystal_plasticity/gradient_crystal_plasticity.cc 
    #gradient_crystal_plasticity/setup.cc
)
ADD_LIBRARY(gCP SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(gCP)