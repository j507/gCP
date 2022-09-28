#include <gCP/run_time_parameters.h>

#include <deal.II/base/conditional_ostream.h>

#include <fstream>

#ifndef __has_include
  static_assert(false, "__has_include not supported");
#else
#  if __cplusplus >= 201703L && __has_include(<filesystem>)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
     namespace fs = boost::filesystem;
#  endif
#endif

namespace gCP
{



namespace RunTimeParameters
{



HookeLawParameters::HookeLawParameters()
:
C1111(134615),
C1122(57692.3),
C1212(38461.5)
{}



void HookeLawParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Hooke-Law's parameters");
  {
    prm.declare_entry("C1111",
                      "134615.0",
                      dealii::Patterns::Double());

    prm.declare_entry("C1122",
                      "57692.3",
                      dealii::Patterns::Double());

    prm.declare_entry("C1212",
                      "38461.5",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();
}



void HookeLawParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Hooke-Law's parameters");
  {
    C1111 = prm.get_double("C1111");
    C1122 = prm.get_double("C1122");
    C1212 = prm.get_double("C1212");

    AssertThrow(C1111 > 0.0,
                dealii::ExcLowerRangeType<double>(C1111, 0.0));
    AssertThrow(C1122 > 0.0,
                dealii::ExcLowerRangeType<double>(C1122, 0.0));
    AssertThrow(C1212 > 0.0,
                dealii::ExcLowerRangeType<double>(C1212, 0.0));
    AssertIsFinite(C1111);
    AssertIsFinite(C1122);
    AssertIsFinite(C1212);
  }
  prm.leave_subsection();
}



ScalarMicroscopicStressLawParameters::ScalarMicroscopicStressLawParameters()
:
regularization_function(RegularizationFunction::Tanh),
regularization_parameter(3e-4),
initial_slip_resistance(0.0),
linear_hardening_modulus(500),
hardening_parameter(1.4)
{}



void ScalarMicroscopicStressLawParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Scalar microscopic stress law's parameters");
  {
    prm.declare_entry("Regularization function",
                      "tanh",
                      dealii::Patterns::Selection("tanh|power-law"));

    prm.declare_entry("Regularization parameter",
                      "3e-4",
                      dealii::Patterns::Double());

    prm.declare_entry("Initial slip resistance",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Linear hardening modulus",
                      "500",
                      dealii::Patterns::Double());

    prm.declare_entry("Hardening parameter",
                      "1.4",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();
}



void ScalarMicroscopicStressLawParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Scalar microscopic stress law's parameters");
  {
    const std::string string_regularization_function(
                      prm.get("Regularization function"));

    if (string_regularization_function == std::string("tanh"))
      regularization_function = RegularizationFunction::Tanh;
    else if (string_regularization_function == std::string("power-law"))
      regularization_function = RegularizationFunction::PowerLaw;
    else
      AssertThrow(false,
                  dealii::ExcMessage("Unexpected identifier for the "
                                    "regularization function."));

    regularization_parameter  = prm.get_double("Regularization parameter");
    initial_slip_resistance   = prm.get_double("Initial slip resistance");
    linear_hardening_modulus  = prm.get_double("Linear hardening modulus");
    hardening_parameter       = prm.get_double("Hardening parameter");

    AssertThrow(regularization_parameter > 0.0,
                dealii::ExcLowerRangeType<double>(regularization_parameter, 0.0));
    AssertThrow(initial_slip_resistance >= 0.0,
                dealii::ExcLowerRangeType<double>(initial_slip_resistance, 0.0));
    AssertThrow(linear_hardening_modulus >= 0.0,
                dealii::ExcLowerRangeType<double>(linear_hardening_modulus, 0.0));
    AssertThrow(hardening_parameter > 0.0,
                dealii::ExcLowerRangeType<double>(hardening_parameter, 0.0));

    AssertIsFinite(regularization_parameter);
    AssertIsFinite(initial_slip_resistance);
    AssertIsFinite(linear_hardening_modulus);
    AssertIsFinite(hardening_parameter);
  }
  prm.leave_subsection();
}



VectorMicroscopicStressLawParameters::VectorMicroscopicStressLawParameters()
:
energetic_length_scale(0.0),
initial_slip_resistance(0.0)
{}



void VectorMicroscopicStressLawParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Vector microscopic stress law's parameters");
  {
    prm.declare_entry("Energetic length scale",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Initial slip resistance",
                      "0.0",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();
}



void VectorMicroscopicStressLawParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Vector microscopic stress law's parameters");
  {
    energetic_length_scale  = prm.get_double("Energetic length scale");
    initial_slip_resistance = prm.get_double("Initial slip resistance");

    AssertThrow(energetic_length_scale >= 0.0,
                dealii::ExcLowerRangeType<double>(energetic_length_scale, 0.0));
    AssertThrow(initial_slip_resistance >= 0.0,
                dealii::ExcLowerRangeType<double>(initial_slip_resistance, 0.0));

    AssertIsFinite(energetic_length_scale);
    AssertIsFinite(initial_slip_resistance);
  }
  prm.leave_subsection();
}



MicroscopicTractionLawParameters::MicroscopicTractionLawParameters()
:
grain_boundary_modulus(0.0)
{}



void MicroscopicTractionLawParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Microscopic traction law's parameters");
  {
    prm.declare_entry("Grain boundary modulus",
                      "0.0",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();
}



void MicroscopicTractionLawParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Microscopic traction law's parameters");
  {
    grain_boundary_modulus  = prm.get_double("Grain boundary modulus");

    AssertThrow(grain_boundary_modulus >= 0.0,
                dealii::ExcLowerRangeType<double>(
                  grain_boundary_modulus, 0.0));

    AssertIsFinite(grain_boundary_modulus);
  }
  prm.leave_subsection();
}



CohesiveLawParameters::CohesiveLawParameters()
:
critical_cohesive_traction(0.0),
critical_opening_displacement(0.0),
tangential_to_normal_stiffness_ratio(1.0),
damage_accumulation_constant(1.0),
damage_decay_constant(0.0),
damage_decay_exponent(1.0),
endurance_limit(0.0),
degradation_exponent(1.0),
flag_set_damage_to_zero(false)
{}



void CohesiveLawParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Decohesion law's parameters");
  {
    prm.declare_entry("Maximum cohesive traction",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Critical opening displacement",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Tangential to normal stiffness ratio",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Damage accumulation constant",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Damage decay constant",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Damage decay exponent",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Endurance limit",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Degradation exponent",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Set damage to zero",
                      "false",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();
}



void CohesiveLawParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Decohesion law's parameters");
  {
    critical_cohesive_traction =
      prm.get_double("Maximum cohesive traction");

    AssertThrow(critical_cohesive_traction > 0.0,
                dealii::ExcLowerRangeType<double>(
                  critical_cohesive_traction, 0.0));

    AssertIsFinite(critical_cohesive_traction);

    critical_opening_displacement =
      prm.get_double("Critical opening displacement");

    AssertThrow(critical_opening_displacement > 0.0,
                dealii::ExcLowerRangeType<double>(
                  critical_opening_displacement, 0.0));

    AssertIsFinite(critical_opening_displacement);

    tangential_to_normal_stiffness_ratio =
      prm.get_double("Tangential to normal stiffness ratio");

    AssertThrow(tangential_to_normal_stiffness_ratio > 0.0,
                dealii::ExcLowerRangeType<double>(
                  tangential_to_normal_stiffness_ratio, 0.0));

    AssertIsFinite(tangential_to_normal_stiffness_ratio);

    damage_accumulation_constant =
      prm.get_double("Damage accumulation constant");

    AssertThrow(damage_accumulation_constant > 0.0,
                dealii::ExcLowerRangeType<double>(
                  damage_accumulation_constant, 0.0));

    AssertIsFinite(damage_accumulation_constant);

    damage_decay_constant = prm.get_double("Damage decay constant");

    AssertThrow(damage_decay_constant >= 0.0,
                dealii::ExcLowerRangeType<double>(
                  damage_decay_constant, 0.0));

    AssertIsFinite(damage_decay_constant);

    damage_decay_exponent = prm.get_double("Damage decay exponent");

    AssertThrow(damage_decay_exponent >= 0.0,
                dealii::ExcLowerRangeType<double>(
                  damage_decay_exponent, 0.0));

    AssertIsFinite(damage_decay_exponent);

    endurance_limit = prm.get_double("Endurance limit");

    AssertThrow(endurance_limit >= 0.0,
                dealii::ExcLowerRangeType<double>(
                  endurance_limit, 0.0));

    AssertIsFinite(endurance_limit);

    degradation_exponent = prm.get_double("Degradation exponent");

    AssertThrow(degradation_exponent >= 0.0,
                dealii::ExcLowerRangeType<double>(
                  degradation_exponent, 0.0));

    AssertIsFinite(degradation_exponent);

    flag_set_damage_to_zero = prm.get_bool("Set damage to zero");

  }
  prm.leave_subsection();
}



SolverParameters::SolverParameters()
:
residual_tolerance(1e-10),
newton_update_tolerance(1e-8),
n_max_nonlinear_iterations(1000),
krylov_relative_tolerance(1e-6),
krylov_absolute_tolerance(1e-8),
n_max_krylov_iterations(1000),
allow_decohesion(false),
boundary_conditions_at_grain_boundaries(
  BoundaryConditionsAtGrainBoundaries::Microfree),
logger_output_directory("results/default/"),
print_sparsity_pattern(false),
verbose(false)
{}



void SolverParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Nonlinear solver's parameters");
  {
    prm.declare_entry("Tolerance of the residual",
                      "1e-10",
                      dealii::Patterns::Double());

    prm.declare_entry("Tolerance of the newton update",
                      "1e-8",
                      dealii::Patterns::Double());

    prm.declare_entry("Maximum number of iterations of the nonlinear solver",
                      "1000",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Relative tolerance of the Krylov-solver",
                      "1e-6",
                      dealii::Patterns::Double());

    prm.declare_entry("Absolute tolerance of the Krylov-solver",
                      "1e-8",
                      dealii::Patterns::Double());

    prm.declare_entry("Maximum number of iterations of the Krylov-solver",
                      "1000",
                      dealii::Patterns::Integer(1));
  }
  prm.leave_subsection();

  prm.enter_subsection("Constitutive laws' parameters");
  {
    HookeLawParameters::declare_parameters(prm);
    ScalarMicroscopicStressLawParameters::declare_parameters(prm);
    VectorMicroscopicStressLawParameters::declare_parameters(prm);
    MicroscopicTractionLawParameters::declare_parameters(prm);
    CohesiveLawParameters::declare_parameters(prm);
  }
  prm.leave_subsection();

  prm.declare_entry("Allow decohesion at grain boundaries",
                    "false",
                    dealii::Patterns::Bool());

  prm.declare_entry("Boundary conditions at grain boundaries",
                    "microfree",
                    dealii::Patterns::Selection(
                      "microhard|microfree|microtraction"));


  prm.declare_entry("Logger output directory",
                    "results/default/",
                    dealii::Patterns::DirectoryName());

  prm.declare_entry("Print sparsity pattern",
                    "false",
                    dealii::Patterns::Bool());

  prm.declare_entry("Verbose",
                    "false",
                    dealii::Patterns::Bool());
}



void SolverParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Nonlinear solver's parameters");
  {
    residual_tolerance =
      prm.get_double("Tolerance of the residual");
    AssertThrow(residual_tolerance > 0,
                dealii::ExcLowerRange(residual_tolerance, 0));

    newton_update_tolerance =
      prm.get_double("Tolerance of the newton update");
    AssertThrow(newton_update_tolerance > 0,
                dealii::ExcLowerRange(newton_update_tolerance, 0));

    n_max_nonlinear_iterations =
      prm.get_integer("Maximum number of iterations of the nonlinear solver");
    AssertThrow(n_max_nonlinear_iterations > 0,
                dealii::ExcLowerRange(n_max_nonlinear_iterations, 0));

    krylov_relative_tolerance =
      prm.get_double("Relative tolerance of the Krylov-solver");
    AssertThrow(krylov_relative_tolerance > 0,
                dealii::ExcLowerRange(krylov_relative_tolerance, 0));

    krylov_absolute_tolerance =
      prm.get_double("Absolute tolerance of the Krylov-solver");
    AssertThrow(krylov_relative_tolerance > krylov_absolute_tolerance,
                dealii::ExcLowerRangeType<double>(
                  krylov_relative_tolerance , krylov_absolute_tolerance));

    n_max_krylov_iterations =
      prm.get_integer("Maximum number of iterations of the Krylov-solver");
    AssertThrow(n_max_krylov_iterations > 0,
                dealii::ExcLowerRange(n_max_krylov_iterations, 0));
  }
  prm.leave_subsection();

  prm.enter_subsection("Constitutive laws' parameters");
  {
    hooke_law_parameters.parse_parameters(prm);
    scalar_microscopic_stress_law_parameters.parse_parameters(prm);
    vector_microscopic_stress_law_parameters.parse_parameters(prm);
    microscopic_traction_law_parameters.parse_parameters(prm);
    cohesive_law_parameters.parse_parameters(prm);
  }
  prm.leave_subsection();

  allow_decohesion = prm.get_bool("Allow decohesion at grain boundaries");

  const std::string string_boundary_conditions_at_grain_boundaries(
                    prm.get("Boundary conditions at grain boundaries"));

  if (string_boundary_conditions_at_grain_boundaries ==
        std::string("microhard"))
  {
    AssertThrow(false,
      dealii::ExcMessage(
        "Microhard boundary conditions have yet to be implemented."));
    boundary_conditions_at_grain_boundaries =
      BoundaryConditionsAtGrainBoundaries::Microhard;
  }
  else if (string_boundary_conditions_at_grain_boundaries ==
            std::string("microfree"))
    boundary_conditions_at_grain_boundaries =
      BoundaryConditionsAtGrainBoundaries::Microfree;
  else if (string_boundary_conditions_at_grain_boundaries ==
            std::string("microtraction"))
    boundary_conditions_at_grain_boundaries =
      BoundaryConditionsAtGrainBoundaries::Microtraction;
  else
    AssertThrow(false,
      dealii::ExcMessage(
        "Unexpected identifier for the boundary conditions at grain "
        "boundaries."));


  logger_output_directory = prm.get("Logger output directory");

  print_sparsity_pattern = prm.get_bool("Print sparsity pattern");

  verbose = prm.get_bool("Verbose");

}



TemporalDiscretizationParameters::TemporalDiscretizationParameters()
:
start_time(0.0),
end_time(1.0),
time_step_size(0.25),
period(1.0),
n_cycles(1.0),
n_discrete_time_points_per_half_cycle(5),
initial_loading_time(1.0),
n_discrete_time_points_in_loading_phase(10),
time_step_size_in_loading_phase(time_step_size),
simulation_time_control(SimulationTimeControl::TimeSteered)
{}



void TemporalDiscretizationParameters::
declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Temporal discretization parameters");
  {
    prm.declare_entry("Start time",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("End time",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Time step size",
                      "1e-1",
                      dealii::Patterns::Double());

    prm.declare_entry("Period",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Number of cycles",
                      "1",
                      dealii::Patterns::Integer());

    prm.declare_entry("Discrete time points per half cycle",
                      "1",
                      dealii::Patterns::Integer());

    prm.declare_entry("Initial loading time",
                      "0.5",
                      dealii::Patterns::Double());

    prm.declare_entry("Discrete time points in loading phase",
                      "1e-1",
                      dealii::Patterns::Integer());

    prm.declare_entry("Simulation time control",
                      "time-steered",
                      dealii::Patterns::Selection(
                        "time-steered|cycle-steered"));
  }
  prm.leave_subsection();
}



void TemporalDiscretizationParameters::
parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Temporal discretization parameters");
  {
    start_time            = prm.get_double("Start time");

    end_time              = prm.get_double("End time");

    time_step_size        = prm.get_double("Time step size");

    period                = prm.get_double("Period");

    n_cycles              = prm.get_integer("Number of cycles");

    initial_loading_time  = prm.get_double("Period");

    n_discrete_time_points_per_half_cycle
      = prm.get_integer("Discrete time points per half cycle");

    n_discrete_time_points_in_loading_phase
      = prm.get_integer("Discrete time points in loading phase");

    const std::string string_simulation_time_control(
                      prm.get("Simulation time control"));

    if (string_simulation_time_control == std::string("time-steered"))
      simulation_time_control = SimulationTimeControl::TimeSteered;
    else if (string_simulation_time_control == std::string("cycle-steered"))
      simulation_time_control = SimulationTimeControl::CycleSteered;
    else
      AssertThrow(
        false,
        dealii::ExcMessage("Unexpected identifier for the simulation"
                           " time control"));

    if (simulation_time_control == SimulationTimeControl::CycleSteered)
    {
      end_time = start_time + initial_loading_time + n_cycles * period;

      time_step_size =
        0.5 * period / (n_discrete_time_points_per_half_cycle - 1);

      time_step_size_in_loading_phase =
        initial_loading_time /
        (n_discrete_time_points_in_loading_phase - 1);
    }
    else
      time_step_size_in_loading_phase = time_step_size;

    Assert(start_time >= 0.0,
           dealii::ExcLowerRangeType<double>(start_time, 0.0));

    Assert(end_time > start_time,
           dealii::ExcLowerRangeType<double>(end_time, start_time));

    Assert(time_step_size > 0,
           dealii::ExcLowerRangeType<double>(time_step_size, 0));

    Assert(period > 0,
           dealii::ExcLowerRangeType<double>(period, 0));

    Assert(n_cycles > 0,
           dealii::ExcLowerRangeType<int>(n_cycles, 0));

    Assert(n_discrete_time_points_per_half_cycle > 1,
           dealii::ExcLowerRangeType<int>(
            n_discrete_time_points_per_half_cycle, 1));

    Assert(n_discrete_time_points_in_loading_phase > 1,
           dealii::ExcLowerRangeType<int>(
            n_discrete_time_points_in_loading_phase, 1));

    Assert(end_time >= (start_time + time_step_size),
           dealii::ExcLowerRangeType<double>(
            end_time, start_time + time_step_size));
  }
  prm.leave_subsection();
}



ProblemParameters::ProblemParameters()
:
dim(2),
mapping_degree(1),
mapping_interior_cells(false),
n_global_refinements(0),
start_time(0.0),
end_time(1.0),
time_step_size(0.5),
fe_degree_displacements(2),
fe_degree_slips(1),
slips_normals_pathname("input/slip_normals"),
slips_directions_pathname("input/slip_directions"),
euler_angles_pathname("input/euler_angles"),
graphical_output_frequency(1),
terminal_output_frequency(1),
graphical_output_directory("results/default/"),
verbose(true)
{}



ProblemParameters::ProblemParameters(
  const std::string &parameter_filename)
:
ProblemParameters()
{
  dealii::ParameterHandler prm;

  declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;

    message << "Input parameter file <"
            << parameter_filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(parameter_filename.c_str());

    prm.print_parameters(parameter_out,
                         dealii::ParameterHandler::OutputStyle::PRM);

    AssertThrow(false, dealii::ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}



void ProblemParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Spatial discretization parameters");
  {
    prm.declare_entry("Spatial dimension",
                      "2",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Mapping - Polynomial degree",
                      "1",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Mapping - Apply to interior cells",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Number of global refinements",
                      "0",
                      dealii::Patterns::Integer(0));

    prm.declare_entry("FE's polynomial degree - Displacements",
                      "2",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("FE's polynomial degree - Slips",
                      "1",
                      dealii::Patterns::Integer(1));
  }
  prm.leave_subsection();

  prm.enter_subsection("Temporal discretization parameters");
  {
    prm.declare_entry("Start time",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("End time",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Time step size",
                      "5e-1",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();

  prm.declare_entry("Verbose",
                    "false",
                    dealii::Patterns::Bool());

  prm.enter_subsection("Solver parameters");
  {
    SolverParameters::declare_parameters(prm);
  }
  prm.leave_subsection();

  prm.enter_subsection("Input files");
  {
    prm.declare_entry("Slip normals path name",
                      "input/slip_normals",
                      dealii::Patterns::FileName());

    prm.declare_entry("Slip directions path name",
                      "input/slip_directions",
                      dealii::Patterns::FileName());

    prm.declare_entry("Euler angles path name",
                      "input/euler_angles",
                      dealii::Patterns::FileName());
  }
  prm.leave_subsection();

  prm.enter_subsection("Output control parameters");
  {
    prm.declare_entry("Graphical output frequency",
                      "1",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Terminal output frequency",
                      "1",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Graphical output directory",
                      "results/default/",
                      dealii::Patterns::DirectoryName());
  }
  prm.leave_subsection();
}



void ProblemParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Spatial discretization parameters");
  {
    dim = prm.get_integer("Spatial dimension");
    AssertThrow(dim > 0, dealii::ExcLowerRange(dim, 0) );
    AssertThrow(dim <= 3,
                dealii::ExcMessage(
                  "The spatial dimension is larger than three.") );

    mapping_degree = prm.get_integer("Mapping - Polynomial degree");
    AssertThrow(mapping_degree > 0, dealii::ExcLowerRange(mapping_degree, 0) );

    mapping_interior_cells = prm.get_bool("Mapping - Apply to interior cells");

    n_global_refinements = prm.get_integer("Number of global refinements");

    fe_degree_displacements = prm.get_integer("FE's polynomial degree - Displacements");
    AssertThrow(fe_degree_displacements > 0,
                dealii::ExcLowerRange(fe_degree_displacements, 0));

    fe_degree_slips = prm.get_integer("FE's polynomial degree - Slips");
    AssertThrow(fe_degree_slips > 0,
                dealii::ExcLowerRange(fe_degree_slips, 0));
  }
  prm.leave_subsection();

  prm.enter_subsection("Temporal discretization parameters");
  {
    start_time = prm.get_double("Start time");

    end_time = prm.get_double("End time");

    time_step_size = prm.get_double("Time step size");

    Assert(start_time >= 0.0,
           dealii::ExcLowerRangeType<double>(start_time, 0.0));
    Assert(end_time > start_time,
           dealii::ExcLowerRangeType<double>(end_time, start_time));
    Assert(time_step_size > 0,
           dealii::ExcLowerRangeType<double>(time_step_size, 0));
    Assert(time_step_size > 0,
           dealii::ExcLowerRangeType<double>(time_step_size, 0));
    Assert(end_time >= (start_time + time_step_size),
           dealii::ExcLowerRangeType<double>(end_time,
                                             start_time + time_step_size));

  }
  prm.leave_subsection();

  verbose = prm.get_bool("Verbose");

  prm.enter_subsection("Solver parameters");
  {
    solver_parameters.parse_parameters(prm);
  }
  prm.leave_subsection();

  prm.enter_subsection("Input files");
  {
    slips_normals_pathname = prm.get("Slip normals path name");

    slips_directions_pathname = prm.get("Slip directions path name");

    euler_angles_pathname = prm.get("Euler angles path name");
  }
  prm.leave_subsection();


  prm.enter_subsection("Output control parameters");
  {
    graphical_output_frequency = prm.get_integer("Graphical output frequency");
    Assert(graphical_output_frequency > 0,
           dealii::ExcLowerRange(graphical_output_frequency, 0));

    terminal_output_frequency = prm.get_integer("Terminal output frequency");
    Assert(terminal_output_frequency > 0,
           dealii::ExcLowerRange(terminal_output_frequency, 0));

    graphical_output_directory = prm.get("Graphical output directory");
  }
  prm.leave_subsection();
}



SimpleShearParameters::SimpleShearParameters()
:
ProblemParameters(),
n_equal_sized_divisions(1),
height(1),
width(0.1)
{}



SimpleShearParameters::SimpleShearParameters(
  const std::string &parameter_filename)
:
SimpleShearParameters()
{
  dealii::ParameterHandler prm;

  declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;

    message << "Input parameter file <"
            << parameter_filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(parameter_filename.c_str());

    prm.print_parameters(parameter_out,
                         dealii::ParameterHandler::OutputStyle::PRM);

    AssertThrow(false, dealii::ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}



void SimpleShearParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  ProblemParameters::declare_parameters(prm);

  prm.enter_subsection("Simple shear");
  {
    prm.declare_entry("Shear strain at the upper boundary",
                      "0.0218",
                      dealii::Patterns::Double());

    prm.declare_entry("Number of equally sized divisions in y-direction",
                      "1",
                      dealii::Patterns::Integer());

    prm.declare_entry("Height of the strip",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Width of the strip",
                      "0.1",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();
}



void SimpleShearParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  ProblemParameters::parse_parameters(prm);

  prm.enter_subsection("Simple shear");
  {
    shear_strain_at_upper_boundary =
      prm.get_double("Shear strain at the upper boundary");

    AssertIsFinite(shear_strain_at_upper_boundary);

    n_equal_sized_divisions =
      prm.get_integer("Number of equally sized divisions in y-direction");

    Assert(n_equal_sized_divisions > 0,
           dealii::ExcLowerRange(n_equal_sized_divisions, 0));

    AssertIsFinite(n_equal_sized_divisions);

    height = prm.get_double("Height of the strip");

    Assert(height > 0,
           dealii::ExcLowerRange(height, 0));

    AssertIsFinite(height);

    width = prm.get_double("Width of the strip");

    Assert(width > 0,
        dealii::ExcLowerRange(width, 0));

    AssertIsFinite(width);
  }
  prm.leave_subsection();
}


/*
template<typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm)
{

}
*/

} // namespace RunTimeParameters



} // namespace gCP


/*
template std::ostream & gCP::RunTimeParameters::operator<<
(std::ostream &, const gCP::RunTimeParameters::Parameters &);
template dealii::ConditionalOStream  & gCP::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const gCP::RunTimeParameters::Parameters &);
*/