#include <gCP/run_time_parameters.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

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
                      dealii::Patterns::Selection("atan|sqrt|gd|tanh|erf"));

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

    if (string_regularization_function == std::string("atan"))
      regularization_function = RegularizationFunction::Atan;
    else if (string_regularization_function == std::string("sqrt"))
      regularization_function = RegularizationFunction::Sqrt;
    else if (string_regularization_function == std::string("gd"))
      regularization_function = RegularizationFunction::Gd;
    else if (string_regularization_function == std::string("tanh"))
      regularization_function = RegularizationFunction::Tanh;
    else if (string_regularization_function == std::string("erf"))
      regularization_function = RegularizationFunction::Erf;
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
critical_cohesive_traction(700.),
critical_opening_displacement(2.5e-2),
tangential_to_normal_stiffness_ratio(1.0),
damage_accumulation_constant(1.0),
damage_decay_constant(0.0),
damage_decay_exponent(1.0),
endurance_limit(0.0),
degradation_exponent(1.0),
flag_couple_microtraction_to_damage(true),
flag_couple_macrotraction_to_damage(false),
flag_set_damage_to_zero(false)
{}



void CohesiveLawParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Decohesion law's parameters");
  {
    prm.declare_entry("Maximum cohesive traction",
                      "700.",
                      dealii::Patterns::Double());

    prm.declare_entry("Critical opening displacement",
                      "2.5e-2",
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

    prm.declare_entry("Couple microtraction to damage",
                      "true",
                      dealii::Patterns::Bool());

    prm.declare_entry("Couple macrotraction to damage",
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

    flag_couple_microtraction_to_damage =
      prm.get_bool("Couple microtraction to damage");

    flag_couple_macrotraction_to_damage =
      prm.get_bool("Couple macrotraction to damage");

  }
  prm.leave_subsection();
}



ContactLawParameters::ContactLawParameters()
:
penalty_coefficient(100.0)
{}



void ContactLawParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Contact law's parameters");
  {
    prm.declare_entry("Penalty coefficient",
                      "100.0",
                      dealii::Patterns::Double());

  }
  prm.leave_subsection();
}



void ContactLawParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Contact law's parameters");
  {
    penalty_coefficient = prm.get_double("Penalty coefficient");

    AssertThrow(penalty_coefficient >= 0.0,
                dealii::ExcLowerRangeType<double>(
                  penalty_coefficient, 0.0));

    AssertIsFinite(penalty_coefficient);
  }
  prm.leave_subsection();
}



KrylovParameters::KrylovParameters()
:
solver_type(SolverType::CG),
relative_tolerance(1e-6),
absolute_tolerance(1e-8),
tolerance_relaxation_factor(1.0),
n_max_iterations(1000)
{}



void KrylovParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Krylov parameters");
  {
    prm.declare_entry("Solver type",
                      "cg",
                      dealii::Patterns::Selection("directsolver|cg|gmres"));

    prm.declare_entry("Relative tolerance",
                      "1e-6",
                      dealii::Patterns::Double());

    prm.declare_entry("Absolute tolerance",
                      "1e-8",
                      dealii::Patterns::Double());

    prm.declare_entry("Relaxation factor of the tolerances",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Maximum number of iterations",
                      "1000",
                      dealii::Patterns::Integer());
  }
  prm.leave_subsection();
}



void KrylovParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Krylov parameters");
  {
    const std::string string_solver_type(prm.get("Solver type"));

    if (string_solver_type == std::string("directsolver"))
    {
      solver_type = SolverType::DirectSolver;
    }
    else if (string_solver_type == std::string("cg"))
    {
      solver_type = SolverType::CG;
    }
    else if (string_solver_type == std::string("gmres"))
    {
      solver_type = SolverType::GMRES;
    }
    else
    {
      AssertThrow(false,
        dealii::ExcMessage("Unexpected identifier for the solver type."));
    }

    relative_tolerance  = prm.get_double("Relative tolerance");

    absolute_tolerance  = prm.get_double("Absolute tolerance");

    tolerance_relaxation_factor =
      prm.get_double("Relaxation factor of the tolerances");

    n_max_iterations =
      prm.get_integer("Maximum number of iterations");

    AssertThrow(relative_tolerance > 0,
                dealii::ExcLowerRange(relative_tolerance, 0));

    AssertThrow(absolute_tolerance > 0,
                dealii::ExcLowerRange(absolute_tolerance, 0));

    AssertThrow(relative_tolerance > absolute_tolerance,
                dealii::ExcLowerRangeType<double>(
                  relative_tolerance , absolute_tolerance));

    AssertThrow(tolerance_relaxation_factor > 0,
                dealii::ExcLowerRangeType<double>(
                  tolerance_relaxation_factor, 0));

    AssertThrow(n_max_iterations > 0,
                dealii::ExcLowerRange(n_max_iterations, 0));
  }
  prm.leave_subsection();
}



NewtonRaphsonParameters::NewtonRaphsonParameters()
:
relative_tolerance(1e-6),
absolute_tolerance(1e-8),
step_tolerance(1e-8),
n_max_iterations(15)
{}



void NewtonRaphsonParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Newton-Raphson parameters");
  {
    prm.declare_entry("Relative tolerance of the residual",
                      "1e-6",
                      dealii::Patterns::Double());

    prm.declare_entry("Absolute tolerance of the residual",
                      "1e-8",
                      dealii::Patterns::Double());

    prm.declare_entry("Absolute tolerance of the step",
                      "1e-8",
                      dealii::Patterns::Double());

    prm.declare_entry("Maximum number of iterations",
                      "15",
                      dealii::Patterns::Integer());
  }
  prm.leave_subsection();
}



void NewtonRaphsonParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Newton-Raphson parameters");
  {
    relative_tolerance =
      prm.get_double("Relative tolerance of the residual");

    absolute_tolerance =
      prm.get_double("Absolute tolerance of the residual");

    step_tolerance =
      prm.get_double("Absolute tolerance of the step");

    n_max_iterations =
      prm.get_integer("Maximum number of iterations");

    AssertThrow(step_tolerance > 0,
                dealii::ExcLowerRange(step_tolerance, 0));

    AssertThrow(relative_tolerance > 0,
                dealii::ExcLowerRange(relative_tolerance, 0));

    AssertThrow(relative_tolerance > absolute_tolerance,
                dealii::ExcLowerRangeType<double>(
                  relative_tolerance , absolute_tolerance));

    AssertThrow(n_max_iterations > 0,
                dealii::ExcLowerRange(n_max_iterations, 0));

  }
  prm.leave_subsection();
}



ConvergenceControlParameters::ConvergenceControlParameters()
:
upscaling_factor(1),
downscaling_factor(1),
upper_threshold(2),
lower_threshold(1),
n_max_iterations(5)
{}



void ConvergenceControlParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Convergence control parameters");
  {
    prm.declare_entry("Upscaling factor",
                      "1",
                      dealii::Patterns::Double());

    prm.declare_entry("Downscaling factor",
                      "1",
                      dealii::Patterns::Double());

    prm.declare_entry("Upper threshold",
                      "2",
                      dealii::Patterns::Double());

    prm.declare_entry("Lower threshold",
                      "1",
                      dealii::Patterns::Double());

    prm.declare_entry("Maximum number of iterations",
                      "5",
                      dealii::Patterns::Integer());
  }
  prm.leave_subsection();
}



void ConvergenceControlParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Convergence control parameters");
  {
    upscaling_factor =
      prm.get_double("Upscaling factor");

    downscaling_factor =
      prm.get_double("Downscaling factor");

    upper_threshold =
      prm.get_double("Upper threshold");

    lower_threshold =
      prm.get_double("Lower threshold");

    n_max_iterations =
      prm.get_integer("Maximum number of iterations");

    AssertThrow(upscaling_factor > 0,
                dealii::ExcLowerRange(upscaling_factor, 0));

    AssertThrow(downscaling_factor > 0,
                dealii::ExcLowerRange(downscaling_factor, 0));

    AssertThrow(lower_threshold > 0,
                dealii::ExcLowerRangeType<double>(
                  lower_threshold, 0));

    AssertThrow(upper_threshold > lower_threshold,
                dealii::ExcLowerRangeType<double>(
                  upper_threshold, lower_threshold));

    AssertThrow(n_max_iterations > 0,
                dealii::ExcLowerRange(n_max_iterations, 0));

  }
  prm.leave_subsection();
}



SolverParameters::SolverParameters()
:
microforce_balance_scaling_factor(1.0),
linear_momentum_balance_scaling_factor(1.0),
allow_decohesion(false),
boundary_conditions_at_grain_boundaries(
  BoundaryConditionsAtGrainBoundaries::Microfree),
logger_output_directory("results/default/"),
flag_skip_extrapolation_at_extrema(false),
flag_zero_damage_during_loading_and_unloading(false),
print_sparsity_pattern(false),
verbose(false)
{}



void SolverParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  KrylovParameters::declare_parameters(prm);
  NewtonRaphsonParameters::declare_parameters(prm);
  ConvergenceControlParameters::declare_parameters(prm);

  prm.enter_subsection("Constitutive laws' parameters");
  {
    HookeLawParameters::declare_parameters(prm);
    ScalarMicroscopicStressLawParameters::declare_parameters(prm);
    VectorMicroscopicStressLawParameters::declare_parameters(prm);
    MicroscopicTractionLawParameters::declare_parameters(prm);
    CohesiveLawParameters::declare_parameters(prm);
    ContactLawParameters::declare_parameters(prm);
  }
  prm.leave_subsection();

  prm.declare_entry("Scaling factor of the microforce balance",
                    "1",
                    dealii::Patterns::Double());

  prm.declare_entry("Scaling factor of the linear momentum balance",
                    "1",
                    dealii::Patterns::Double());

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

  prm.declare_entry("Skip extrapolation of start value at extrema",
                    "false",
                    dealii::Patterns::Bool());

  prm.declare_entry("Zero damage evolution during un- and loading",
                    "false",
                    dealii::Patterns::Bool());

  prm.declare_entry("Print sparsity pattern",
                    "false",
                    dealii::Patterns::Bool());

  prm.declare_entry("Verbose",
                    "false",
                    dealii::Patterns::Bool());
}



void SolverParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  krylov_parameters.parse_parameters(prm);

  newton_parameters.parse_parameters(prm);

  convergence_control_parameters.parse_parameters(prm);

  prm.enter_subsection("Constitutive laws' parameters");
  {
    hooke_law_parameters.parse_parameters(prm);
    scalar_microscopic_stress_law_parameters.parse_parameters(prm);
    vector_microscopic_stress_law_parameters.parse_parameters(prm);
    microscopic_traction_law_parameters.parse_parameters(prm);
    cohesive_law_parameters.parse_parameters(prm);
    contact_law_parameters.parse_parameters(prm);
  }
  prm.leave_subsection();

  microforce_balance_scaling_factor =
    prm.get_double("Scaling factor of the microforce balance");

    AssertThrow(microforce_balance_scaling_factor > 0,
                dealii::ExcLowerRangeType<double>(
                  microforce_balance_scaling_factor, 0));

  linear_momentum_balance_scaling_factor =
    prm.get_double("Scaling factor of the linear momentum balance");

    AssertThrow(linear_momentum_balance_scaling_factor > 0,
                dealii::ExcLowerRangeType<double>(
                  linear_momentum_balance_scaling_factor, 0));

  allow_decohesion = prm.get_bool("Allow decohesion at grain boundaries");

  const std::string string_boundary_conditions_at_grain_boundaries(
                    prm.get("Boundary conditions at grain boundaries"));

  if (string_boundary_conditions_at_grain_boundaries ==
        std::string("microhard"))
  {
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

  flag_skip_extrapolation_at_extrema =
    prm.get_bool("Skip extrapolation of start value at extrema");

  flag_zero_damage_during_loading_and_unloading =
    prm.get_bool("Zero damage evolution during un- and loading");

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
initial_loading_time(1.0),
start_of_unloading_phase(2.0),
n_steps_in_loading_phase(2),
n_steps_per_half_cycle(2),
n_steps_in_unloading_phase(2),
time_step_size_in_loading_phase(time_step_size),
time_step_size_in_unloading_phase(time_step_size),
loading_type(LoadingType::Monotonic)
{}



void TemporalDiscretizationParameters::
declare_parameters(dealii::ParameterHandler &prm)
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

  prm.declare_entry("Number of steps per half cycle",
                    "2",
                    dealii::Patterns::Integer());

  prm.declare_entry("Number of steps in unloading phase",
                    "2",
                    dealii::Patterns::Integer());

  prm.declare_entry("Initial loading time",
                    "0.5",
                    dealii::Patterns::Double());

  prm.declare_entry("Number of steps in loading phase",
                    "2",
                    dealii::Patterns::Integer());

  prm.declare_entry("Loading type",
                    "monotonic",
                    dealii::Patterns::Selection(
                      "monotonic|cyclic|cyclic_unloading"));
}



void TemporalDiscretizationParameters::
parse_parameters(dealii::ParameterHandler &prm)
{
  const std::string string_loading_type(
                    prm.get("Loading type"));

  if (string_loading_type == std::string("monotonic"))
    loading_type = LoadingType::Monotonic;
  else if (string_loading_type == std::string("cyclic"))
    loading_type = LoadingType::Cyclic;
  else if (string_loading_type == std::string("cyclic_unloading"))
    loading_type = LoadingType::CyclicWithUnloading;
  else
    AssertThrow(
      false,
      dealii::ExcMessage("Unexpected identifier for the simulation"
                          " time control"));

  start_time            = prm.get_double("Start time");

  end_time              = prm.get_double("End time");

  time_step_size        = prm.get_double("Time step size");

  period                = prm.get_double("Period");

  n_cycles              = prm.get_integer("Number of cycles");

  initial_loading_time  = prm.get_double("Initial loading time");

  start_of_unloading_phase  = prm.get_double("End time");

  n_steps_per_half_cycle
    = prm.get_integer("Number of steps per half cycle");

  n_steps_in_loading_phase
    = prm.get_integer("Number of steps in loading phase");

  n_steps_in_unloading_phase
    = prm.get_integer("Number of steps in unloading phase");

  if (loading_type == LoadingType::Cyclic)
  {
    end_time = start_time + initial_loading_time + n_cycles * period;

    time_step_size = 0.5 * period / n_steps_per_half_cycle;

    time_step_size_in_loading_phase =
      initial_loading_time / n_steps_in_loading_phase;
  }
  else if (loading_type == LoadingType::CyclicWithUnloading)
  {
    start_of_unloading_phase =
      start_time + initial_loading_time + n_cycles * period;

    end_time = start_of_unloading_phase + 1.0;

    time_step_size = 0.5 * period / n_steps_per_half_cycle;

    time_step_size_in_loading_phase =
      initial_loading_time / n_steps_in_loading_phase;

    time_step_size_in_unloading_phase =
      1.0 / n_steps_in_unloading_phase;
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

  Assert(n_steps_per_half_cycle > 1,
          dealii::ExcLowerRangeType<int>(
          n_steps_per_half_cycle, 1));

  Assert(n_steps_in_loading_phase > 1,
          dealii::ExcLowerRangeType<int>(
          n_steps_in_loading_phase, 1));

  Assert(end_time >= (start_time + time_step_size),
          dealii::ExcLowerRangeType<double>(
          end_time, start_time + time_step_size));
}



ProblemParameters::ProblemParameters()
:
dim(2),
mapping_degree(1),
mapping_interior_cells(false),
n_global_refinements(0),
fe_degree_displacements(2),
fe_degree_slips(1),
slips_normals_pathname("input/slip_normals"),
slips_directions_pathname("input/slip_directions"),
euler_angles_pathname("input/euler_angles"),
graphical_output_frequency(1),
terminal_output_frequency(1),
homogenization_frequency(1),
graphical_output_directory("results/default/"),
flag_compute_macroscopic_quantities(false),
flag_output_damage_variable(false),
flag_output_residual(false),
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

  prm.enter_subsection("Temporal discretization parameters");
  {
    TemporalDiscretizationParameters::declare_parameters(prm);
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

    prm.declare_entry("Output damage variable field",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Output residual field",
                      "false",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();

  /*!
   * @note Temporary parameters
   */
  prm.enter_subsection("Postprocessing parameters");
  {
    prm.declare_entry("Homogenization",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Homogenization frequency",
                      "1",
                      dealii::Patterns::Integer(1));
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
    temporal_discretization_parameters.parse_parameters(prm);
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

    flag_output_damage_variable = prm.get_bool("Output damage variable field");

    flag_output_residual = prm.get_bool("Output residual field");

    if ((dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
        !fs::exists(graphical_output_directory + "paraview/"))
    {
      try
      {
        fs::create_directories(graphical_output_directory + "paraview/");
      }
      catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception in the creation of the output directory: "
                  << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
      catch (...)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                    << std::endl;
        std::cerr << "Unknown exception in the creation of the output directory!"
                  << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
    }
  }
  prm.leave_subsection();

  prm.enter_subsection("Postprocessing parameters");
  {
    flag_compute_macroscopic_quantities =
      prm.get_bool("Homogenization");

    homogenization_frequency = prm.get_integer("Homogenization frequency");
    Assert(homogenization_frequency > 0,
           dealii::ExcLowerRange(homogenization_frequency, 0));
  }
  prm.leave_subsection();
}



SimpleShearParameters::SimpleShearParameters()
:
ProblemParameters(),
max_shear_strain_at_upper_boundary(0.5),
min_shear_strain_at_upper_boundary(0.1),
height(1),
width(0.1),
n_elements_in_y_direction(100),
n_equal_sized_divisions(1)
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

  std::string output_folder_path;

  prm.enter_subsection("Output control parameters");
  {
    output_folder_path = prm.get("Graphical output directory");
  }
  prm.leave_subsection();

  prm.print_parameters(
    output_folder_path + "parameter_file.prm",
    dealii::ParameterHandler::OutputStyle::ShortPRM);
}



void SimpleShearParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  ProblemParameters::declare_parameters(prm);

  prm.enter_subsection("Simple shear");
  {
    prm.declare_entry("Maximum shear strain at the upper boundary",
                      "0.0218",
                      dealii::Patterns::Double());

    prm.declare_entry("Minimum shear strain at the upper boundary",
                      "0.0218",
                      dealii::Patterns::Double());

    prm.declare_entry("Number of elements in y-direction",
                      "100",
                      dealii::Patterns::Integer(0));

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
    max_shear_strain_at_upper_boundary =
      prm.get_double("Maximum shear strain at the upper boundary");

    AssertIsFinite(max_shear_strain_at_upper_boundary);

    min_shear_strain_at_upper_boundary =
      prm.get_double("Minimum shear strain at the upper boundary");

    AssertIsFinite(min_shear_strain_at_upper_boundary);

    n_elements_in_y_direction = prm.get_integer("Number of elements in y-direction");

    Assert(n_elements_in_y_direction > 0,
           dealii::ExcLowerRange(n_elements_in_y_direction, 0));

    AssertIsFinite(n_elements_in_y_direction);

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



SemicoupledParameters::SemicoupledParameters()
:
ProblemParameters(),
strain_component_11(0.01),
strain_component_22(0.01),
strain_component_33(0.01),
strain_component_23(0.01),
strain_component_13(0.01),
strain_component_12(0.01),
min_to_max_strain_load_ratio(.5),
msh_file_pathname("input/mesh/periodic_polycrystal.msh")
{}



SemicoupledParameters::SemicoupledParameters(
  const std::string &parameter_filename)
:
SemicoupledParameters()
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

  std::string output_folder_path;

  prm.enter_subsection("Output control parameters");
  {
    output_folder_path = prm.get("Graphical output directory");
  }
  prm.leave_subsection();

  if ((dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    prm.print_parameters(
      output_folder_path + "parameter_file.prm",
      dealii::ParameterHandler::OutputStyle::ShortPRM);
}



void SemicoupledParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  ProblemParameters::declare_parameters(prm);

  prm.enter_subsection("Semi-coupled problem");
  {
    prm.declare_entry("Max strain component 11",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Max strain component 22",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Max strain component 33",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Max strain component 23",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Max strain component 13",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Max strain component 12",
                      "0.01",
                      dealii::Patterns::Double());

    prm.declare_entry("Minimum to maximum strain load ratio",
                      "0.5",
                      dealii::Patterns::Double());

    prm.declare_entry("Mesh file (*.msh) path name",
                      "input/mesh/periodic_polycrystal.msh",
                      dealii::Patterns::FileName());
  }
  prm.leave_subsection();
}



void SemicoupledParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  ProblemParameters::parse_parameters(prm);

  prm.enter_subsection("Semi-coupled problem");
  {
    strain_component_11   = prm.get_double("Max strain component 11");

    strain_component_22   = prm.get_double("Max strain component 22");

    strain_component_33   = prm.get_double("Max strain component 33");

    strain_component_23   = prm.get_double("Max strain component 23");

    strain_component_13   = prm.get_double("Max strain component 13");

    strain_component_12   = prm.get_double("Max strain component 12");

    msh_file_pathname     = prm.get("Mesh file (*.msh) path name");

    min_to_max_strain_load_ratio =
      prm.get_double("Minimum to maximum strain load ratio");

    AssertIsFinite(strain_component_11);

    AssertIsFinite(strain_component_22);

    AssertIsFinite(strain_component_33);

    AssertIsFinite(strain_component_23);

    AssertIsFinite(strain_component_13);

    AssertIsFinite(strain_component_12);

    AssertIsFinite(min_to_max_strain_load_ratio);

    Assert((0 <= min_to_max_strain_load_ratio) &&
           (min_to_max_strain_load_ratio < 1.0),
           dealii::ExcMessage("The ratio has to be inside the range "
                              "[0,1)"));

    Assert(
      (msh_file_pathname.find_last_of(".") != std::string::npos) &&
      (msh_file_pathname.substr(msh_file_pathname.find_last_of(".")+1)
        == "msh"),
      dealii::ExcMessage(
        "The path *.msh file has no extension or the wrong one"));
  };
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