#include <gCP/run_time_parameters.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <cmath>

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
                      dealii::Patterns::Double(0.));

    prm.declare_entry("C1122",
                      "57692.3",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("C1212",
                      "38461.5",
                      dealii::Patterns::Double(0.));
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
  }
  prm.leave_subsection();
}



ScalarMicrostressLawParameters::ScalarMicrostressLawParameters()
:
regularization_function(RegularizationFunction::Tanh),
regularization_parameter(3e-4),
flag_rate_independent(false)
{}



void ScalarMicrostressLawParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Scalar microstress law's parameters");
  {
    prm.declare_entry("Regularization function",
                      "tanh",
                      dealii::Patterns::Selection("atan|sqrt|gd|tanh|erf"));

    prm.declare_entry("Regularization parameter",
                      "3e-4",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Rate-independent behavior",
                      "false",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();
}



void ScalarMicrostressLawParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Scalar microstress law's parameters");
  {
    const std::string string_regularization_function(
                      prm.get("Regularization function"));

    if (string_regularization_function == std::string("atan"))
    {
      regularization_function = RegularizationFunction::Atan;
    }
    else if (string_regularization_function == std::string("sqrt"))
    {
      regularization_function = RegularizationFunction::Sqrt;
    }
    else if (string_regularization_function == std::string("gd"))
    {
      regularization_function = RegularizationFunction::Gd;
    }
    else if (string_regularization_function == std::string("tanh"))
    {
      regularization_function = RegularizationFunction::Tanh;
    }
    else if (string_regularization_function == std::string("erf"))
    {
      regularization_function = RegularizationFunction::Erf;
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Unexpected identifier for the "
                                    "regularization function."));
    }

    regularization_parameter = prm.get_double("Regularization parameter");

    flag_rate_independent = prm.get_bool("Rate-independent behavior");


    AssertThrow(
      regularization_parameter > 0.0,
      dealii::ExcLowerRangeType<double>(regularization_parameter, 0.0));
  }
  prm.leave_subsection();
}



VectorialMicrostressLawParameters::VectorialMicrostressLawParameters()
:
energetic_length_scale(0.0),
initial_slip_resistance(0.0),
defect_energy_index(2.0),
regularization_parameter(1e-6)
{}



void VectorialMicrostressLawParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Vectorial microstress law's parameters");
  {
    prm.declare_entry("Energetic length scale",
                      "0.0",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Initial slip resistance",
                      "0.0",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Defect energy index",
                      "2.0",
                      dealii::Patterns::Double(1.0));

    prm.declare_entry("Regularization parameter",
                      "1e-6",
                      dealii::Patterns::Double(0.0));
  }
  prm.leave_subsection();
}



void VectorialMicrostressLawParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Vectorial microstress law's parameters");
  {

    energetic_length_scale =
      prm.get_double("Energetic length scale");

    initial_slip_resistance =
      prm.get_double("Initial slip resistance");

    defect_energy_index =
      prm.get_double("Defect energy index");

    regularization_parameter =
      prm.get_double("Regularization parameter");

    AssertThrow(
      regularization_parameter > 0.0,
      dealii::ExcLowerRangeType<double>(regularization_parameter, 0.0));
  }
  prm.leave_subsection();
}



MicrotractionLawParameters::MicrotractionLawParameters()
:
grain_boundary_modulus(0.0)
{}



void MicrotractionLawParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Microtraction law's parameters");
  {
    prm.declare_entry("Grain boundary modulus",
                      "0.0",
                      dealii::Patterns::Double(0.));
  }
  prm.leave_subsection();
}



void MicrotractionLawParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Microtraction law's parameters");
  {
    grain_boundary_modulus  = prm.get_double("Grain boundary modulus");
  }
  prm.leave_subsection();
}



CohesiveLawParameters::CohesiveLawParameters()
:
cohesive_law_model(CohesiveLawModel::OrtizEtAl),
critical_cohesive_traction(700.),
critical_opening_displacement(2.5e-2),
tangential_to_normal_stiffness_ratio(1.0)
{}



void CohesiveLawParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Cohesive law's parameters");
  {
    prm.declare_entry("Cohesive law model",
                      "OrtizEtAl",
                      dealii::Patterns::Selection("OrtizEtAl"));

    prm.declare_entry("Critical cohesive traction",
                      "700.",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Critical opening displacement",
                      "2.5e-2",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Tangential to normal stiffness ratio",
                      "1.0",
                      dealii::Patterns::Double(0.0));
  }
  prm.leave_subsection();
}



void CohesiveLawParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Cohesive law's parameters");
  {
    const std::string string_cohesive_law_model(
      prm.get("Cohesive law model"));

    if (string_cohesive_law_model == std::string("OrtizEtAl"))
    {
      cohesive_law_model = CohesiveLawModel::OrtizEtAl;
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Unexpected identifier for the cohesive law model."));
    }

    critical_cohesive_traction =
      prm.get_double("Critical cohesive traction");

    critical_opening_displacement =
      prm.get_double("Critical opening displacement");

    tangential_to_normal_stiffness_ratio =
      prm.get_double("Tangential to normal stiffness ratio");
  }
  prm.leave_subsection();
}



DegradationFunction::DegradationFunction()
:
degradation_exponent(1.0)
{}



void DegradationFunction::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Degradation function parameters");
  {
    prm.declare_entry("Degradation exponent",
                      "1.0",
                      dealii::Patterns::Double(0.0));
  }
  prm.leave_subsection();
}



void DegradationFunction::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Degradation function parameters");
  {
    degradation_exponent = prm.get_double("Degradation exponent");
  }
  prm.leave_subsection();
}



HardeningLaw::HardeningLaw()
:
initial_slip_resistance(0.0),
linear_hardening_modulus(500),
hardening_parameter(1.4),
flag_perfect_plasticity(false)
{}



void HardeningLaw::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Hardening law's parameters");
  {
    prm.declare_entry("Initial slip resistance",
                      "0.0",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Linear hardening modulus",
                      "500",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Hardening parameter",
                      "1.4",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Perfect plasticity",
                      "false",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();
}



void HardeningLaw::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Hardening law's parameters");
  {
    initial_slip_resistance   = prm.get_double("Initial slip resistance");

    linear_hardening_modulus  = prm.get_double("Linear hardening modulus");

    hardening_parameter       = prm.get_double("Hardening parameter");

    flag_perfect_plasticity   = prm.get_bool("Perfect plasticity");
  }
  prm.leave_subsection();

  if (prm.subsection_path_exists(
        {"Vectorial microstress law's parameters"}))
  {
    prm.enter_subsection("Vectorial microstress law's parameters");
    {
      Assert(initial_slip_resistance ==
              prm.get_double("Initial slip resistance"),
            dealii::ExcMessage(
              "The initial slip resistance of the hardening law and of "
              "the vector-valued microstress has to match"));
    }
    prm.leave_subsection();
  }
}



DamageEvolution::DamageEvolution()
:
damage_evolution_model(DamageEvolutionModel::M1),
damage_accumulation_constant(1.0),
damage_decay_constant(0.0),
damage_decay_exponent(1.0),
endurance_limit(0.0),
flag_couple_microtraction_to_damage(true),
flag_couple_macrotraction_to_damage(true),
flag_set_damage_to_zero(false)
{}



void DamageEvolution::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Damage evolution parameters");
  {
    prm.declare_entry("Damage evolution model",
                      "M1",
                      dealii::Patterns::Selection("OrtizEtAl|M1"));

    prm.declare_entry("Damage accumulation constant",
                      "1.0",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Damage decay constant",
                      "0.0",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Damage decay exponent",
                      "1.0",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Endurance limit",
                      "0.0",
                      dealii::Patterns::Double(0.0));

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



void DamageEvolution::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Damage evolution parameters");
  {
    const std::string string_damage_evolution_model(
      prm.get("Damage evolution model"));

    if (string_damage_evolution_model == std::string("OrtizEtAl"))
    {
      damage_evolution_model = DamageEvolutionModel::OrtizEtAl;
    }
    else if (string_damage_evolution_model == std::string("M1"))
    {
      damage_evolution_model = DamageEvolutionModel::M1;
    }
    else
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Unexpected identifier for the damage evolution model."));
    }

    damage_accumulation_constant =
      prm.get_double("Damage accumulation constant");

    damage_decay_constant = prm.get_double("Damage decay constant");

    damage_decay_exponent = prm.get_double("Damage decay exponent");

    endurance_limit = prm.get_double("Endurance limit");

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
                      dealii::Patterns::Double(0.0));
  }
  prm.leave_subsection();
}



void ContactLawParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Contact law's parameters");
  {
    penalty_coefficient = prm.get_double("Penalty coefficient");
  }
  prm.leave_subsection();
}



ConstitutiveLawsParameters::ConstitutiveLawsParameters()
{}



void ConstitutiveLawsParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Constitutive laws' parameters");
  {
    HookeLawParameters::declare_parameters(prm);

    ScalarMicrostressLawParameters::declare_parameters(prm);

    VectorialMicrostressLawParameters::declare_parameters(prm);

    MicrotractionLawParameters::declare_parameters(prm);

    CohesiveLawParameters::declare_parameters(prm);

    ContactLawParameters::declare_parameters(prm);

    DegradationFunction::declare_parameters(prm);

    HardeningLaw::declare_parameters(prm);

    DamageEvolution::declare_parameters(prm);
  }
  prm.leave_subsection();

}



void ConstitutiveLawsParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Constitutive laws' parameters");
  {
    hooke_law_parameters.parse_parameters(prm);

    scalar_microstress_law_parameters.parse_parameters(prm);

    vectorial_microstress_law_parameters.parse_parameters(prm);

    microtraction_law_parameters.parse_parameters(prm);

    cohesive_law_parameters.parse_parameters(prm);

    contact_law_parameters.parse_parameters(prm);

    hardening_law_parameters.parse_parameters(prm);

    degradation_function_parameters.parse_parameters(prm);

    damage_evolution_parameters.parse_parameters(prm);
  }
  prm.leave_subsection();
}



CharacteristicQuantities::CharacteristicQuantities()
:
length(1.0),
time(1.0),
displacement(1.0),
stiffness(1.0),
slip_resistance(1.0),
strain(1.0),
stress(1.0),
resolved_shear_stress(stress),
macro_traction(stress),
micro_traction(1.0),
body_force(1.0),
dislocation_density(1.0)
{}



DimensionlessForm::DimensionlessForm()
:
dimensionless_numbers(4, 1.0),
flag_solve_dimensionless_problem(false)
{}



void DimensionlessForm::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Dimensionless formulation parameters");
  {
    prm.declare_entry("Solve the dimensionless problem",
                      "false",
                      dealii::Patterns::Bool());

    prm.enter_subsection("Reference parameters");
    {
      prm.declare_entry("Reference length value",
                        "1.0",
                        dealii::Patterns::Double(0.0));

      prm.declare_entry("Reference time value",
                        "1.0",
                        dealii::Patterns::Double(0.0));

      prm.declare_entry("Reference displacement value",
                        "1.0",
                        dealii::Patterns::Double(0.0));

      prm.declare_entry("Reference stiffness value",
                        "1.0",
                        dealii::Patterns::Double(0.0));

      prm.declare_entry("Reference slip resistance value",
                        "1.0",
                        dealii::Patterns::Double(0.0));
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void DimensionlessForm::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Dimensionless formulation parameters");
  {
    flag_solve_dimensionless_problem =
      prm.get_bool("Solve the dimensionless problem");

    if (flag_solve_dimensionless_problem)
    {
      prm.enter_subsection("Reference parameters");
      {
        characteristic_quantities.length =
          prm.get_double("Reference length value");

        characteristic_quantities.time =
          prm.get_double("Reference time value");

        characteristic_quantities.displacement =
          prm.get_double("Reference displacement value");

        characteristic_quantities.stiffness =
          prm.get_double("Reference stiffness value");

        characteristic_quantities.slip_resistance =
          prm.get_double("Reference slip resistance value");
      }
      prm.leave_subsection();
    }
  }
  prm.leave_subsection();
}



void DimensionlessForm::init(
  const RunTimeParameters::ConstitutiveLawsParameters &prm)
{
  if (!flag_solve_dimensionless_problem)
  {
    return;
  }

  const double &initial_slip_resistance =
    prm.vectorial_microstress_law_parameters.initial_slip_resistance;

  const double &defect_energy_index =
    prm.vectorial_microstress_law_parameters.defect_energy_index;

  const double &energetic_length_scale =
    prm.vectorial_microstress_law_parameters.energetic_length_scale;

  const double &linear_hardening_modulus =
    prm.hardening_law_parameters.linear_hardening_modulus;

  characteristic_quantities.strain =
    characteristic_quantities.displacement /
      characteristic_quantities.length;

  characteristic_quantities.stress =
    characteristic_quantities.stiffness *
      characteristic_quantities.strain;

  characteristic_quantities.resolved_shear_stress =
    characteristic_quantities.stress;

  characteristic_quantities.macro_traction =
    characteristic_quantities.stress;

  characteristic_quantities.micro_traction =
    initial_slip_resistance *
    std::pow(energetic_length_scale, defect_energy_index) /
    std::pow(characteristic_quantities.length,
             defect_energy_index - 1.0);

  characteristic_quantities.body_force =
    characteristic_quantities.stress /
      characteristic_quantities.length;

  characteristic_quantities.dislocation_density =
    1.0 / characteristic_quantities.length;

  dimensionless_numbers[0] =
    characteristic_quantities.length /
      characteristic_quantities.displacement;

  dimensionless_numbers[1] =
    linear_hardening_modulus /
      characteristic_quantities.slip_resistance;

  dimensionless_numbers[2] =
    initial_slip_resistance /
    characteristic_quantities.slip_resistance *
    std::pow(energetic_length_scale / characteristic_quantities.length,
             defect_energy_index);

  dimensionless_numbers[3] =
    characteristic_quantities.stress /
      characteristic_quantities.slip_resistance;
}



KrylovParameters::KrylovParameters()
:
solver_type(SolverType::CG),
relative_tolerance(1e-6),
absolute_tolerance(1e-8),
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
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Absolute tolerance",
                      "1e-8",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Maximum number of iterations",
                      "1000",
                      dealii::Patterns::Integer(1));
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

    n_max_iterations =
      prm.get_integer("Maximum number of iterations");

    AssertThrow(
      relative_tolerance > 0.0,
      dealii::ExcLowerRangeType<double>(relative_tolerance, 0.0));

    AssertThrow(
      absolute_tolerance > 0.0,
      dealii::ExcLowerRangeType<double>(absolute_tolerance, 0.0));

    AssertThrow(relative_tolerance > absolute_tolerance,
                dealii::ExcLowerRangeType<double>(
                  relative_tolerance , absolute_tolerance));
  }
  prm.leave_subsection();
}



NewtonRaphsonParameters::NewtonRaphsonParameters()
:
relative_tolerance(1e-6),
absolute_tolerance(1e-8),
step_tolerance(1e-8),
n_max_iterations(15),
flag_line_search(true)
{}



void NewtonRaphsonParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Newton-Raphson parameters");
  {
    prm.declare_entry("Relative tolerance of the residual",
                      "1e-6",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Absolute tolerance of the residual",
                      "1e-8",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Absolute tolerance of the step",
                      "1e-8",
                      dealii::Patterns::Double(0.));

    prm.declare_entry("Maximum number of iterations",
                      "15",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Line search algorithm",
                      "true",
                      dealii::Patterns::Bool());
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

    flag_line_search =
      prm.get_bool("Line search algorithm");

    AssertThrow(
      relative_tolerance > 0.0,
      dealii::ExcLowerRangeType<double>(relative_tolerance, 0.0));

    AssertThrow(
      absolute_tolerance > 0.0,
      dealii::ExcLowerRangeType<double>(absolute_tolerance, 0.0));

    AssertThrow(
      relative_tolerance > absolute_tolerance,
      dealii::ExcLowerRangeType<double>(relative_tolerance ,
                                        absolute_tolerance));
  }
  prm.leave_subsection();
}



LineSearchParameters::LineSearchParameters()
:
alpha(1e-4),
beta(0.9),
n_max_iterations(15)
{}



void LineSearchParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Line search parameters");
  {
    prm.declare_entry("Alpha condition constant",
                      "1e-4",
                      dealii::Patterns::Double(0.0,1.0));

    prm.declare_entry("Beta condition constant",
                      "0.9",
                      dealii::Patterns::Double(0.0,1.0));

    prm.declare_entry("Maximum number of iterations",
                      "15",
                      dealii::Patterns::Integer(0));
  }
  prm.leave_subsection();
}



void LineSearchParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Line search parameters");
  {
    alpha =
      prm.get_double("Alpha condition constant");

    beta =
      prm.get_double("Beta condition constant");

    n_max_iterations =
      prm.get_integer("Maximum number of iterations");

    AssertThrow(
      alpha > 0.0,
      dealii::ExcLowerRangeType<double>(alpha, 0.0));

    AssertThrow(
      beta > alpha,
      dealii::ExcLowerRangeType<double>(beta, alpha));

    AssertThrow(
      1.0 > beta,
      dealii::ExcMessage("The input value is outside the range (0,1)"));
  }
  prm.leave_subsection();
}



NonlinearSystemSolverParameters::NonlinearSystemSolverParameters(){}



void NonlinearSystemSolverParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  NewtonRaphsonParameters::declare_parameters(prm);

  LineSearchParameters::declare_parameters(prm);

  KrylovParameters::declare_parameters(prm);
}



void NonlinearSystemSolverParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  krylov_parameters.parse_parameters(prm);

  newton_parameters.parse_parameters(prm);

  line_search_parameters.parse_parameters(prm);
}



MonolithicAlgorithmParameters::MonolithicAlgorithmParameters()
:
monolithic_preconditioner(MonolithicPreconditioner::BuiltIn)
{}



void MonolithicAlgorithmParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Monolithic algorithm");
  {
    NonlinearSystemSolverParameters::declare_parameters(prm);

    prm.declare_entry("Monolithic preconditioner",
                      "built-in",
                      dealii::Patterns::Selection(
                        "built-in|block"));
  }
  prm.leave_subsection();
}



void MonolithicAlgorithmParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Monolithic algorithm");
  {
    monolithic_system_solver_parameters.parse_parameters(prm);

    const std::string string_monolithic_preconditioner(
                      prm.get("Monolithic preconditioner"));

    if (string_monolithic_preconditioner == std::string("built-in"))
    {
      monolithic_preconditioner = MonolithicPreconditioner::BuiltIn;
    }
    else if (string_monolithic_preconditioner == std::string("block"))
    {
      Assert(false, dealii::ExcMessage(
        "Block preconditioner has not been implemented"));

      monolithic_preconditioner = MonolithicPreconditioner::Block;
    }
    else
    {
      AssertThrow(false,
        dealii::ExcMessage(
          "Unexpected identifier for the monolithic preconditioner."));
    }
  }
  prm.leave_subsection();
}



StaggeredAlgorithmParameters::StaggeredAlgorithmParameters()
:
max_n_solution_loops(15),
flag_reset_trial_solution_at_micro_loop(true)
{}



void StaggeredAlgorithmParameters::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Staggered algorithm");
  {
    prm.enter_subsection("Linear momentum balance");
    {
      NonlinearSystemSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Pseudo-balance");
    {
      NonlinearSystemSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();

    prm.declare_entry("Maximum number of solution loops",
                      "15",
                      dealii::Patterns::Integer(2));

    prm.declare_entry("Reset trial solution at each micro loop",
                      "true",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();
}



void StaggeredAlgorithmParameters::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Staggered algorithm");
  {
    prm.enter_subsection("Linear momentum balance");
    {
      linear_momentum_solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Pseudo-balance");
    {
      pseudo_balance_solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();

    max_n_solution_loops =
      prm.get_integer("Maximum number of solution loops");

    flag_reset_trial_solution_at_micro_loop =
      prm.get_bool("Reset trial solution at each micro loop");
  }
  prm.leave_subsection();
}



SolverParameters::SolverParameters()
:
solution_algorithm(SolutionAlgorithm::Monolithic),
allow_decohesion(false),
boundary_conditions_at_grain_boundaries(
  BoundaryConditionsAtGrainBoundaries::Microfree),
logger_output_directory("results/default/"),
flag_skip_extrapolation(false),
flag_skip_extrapolation_at_extrema(false),
extrapolation_factor(1.0),
flag_zero_damage_during_loading_and_unloading(false),
flag_output_debug_fields(false),
print_sparsity_pattern(false),
verbose(false)
{}



void SolverParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("6. Solver parameters");
  {
    ConstitutiveLawsParameters::declare_parameters(prm);

    MonolithicAlgorithmParameters::declare_parameters(prm);

    StaggeredAlgorithmParameters::declare_parameters(prm);

    DimensionlessForm::declare_parameters(prm);

    prm.declare_entry("Solution algorithm",
                      "monolithic",
                      dealii::Patterns::Selection(
                        "monolithic|bouncing|embracing"));

    prm.declare_entry("Allow decohesion at grain boundaries",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Boundary conditions at grain boundaries",
                      "microfree",
                      dealii::Patterns::Selection(
                        "microhard|microfree|microtraction"));

    prm.declare_entry("Skip extrapolation of start value",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Skip extrapolation of start value at extrema",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Extrapolation factor",
                      "1.0",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Zero damage evolution during un- and loading",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Print sparsity pattern",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Output debug fields",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Verbose",
                      "false",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();
}



void SolverParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("6. Solver parameters");
  {
    constitutive_laws_parameters.parse_parameters(prm);

    const std::string string_solution_algorithm(
                      prm.get("Solution algorithm"));

    if (string_solution_algorithm == std::string("monolithic"))
    {
      solution_algorithm = SolutionAlgorithm::Monolithic;

      monolithic_algorithm_parameters.parse_parameters(prm);
    }
    else if (string_solution_algorithm == std::string("bouncing"))
    {
      solution_algorithm = SolutionAlgorithm::Bouncing;

      staggered_algorithm_parameters.parse_parameters(prm);
    }
    else if (string_solution_algorithm == std::string("embracing"))
    {
      solution_algorithm = SolutionAlgorithm::Embracing;

      staggered_algorithm_parameters.parse_parameters(prm);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage(
        "Unexpected identifier for the solution algorithm."));
    }

    dimensionless_form_parameters.parse_parameters(prm);

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
    {
      boundary_conditions_at_grain_boundaries =
        BoundaryConditionsAtGrainBoundaries::Microfree;
    }
    else if (string_boundary_conditions_at_grain_boundaries ==
              std::string("microtraction"))
    {
      boundary_conditions_at_grain_boundaries =
        BoundaryConditionsAtGrainBoundaries::Microtraction;
    }
    else
    {
      AssertThrow(false,
        dealii::ExcMessage(
          "Unexpected identifier for the boundary conditions at grain "
          "boundaries."));
    }

    flag_skip_extrapolation =
      prm.get_bool("Skip extrapolation of start value");

    flag_skip_extrapolation_at_extrema =
      prm.get_bool("Skip extrapolation of start value at extrema");

    extrapolation_factor = prm.get_double("Extrapolation factor");

    flag_zero_damage_during_loading_and_unloading =
      prm.get_bool("Zero damage evolution during un- and loading");

    print_sparsity_pattern = prm.get_bool("Print sparsity pattern");

    flag_output_debug_fields = prm.get_bool("Output debug fields");

    verbose = prm.get_bool("Verbose");
  }
  prm.leave_subsection();

  prm.enter_subsection("4. Output parameters");
  {
    logger_output_directory = prm.get("Graphical output directory");
  }
  prm.leave_subsection();

  dimensionless_form_parameters.init(
    constitutive_laws_parameters);
}



TemporalDiscretizationParameters::TemporalDiscretizationParameters()
:
start_time(0.0),
end_time(1.0),
time_step_size(0.25)
{}



void TemporalDiscretizationParameters::
declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("2. Temporal discretization parameters");
  {
    prm.declare_entry("Start time",
                      "0.0",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("End time",
                      "1.0",
                      dealii::Patterns::Double(0.0));

    prm.declare_entry("Time step size",
                      "1e-1",
                      dealii::Patterns::Double(0.0));
  }
  prm.leave_subsection();
}



void TemporalDiscretizationParameters::
parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("2. Temporal discretization parameters");
  {
    start_time            = prm.get_double("Start time");

    end_time              = prm.get_double("End time");

    time_step_size        = prm.get_double("Time step size");

    Assert(start_time >= 0.0,
            dealii::ExcLowerRangeType<double>(start_time, 0.0));

    Assert(end_time > start_time,
            dealii::ExcLowerRangeType<double>(end_time, start_time));

    Assert(time_step_size > 0,
            dealii::ExcLowerRangeType<double>(time_step_size, 0));

    Assert(end_time >= (start_time + time_step_size),
            dealii::ExcLowerRangeType<double>(
            end_time, start_time + time_step_size));
  }
  prm.leave_subsection();
}




SimpleLoading::SimpleLoading()
:
loading_type(LoadingType::Monotonic),
max_load(0.02),
min_load(0.00),
duration_monotonic_load(1.0),
n_steps_monotonic_load(20),
time_step_size_monotonic_load(5e-2),
duration_loading_and_unloading_phase(1.0),
n_steps_loading_and_unloading_phase(20),
time_step_size_loading_and_unloading_phase(5e-2),
n_cycles(1),
period(1.0),
n_steps_quarter_period(5),
time_step_size_cyclic_phase(5e-2),
flag_skip_unloading_phase(true)
{}



void SimpleLoading::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("2. Temporal discretization parameters");
  {
    prm.enter_subsection("Simple loading");
    {
      prm.declare_entry("Loading type",
                        "monotonic",
                        dealii::Patterns::Selection(
                          "monotonic|cyclic"));

      prm.declare_entry("Maximum load",
                        "1.0",
                        dealii::Patterns::Double(0.0));

      prm.declare_entry("Minimum load",
                        "0.0",
                        dealii::Patterns::Double(0.0));

      prm.enter_subsection("Monotonic load parameters");
      {
        prm.declare_entry("Duration of monotonic load",
                          "1.0",
                          dealii::Patterns::Double(0.0));

        prm.declare_entry("Number of steps during monotonic load",
                          "20",
                          dealii::Patterns::Integer(1));
      }
      prm.leave_subsection();

      prm.enter_subsection("Cyclic load parameters");
      {
        prm.declare_entry("Duration of un- and loading phase",
                          "1.0",
                          dealii::Patterns::Double(0.0));

        prm.declare_entry("Number of steps during un- and loading phase",
                          "20",
                          dealii::Patterns::Integer(1));

        prm.declare_entry("Period",
                          "1.0",
                          dealii::Patterns::Double(0.0));

        prm.declare_entry("Number of steps during a quarter cycle",
                          "5",
                          dealii::Patterns::Integer(1));

        prm.declare_entry("Number of cycles",
                          "1",
                          dealii::Patterns::Integer(0));

        prm.declare_entry("Skip unloading phase",
                          "true",
                          dealii::Patterns::Bool());
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void SimpleLoading::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("2. Temporal discretization parameters");
  {
    prm.enter_subsection("Simple loading");
    {
      prm.enter_subsection("Monotonic load parameters");
      {
        duration_monotonic_load =
          prm.get_double("Duration of monotonic load");

        n_steps_monotonic_load =
          prm.get_integer("Number of steps during monotonic load");

        time_step_size_monotonic_load =
          duration_monotonic_load / n_steps_monotonic_load;
      }
      prm.leave_subsection();

      prm.enter_subsection("Cyclic load parameters");
      {
        duration_loading_and_unloading_phase =
          prm.get_double("Duration of un- and loading phase");

        n_steps_loading_and_unloading_phase =
          prm.get_integer("Number of steps during un- and loading phase");

        time_step_size_loading_and_unloading_phase =
          duration_loading_and_unloading_phase /
            n_steps_loading_and_unloading_phase;

        n_cycles = prm.get_integer("Number of cycles");

        period = prm.get_double("Period");

        n_steps_quarter_period =
          prm.get_integer("Number of steps during a quarter cycle");

        time_step_size_cyclic_phase = period / 4. / n_steps_quarter_period;

        flag_skip_unloading_phase = prm.get_bool("Skip unloading phase");
      }
      prm.leave_subsection();

      const std::string string_loading_type(prm.get("Loading type"));

      if (string_loading_type == std::string("monotonic"))
      {
        loading_type = LoadingType::Monotonic;

        end_time = start_time + duration_monotonic_load;
      }
      else if (string_loading_type == std::string("cyclic"))
      {
        loading_type = LoadingType::Cyclic;

        end_time =
          start_time +
          ((flag_skip_unloading_phase) ? 1. : 2.) *
          duration_loading_and_unloading_phase + n_cycles * period;
      }
      else
      {
        std::ostringstream message;

        message << "Unexpected identifier for the loading type\n";

        AssertThrow(false, dealii::ExcMessage(message.str().c_str()));
      }

      max_load = prm.get_double("Maximum load");

      min_load = prm.get_double("Minimum load");

      start_of_cyclic_phase =
        start_time + duration_loading_and_unloading_phase;

      start_of_unloading_phase =
        start_of_cyclic_phase + n_cycles * period;

      Assert(
        duration_monotonic_load > 0.,
        dealii::ExcLowerRangeType<double>(duration_monotonic_load, 0.));

      Assert(
        period > 0.,
        dealii::ExcLowerRangeType<double>(period, 0.));

      std::ostringstream message;

      message << "The maximum load has to be greater than the minimum load\n";

      Assert(
        max_load > min_load,
        dealii::ExcMessage(message.str().c_str()));
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



double SimpleLoading::get_next_time_step_size(const unsigned int step_number) const
{
  if (loading_type == LoadingType::Monotonic)
  {
    return (time_step_size_monotonic_load);
  }
  else
  {
    if (step_number < n_steps_loading_and_unloading_phase)
    {
      return (time_step_size_loading_and_unloading_phase);
    }
    else if (step_number < (n_steps_loading_and_unloading_phase +
          4.0 * n_steps_quarter_period * n_cycles))
    {
      return (time_step_size_cyclic_phase);
    }
    else
    {
      return (time_step_size_loading_and_unloading_phase);
    }
  }

  std::ostringstream message;

  message << "Uppsala. Wie sind wir hier gelandet?\n";

  AssertThrow(false, dealii::ExcMessage(message.str().c_str()));
}



bool SimpleLoading::skip_extrapolation(const unsigned int step_number) const
{
  bool flag_skip_extrapolation = false;

  if (loading_type == LoadingType::Cyclic)
  {
    const bool start_of_cyclic_phase =
      step_number == n_steps_loading_and_unloading_phase;

    const unsigned int n_steps_cyclic_phase =
      4 * n_steps_quarter_period * n_cycles;

    const bool start_of_unloading_phase =
      step_number == (n_steps_loading_and_unloading_phase +
                        n_steps_cyclic_phase);

    const bool cyclic_phase =
      step_number > n_steps_loading_and_unloading_phase &&
      step_number < (n_steps_cyclic_phase +
                      n_steps_loading_and_unloading_phase);

    double tmp;

    const double effective_cycle_percentage =
      std::modf(
      (step_number - n_steps_loading_and_unloading_phase) /
      (4.0 * n_steps_quarter_period), &tmp);

    const bool extrema_of_cyclic_phase =
      effective_cycle_percentage == 0.25 ||
      effective_cycle_percentage == 0.75;

    if (start_of_cyclic_phase ||
        start_of_unloading_phase)
    {
      flag_skip_extrapolation = true;
    }
    else if (cyclic_phase && extrema_of_cyclic_phase)
    {
      flag_skip_extrapolation = true;
    }
  }

  return flag_skip_extrapolation;
}



SpatialDiscretizationBase::SpatialDiscretizationBase()
:
dim(2),
fe_degree_displacements(2),
fe_degree_slips(1),
n_global_refinements(0),
mapping_degree(1),
flag_apply_mapping_to_interior_cells(false)
{}



void SpatialDiscretizationBase::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("1. Spatial discretization parameters");
  {
    prm.declare_entry("Spatial dimension",
                      "2",
                      dealii::Patterns::Integer(2));

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
}



void SpatialDiscretizationBase::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("1. Spatial discretization parameters");
  {
    dim = prm.get_integer("Spatial dimension");

    fe_degree_displacements =
      prm.get_integer("FE's polynomial degree - Displacements");

    fe_degree_slips =
      prm.get_integer("FE's polynomial degree - Slips");

    n_global_refinements =
      prm.get_integer("Number of global refinements");

    mapping_degree =
      prm.get_integer("Mapping - Polynomial degree");

    flag_apply_mapping_to_interior_cells =
      prm.get_bool("Mapping - Apply to interior cells");

    AssertThrow(dim == 2 || dim == 3,
                dealii::ExcDimensionMismatch2(dim, 2, 3));
  }
  prm.leave_subsection();
}



Input::Input()
:
slips_normals_pathname(
  "input/crystal_structure/symmetric_double_slip_system/slip_normals"),
slips_directions_pathname(
  "input/crystal_structure/symmetric_double_slip_system/slip_directions"),
euler_angles_pathname("input/crystal_orientation/euler_angles_0_30")
{}



void Input::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("3. Input parameters");
  {
    prm.declare_entry(
      "Slip normals path name",
      "input/crystal_structure/symmetric_double_slip_system/slip_normals",
      dealii::Patterns::FileName());

    prm.declare_entry(
      "Slip directions path name",
      "input/crystal_structure/symmetric_double_slip_system/slip_directions",
      dealii::Patterns::FileName());

    prm.declare_entry(
      "Euler angles path name",
      "input/crystal_orientation/euler_angles_0_30",
      dealii::Patterns::FileName());
  }
  prm.leave_subsection();
}



void Input::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("3. Input parameters");
  {
    slips_normals_pathname    = prm.get("Slip normals path name");

    slips_directions_pathname = prm.get("Slip directions path name");

    euler_angles_pathname     = prm.get("Euler angles path name");
  }
  prm.leave_subsection();
}



Output::Output()
:
output_directory("results/default/"),
graphical_output_frequency(1),
terminal_output_frequency(1),
homogenization_output_frequency(1),
flag_output_damage_variable(false),
flag_output_fluctuations(false),
flag_output_dimensionless_quantities(false),
flag_store_checkpoint(false)
{}



void Output::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("4. Output parameters");
  {
    prm.declare_entry("Graphical output directory",
                      "results/default/",
                      dealii::Patterns::DirectoryName());

    prm.declare_entry("Graphical output frequency",
                      "1",
                      dealii::Patterns::Integer(0));

    prm.declare_entry("Terminal output frequency",
                      "1",
                      dealii::Patterns::Integer(0));

    prm.declare_entry("Output damage variable field",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Output dimensionless quantities",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Output fluctuations fields",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Store checkpoints",
                      "false",
                      dealii::Patterns::Bool());
  }
  prm.leave_subsection();
}



void Output::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("4. Output parameters");
  {
    output_directory = prm.get("Graphical output directory");

    if ((dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
        !fs::exists(output_directory + "paraview/"))
    {
      try
      {
        fs::create_directories(output_directory + "paraview/");
      }
      catch (std::exception &exc)
      {
        std::cerr
          << std::endl << std::endl
          << "----------------------------------------------------"
          << std::endl;

        std::cerr
          << "Exception in the creation of the output directory: "
          << std::endl
          << exc.what() << std::endl
          << "Aborting!" << std::endl
          << "----------------------------------------------------"
          << std::endl;

        std::abort();
      }
      catch (...)
      {
        std::cerr
          << std::endl << std::endl
          << "----------------------------------------------------"
          << std::endl;

        std::cerr
          << "Unknown exception in the creation of the output directory!"
          << std::endl
          << "Aborting!" << std::endl
          << "----------------------------------------------------"
          << std::endl;

        std::abort();
      }
    }

    graphical_output_frequency =
      prm.get_integer("Graphical output frequency");

    terminal_output_frequency =
      prm.get_integer("Terminal output frequency");

    flag_output_damage_variable =
      prm.get_bool("Output damage variable field");

    flag_output_fluctuations =
      prm.get_bool("Output fluctuations fields");

    flag_output_dimensionless_quantities =
      prm.get_bool("Output dimensionless quantities");

    flag_store_checkpoint =
      prm.get_bool("Store checkpoints");

    if ((dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
        !fs::exists(output_directory + "checkpoints/") &&
        flag_store_checkpoint)
    {
      try
      {
        fs::create_directories(output_directory + "checkpoints/");
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
}



Homogenization::Homogenization()
:
homogenization_frequency(1),
flag_compute_homogenized_quantities(false)
{}



void Homogenization::declare_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Homogenization");
  {
    prm.declare_entry("Compute homogenized quantities",
                      "false",
                      dealii::Patterns::Bool());

    prm.declare_entry("Homogenization frequency",
                      "1",
                      dealii::Patterns::Integer(0));
  }
  prm.leave_subsection();
}



void Homogenization::parse_parameters(
  dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Homogenization");
  {
    homogenization_frequency =
      prm.get_integer("Homogenization frequency");

    flag_compute_homogenized_quantities =
      prm.get_bool("Compute homogenized quantities");
  }
  prm.leave_subsection();
}



BasicProblem::BasicProblem()
:
verbose(true)
{}



BasicProblem::BasicProblem(
  const std::string &parameter_filename)
:
BasicProblem()
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



void BasicProblem::declare_parameters(dealii::ParameterHandler &prm)
{
  SpatialDiscretizationBase::declare_parameters(prm);

  TemporalDiscretizationParameters::declare_parameters(prm);

  Input::declare_parameters(prm);

  Output::declare_parameters(prm);

  SolverParameters::declare_parameters(prm);

  prm.enter_subsection("5. Postprocessing parameters");
  {
    Homogenization::declare_parameters(prm);
  }
  prm.leave_subsection();

  prm.declare_entry("Verbose",
                    "false",
                    dealii::Patterns::Bool());
}



void BasicProblem::parse_parameters(dealii::ParameterHandler &prm)
{
  spatial_discretization.parse_parameters(prm);

  temporal_discretization_parameters.parse_parameters(prm);

  input.parse_parameters(prm);

  output.parse_parameters(prm);

  solver_parameters.parse_parameters(prm);

  prm.enter_subsection("5. Postprocessing parameters");
  {
    homogenization.parse_parameters(prm);
  }
  prm.leave_subsection();

  verbose = prm.get_bool("Verbose");
}



InfiniteStripProblem::InfiniteStripProblem()
:
BasicProblem(),
control_type(ControlType::Displacement),
height(1.),
n_elements_in_y_direction(100),
n_equal_sized_crystals(1)
{}



InfiniteStripProblem::InfiniteStripProblem(
  const std::string &parameter_filename)
:
InfiniteStripProblem()
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

  // Store a copy of the parameter file in the output folder
  {
    std::string output_folder_path;

    prm.enter_subsection("4. Output parameters");
    {
      output_folder_path = prm.get("Graphical output directory");
    }
    prm.leave_subsection();

    prm.print_parameters(
      output_folder_path + "parameter_file.prm",
      dealii::ParameterHandler::OutputStyle::ShortPRM);
  }
}



void InfiniteStripProblem::declare_parameters(dealii::ParameterHandler &prm)
{
  BasicProblem::declare_parameters(prm);

  SimpleLoading::declare_parameters(prm);

  prm.enter_subsection("1. Spatial discretization parameters");
  {
    prm.enter_subsection("Problem specific parameters");
    {
      prm.declare_entry("Height of the strip",
                        "1.0",
                        dealii::Patterns::Double(0.));

      prm.declare_entry("Number of elements in y-direction",
                        "100",
                        dealii::Patterns::Integer(1));

      prm.declare_entry("Number of equally sized crystals in y-direction",
                        "1",
                        dealii::Patterns::Integer(1));
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();

  prm.enter_subsection("0. Infinite strip parameters (Extended in 1. and 2.)");
  {
    prm.enter_subsection("Loading parameters");
    {
      prm.declare_entry("Control type",
                        "displacement_control",
                        dealii::Patterns::Selection(
                          "load_control|displacement_control"));
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void InfiniteStripProblem::parse_parameters(dealii::ParameterHandler &prm)
{
  BasicProblem::parse_parameters(prm);

  simple_loading.parse_parameters(prm);

  prm.enter_subsection("1. Spatial discretization parameters");
  {
    prm.enter_subsection("Problem specific parameters");
    {
      height = prm.get_double("Height of the strip");

      n_elements_in_y_direction =
        prm.get_double("Number of elements in y-direction");

      n_equal_sized_crystals =
        prm.get_double("Number of equally sized crystals in y-direction");

      AssertThrow(
        height > 0.0,
        dealii::ExcLowerRangeType<double>(height, 0.0));
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();

  prm.enter_subsection("0. Infinite strip parameters (Extended in 1. and 2.)");
  {

    prm.enter_subsection("Loading parameters");
    {
      const std::string string_control_type(prm.get("Control type"));

      if (string_control_type == std::string("load_control"))
      {
        control_type = ControlType::Load;
      }
      else if (string_control_type == std::string("displacement_control"))
      {
        control_type = ControlType::Displacement;
      }
      else
      {
        std::ostringstream message;

        message << "Unexpected enum identifier for the control type\n";

        AssertThrow(false, dealii::ExcMessage(message.str().c_str()));
      }
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



SemicoupledParameters::SemicoupledParameters()
:
BasicProblem(),
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

  prm.enter_subsection("4. Output parameters");
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
  BasicProblem::declare_parameters(prm);

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
  BasicProblem::parse_parameters(prm);

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