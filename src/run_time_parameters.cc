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



SolverParameters::SolverParameters()
:
nonlinear_tolerance(1e-4),
n_max_nonlinear_iterations(1000),
krylov_relative_tolerance(1e-6),
krylov_absolute_tolerance(1e-8),
n_max_krylov_iterations(1000)
{}



void SolverParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Nonlinear solver's parameters");
  {
    prm.declare_entry("Tolerance of the nonlinear solver",
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
  }
  prm.leave_subsection();

  prm.declare_entry("Verbose",
                    "false",
                    dealii::Patterns::Bool());
}



void SolverParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Nonlinear solver's parameters");
  {
    nonlinear_tolerance =
      prm.get_double("Tolerance of the nonlinear solver");
    AssertThrow(nonlinear_tolerance > 0,
                dealii::ExcLowerRange(nonlinear_tolerance, 0));

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
  }
  prm.leave_subsection();

  verbose = prm.get_bool("Verbose");
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
graphical_output_directory("./"),
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
                      "./",
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