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


Parameters::Parameters()
:
dim(2),
mapping_degree(1),
mapping_interior_cells(false),
fe_degree_displacements(2),
fe_degree_slips(1),
regularization_function(RegularizationFunction::Tanh),
relative_tolerance(1e-6),
absolute_tolerance(1e-8),
n_maximum_iterations(1000),
graphical_output_frequency(1),
terminal_output_frequency(1),
graphical_output_directory("./"),
verbose(true)
{}



Parameters::Parameters(
  const std::string &parameter_filename)
:
Parameters()
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



void Parameters::declare_parameters(dealii::ParameterHandler &prm)
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

    prm.declare_entry("FE's polynomial degree - Displacements",
                      "2",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("FE's polynomial degree - Slips",
                      "1",
                      dealii::Patterns::Integer(1));
  }
  prm.leave_subsection();

  prm.declare_entry("Regularization function",
                    "tanh",
                    dealii::Patterns::Selection("tanh|power-law"));

  prm.declare_entry("Verbose",
                    "false",
                    dealii::Patterns::Bool());

  prm.enter_subsection("Solver parameters");
  {
    prm.declare_entry("Maximum number of iterations",
                      "1000",
                      dealii::Patterns::Integer(1));

    prm.declare_entry("Relative tolerance",
                      "1e-6",
                      dealii::Patterns::Double());

    prm.declare_entry("Absolute tolerance",
                      "1e-8",
                      dealii::Patterns::Double());
  }
  prm.leave_subsection();

  prm.enter_subsection("Constitutive laws' parameters");
  {
    HookeLawParameters::declare_parameters(prm);
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



void Parameters::parse_parameters(dealii::ParameterHandler &prm)
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

    fe_degree_displacements = prm.get_integer("FE's polynomial degree - Displacements");
    AssertThrow(fe_degree_displacements > 0,
                dealii::ExcLowerRange(fe_degree_displacements, 0));

    fe_degree_slips = prm.get_integer("FE's polynomial degree - Slips");
    AssertThrow(fe_degree_slips > 0,
                dealii::ExcLowerRange(fe_degree_slips, 0));
  }
  prm.leave_subsection();

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

  verbose = prm.get_bool("Verbose");


  prm.enter_subsection("Solver parameters");
  {
    n_maximum_iterations = prm.get_integer("Maximum number of iterations");
    AssertThrow(n_maximum_iterations > 0,
                dealii::ExcLowerRange(n_maximum_iterations, 0));

    relative_tolerance = prm.get_double("Relative tolerance");
    AssertThrow(relative_tolerance > 0,
                dealii::ExcLowerRange(relative_tolerance, 0));

    absolute_tolerance = prm.get_double("Absolute tolerance");
    AssertThrow(relative_tolerance > absolute_tolerance,
                dealii::ExcLowerRangeType<double>(
                  relative_tolerance , absolute_tolerance));
  }
  prm.leave_subsection();

  prm.enter_subsection("Constitutive laws' parameters");
  {
    hooke_law_parameters.parse_parameters(prm);
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