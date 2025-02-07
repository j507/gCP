
#include <gCP/utilities.h>

#include <cmath>

namespace gCP
{



namespace internal
{



void handle_std_excepction(
  const std::exception  &exception,
  const std::string     string)
{
  std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl
            << "Exception in the " << string << ": " << std::endl
            << exception.what() << std::endl << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;

  std::abort();
}



void handle_unknown_exception(const std::string string)
{
  std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl
            << "Exception in the " << string << ": " << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;

  std::abort();
}



} // internal



namespace Utilities
{



double macaulay_brackets(const double value)
{
  AssertIsFinite(value);

  return (value > 0) ? value : 0.0;
}



double signum_function(const double value)
{
  AssertIsFinite(value);

  return (0.0 < value) - (value < 0.0);
}



double sigmoid_function(
  const double value,
  const double parameter,
  const RunTimeParameters::RegularizationFunction function_type)
{
  AssertIsFinite(value);
  AssertIsFinite(parameter);

  double function_value = 0.;

  switch (function_type)
  {
    case RunTimeParameters::RegularizationFunction::Atan:
      {
        function_value =
          2.0 / M_PI * std::atan(M_PI / 2.0 * value / parameter);
      }
      break;
    case RunTimeParameters::RegularizationFunction::Sqrt:
      {
        function_value =
          value / std::sqrt(value * value + parameter * parameter);
      }
      break;
    case RunTimeParameters::RegularizationFunction::Gd:
      {
        function_value =
          2.0 / M_PI * std::atan(std::sinh(
            M_PI / 2.0 * value / parameter));
      }
      break;
    case RunTimeParameters::RegularizationFunction::Tanh:
      {
        function_value = std::tanh(value / parameter);
      }
      break;
    case RunTimeParameters::RegularizationFunction::Erf:
      {
        function_value =
          std::erf(std::sqrt(M_PI) / 2.0 * value / parameter);
      }
      break;
    default:
      {
        AssertThrow(false,
          dealii::ExcMessage(
            "The given regularization function is not currently "
            "implemented."));
      }
      break;
  }

  AssertIsFinite(function_value);

  return function_value;
}



double sigmoid_function_derivative(
  const double value,
  const double parameter,
  const RunTimeParameters::RegularizationFunction function_type)
{
  AssertIsFinite(value);
  AssertIsFinite(parameter);

  double derivative_value = 0.;

  switch (function_type)
  {
    case RunTimeParameters::RegularizationFunction::Atan:
      {
        derivative_value =
          parameter / (parameter * parameter +
            M_PI * M_PI * value * value / 4.);
      }
      break;
    case RunTimeParameters::RegularizationFunction::Sqrt:
      {
        derivative_value =
          parameter * parameter / std::pow(
            value * value + parameter * parameter, 1.5);
      }
      break;
    case RunTimeParameters::RegularizationFunction::Gd:
      {
        derivative_value =
          1.0 / std::cosh(M_PI / 2. * value / parameter) / parameter;
      }
      break;
    case RunTimeParameters::RegularizationFunction::Tanh:
      {
        derivative_value =
          std::pow(1.0 / std::cosh(value / parameter), 2) / parameter;
      }
      break;
    case RunTimeParameters::RegularizationFunction::Erf:
      {
        derivative_value =
          1. / parameter * std::exp(
            -M_PI * value * value / parameter / parameter / 4.);
      }
      break;
    default:
      {
        AssertThrow(false,
          dealii::ExcMessage(
            "The given regularization function is not currently "
            "implemented."));
      }
      break;
  }

  AssertIsFinite(derivative_value);

  return derivative_value;
}



bool files_are_identical(
  const std::string& first_filename,
  const std::string& second_filename)
{
  std::ifstream first_file(
    first_filename,
    std::ifstream::binary | std::ifstream::ate);

  std::ifstream second_file(
    second_filename,
    std::ifstream::binary | std::ifstream::ate);

  if (first_file.fail() || second_file.fail())
  {
    return false; //file problem
  }

  if (first_file.tellg() != second_file.tellg())
  {
    return false; //size mismatch
  }

  //seek back to beginning and use std::equal to compare contents
  first_file.seekg(0, std::ifstream::beg);
  second_file.seekg(0, std::ifstream::beg);

  return std::equal(
    std::istreambuf_iterator<char>(first_file.rdbuf()),
    std::istreambuf_iterator<char>(),
    std::istreambuf_iterator<char>(second_file.rdbuf()));
}


Logger::Logger(const std::string output_filepath)
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
output_filepath(output_filepath)
{}



void Logger::declare_column(
  const std::string column_name,
  const Format format)
{
  for (const auto& pair : data_map)
  {
    AssertThrow(
      pair.first != column_name,
      dealii::ExcMessage("This column was already defined!"))
  }

  data_map[column_name] = std::make_pair(0.0, format);

  keys_order.push_back(column_name);
}



void Logger::update_value(const std::string column_name,
                          const double value)
{
  AssertIsFinite(value);

  data_map[column_name].first = value;
}



void Logger::log_to_file()
{
  output_filepath << std::string(2, ' ');

  for (const auto& key : keys_order)
  {
    const double value = data_map[key].first;

    const Format format = data_map[key].second;

    const unsigned int precision = 6;

    switch (format)
    {
    case Format::Integer:
      output_filepath << std::setw(key.length()) << std::right
        << std::fixed << (unsigned int)value;
      break;

    case Format::Decimal:
      output_filepath << std::setw(key.length()) << std::right
        << std::fixed << std::setprecision(precision) << value;
      break;

    case Format::Scientific:
      output_filepath << std::setw(key.length()) << std::right
        << std::fixed << std::scientific << std::setprecision(precision)
        << value;
      break;

    default:
      break;
    }

    output_filepath << std::string(3, ' ');
  }

  output_filepath << std::endl;
}



void Logger::log_headers_to_terminal()
{
  pcout << std::string(2, ' ');

  const unsigned int precision = 2;

  for (const auto& key : keys_order)
  {
    const unsigned int width =
      data_map[key].second == Format::Scientific ?
        (precision + 6) : key.length();

    pcout
      << std::setw(width)
      << std::right
      << key
      << std::string(3, ' ');
  }

  pcout << std::endl;
}



void Logger::log_values_to_terminal()
{
  pcout << std::string(2, ' ');

  for (const auto& key : keys_order)
  {
    const double value = data_map[key].first;

    const Format format = data_map[key].second;

    const unsigned int precision = 2;

    switch (format)
    {
    case Format::Integer:
      pcout << std::setw(key.length()) << std::right << std::fixed
        << (unsigned int)value;
      break;

    case Format::Decimal:
      pcout << std::setw(key.length()) << std::right << std::fixed
        << std::setprecision(precision) << value;
      break;

    case Format::Scientific:
      pcout << std::setw(key.length()) << std::right << std::fixed
        << std::scientific << std::setprecision(precision) << value;
      break;

    default:
      break;
    }

    pcout << std::string(3, ' ');
  }

  pcout << std::endl;
}



void Logger::log_to_all(const std::string message)
{
  output_filepath <<  message << "\n";

  pcout << message << "\n";
}



void Logger::add_break(const std::string message)
{
  output_filepath << std::endl
                  << message
                  << std::endl << std::endl
                  << std::string(2, ' ');

  for (const auto& key : keys_order)
    output_filepath << key << std::string(4, ' ');

  output_filepath << std::endl;
}



std::string get_fullmatrix_as_string(
  const dealii::FullMatrix<double>  fullmatrix,
  const unsigned int                offset,
  const unsigned int                width,
  const unsigned int                precision,
  const bool                        scientific)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision);

  if (scientific)
    ss << std::scientific;

  for (unsigned int i = 0; i < fullmatrix.m(); ++i)
  {
    for (unsigned int j = 0; j < fullmatrix.n(); ++j)
      ss << std::setw(width) << fullmatrix[i][j];

    if (i != (fullmatrix.m()-1))
      ss << "\n" << std::string(offset, ' ');
  }

  return ss.str();
}



} // namespace Utilities



} // namespace gCP

