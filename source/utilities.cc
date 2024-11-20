
#include <gCP/utilities.h>



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



Logger::Logger(const std::string output_filepath)
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
output_filepath(output_filepath)
{}



void Logger::declare_column(const std::string column_name)
{
  for (const auto& pair : data_map)
      AssertThrow(pair.first != column_name,
                  dealii::ExcMessage("This column was already defined!"))

  data_map[column_name] = std::make_pair(0.0, false);

  keys_order.push_back(column_name);
}



void Logger::update_value(const std::string column_name,
                          const double value)
{
  AssertIsFinite(value);

  data_map[column_name].first = value;
}



void Logger::set_scientific(const std::string column_name,
                            const bool boolean)
{
  data_map[column_name].second = boolean;
}



void Logger::log_to_file()
{
  output_filepath << std::string(2, ' ');

  for (const auto& key : keys_order)
  {
    if (data_map[key].second)
      output_filepath << std::fixed << std::scientific;
    else
      output_filepath << std::defaultfloat;

    const unsigned int precision = 6;

    output_filepath << std::setw(key.length()) << std::right
                    << std::setprecision(precision)
                    << data_map[key].first << std::string(4, ' ');
  }

  output_filepath << std::endl;
}



void Logger::log_headers_to_terminal()
{
  pcout << std::string(2, ' ');

  for (const auto& key : keys_order)
  {
    if (data_map[key].second)
      pcout << std::fixed << std::scientific;
    else
      pcout << std::defaultfloat;

    const unsigned int precision = 2;

    pcout
      << std::setw( (data_map[key].second) ? (precision + 6) : key.length())
      << std::right
      << std::setprecision(precision)
      << key
      << std::string(4, ' ');
  }

  pcout << std::endl;
}



void Logger::log_values_to_terminal()
{
  pcout << std::string(2, ' ');

  for (const auto& key : keys_order)
  {
    if (data_map[key].second)
      pcout << std::fixed << std::scientific;
    else
      pcout << std::defaultfloat;

    const unsigned int precision = 1;

    pcout << std::setw(key.length()) << std::right
          << std::setprecision(precision)
          << data_map[key].first << std::string(4, ' ');
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

