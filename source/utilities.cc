
#include <gCP/utilities.h>



namespace gCP
{



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

  for (const auto& pair : data_map)
  {
    if (pair.second.second)
      output_filepath << std::fixed << std::scientific;
    else
      output_filepath << std::defaultfloat;

    output_filepath << std::setw(pair.first.length()) << std::right
                    << std::setprecision(6)
                    << pair.second.first << std::string(5, ' ');
  }

  output_filepath << std::endl;
}



void Logger::log_headers_to_terminal()
{
  pcout << std::string(2, ' ');

  for (const auto& pair : data_map)
  {
    if (pair.second.second)
      pcout << std::fixed << std::scientific;
    else
      pcout << std::defaultfloat;

    pcout << std::setw(pair.first.length()) << std::right
          << std::setprecision(6)
          << pair.first << std::string(5, ' ');
  }

  pcout << std::endl;
}



void Logger::log_values_to_terminal()
{
  pcout << std::string(2, ' ');

  for (const auto& pair : data_map)
  {
    if (pair.second.second)
      pcout << std::fixed << std::scientific;
    else
      pcout << std::defaultfloat;

    pcout << std::setw(pair.first.length()) << std::right
          << std::setprecision(6)
          << pair.second.first << std::string(5, ' ');
  }

  pcout << std::endl;
}



void Logger::add_break(const std::string message)
{
  output_filepath << std::endl
                  << message
                  << std::endl << std::endl
                  << std::string(2, ' ');

  for (const auto& pair : data_map)
    output_filepath << pair.first << std::string(5, ' ');

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

