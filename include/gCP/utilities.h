#ifndef INCLUDE_UTILITIES_H_
#define INCLUDE_UTILITIES_H_

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <fstream>
#include <map>
#include <string>


namespace gCP
{



namespace Utilities
{



class Logger
{
public:
  Logger(const std::string output_filepath);

  void declare_column(const std::string column_name);

  void set_scientific(const std::string column_name,
                      const bool        boolean);

  void update_value(const std::string column_name,
                    const double      value);

  void log_to_file();

  void add_break(const std::string message);

private:
  std::ofstream                                   output_filepath;

  std::map<std::string, std::pair<double, bool>>  data_map;
};



template <int dim>
std::string get_tensor_as_string(
  const dealii::Tensor<1,dim> tensor,
  const unsigned int          width = 15,
  const unsigned int          precision = 6)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision); // set # places after decimal

  ss << std::setw(width) << tensor[0]
      << std::setw(width) << tensor[1]
      << std::setw(width) << tensor[2];

  return ss.str();
}



template <int dim>
std::string get_tensor_as_string(
  const dealii::Tensor<2,dim> tensor,
  const unsigned int          offset = 0,
  const unsigned int          width = 15,
  const unsigned int          precision = 6)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision); // set # places after decimal

  ss << std::setw(width) << tensor[0][0]
      << std::setw(width) << tensor[0][1]
      << std::setw(width) << tensor[0][2] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tensor[1][0]
      << std::setw(width) << tensor[1][1]
      << std::setw(width) << tensor[1][2] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tensor[2][0]
      << std::setw(width) << tensor[2][1]
      << std::setw(width) << tensor[2][2];

  return ss.str();
}



template <int dim>
std::string get_tensor_as_string(
  const dealii::SymmetricTensor<2,dim>  tensor,
  const unsigned int                    offset = 0,
  const unsigned int                    width = 15,
  const unsigned int                    precision = 6)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision);

  ss << std::setw(width) << tensor[0][0]
      << std::setw(width) << tensor[0][1]
      << std::setw(width) << tensor[0][2] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tensor[1][0]
      << std::setw(width) << tensor[1][1]
      << std::setw(width) << tensor[1][2] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tensor[2][0]
      << std::setw(width) << tensor[2][1]
      << std::setw(width) << tensor[2][2];

  return ss.str();
}



template <int dim>
std::string print_tetrad(
  const dealii::SymmetricTensor<4,dim> tetrad,
  const unsigned int                    offset = 0,
  const unsigned int                    width = 15,
  const unsigned int                    precision = 6)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision);

  ss << std::setw(width) << tetrad[0][0][0][0]
      << std::setw(width) << tetrad[0][0][1][1]
      << std::setw(width) << tetrad[0][0][2][2]
      << std::setw(width) << tetrad[0][0][1][2]
      << std::setw(width) << tetrad[0][0][0][2]
      << std::setw(width) << tetrad[0][0][0][1] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tetrad[1][1][0][0]
      << std::setw(width) << tetrad[1][1][1][1]
      << std::setw(width) << tetrad[1][1][2][2]
      << std::setw(width) << tetrad[1][1][1][2]
      << std::setw(width) << tetrad[1][1][0][2]
      << std::setw(width) << tetrad[1][1][0][1] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tetrad[2][2][0][0]
      << std::setw(width) << tetrad[2][2][1][1]
      << std::setw(width) << tetrad[2][2][2][2]
      << std::setw(width) << tetrad[2][2][1][2]
      << std::setw(width) << tetrad[2][2][0][2]
      << std::setw(width) << tetrad[2][2][0][1] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tetrad[1][2][0][0]
      << std::setw(width) << tetrad[1][2][1][1]
      << std::setw(width) << tetrad[1][2][2][2]
      << std::setw(width) << tetrad[1][2][1][2]
      << std::setw(width) << tetrad[1][2][0][2]
      << std::setw(width) << tetrad[1][2][0][1] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tetrad[0][2][0][0]
      << std::setw(width) << tetrad[0][2][1][1]
      << std::setw(width) << tetrad[0][2][2][2]
      << std::setw(width) << tetrad[0][2][1][2]
      << std::setw(width) << tetrad[0][2][0][2]
      << std::setw(width) << tetrad[0][2][0][1] << std::endl
      << std::string(offset, ' ')
      << std::setw(width) << tetrad[0][1][0][0]
      << std::setw(width) << tetrad[0][1][1][1]
      << std::setw(width) << tetrad[0][1][2][2]
      << std::setw(width) << tetrad[0][1][1][2]
      << std::setw(width) << tetrad[0][1][0][2]
      << std::setw(width) << tetrad[0][1][0][1];

  return ss.str();
}


}  // Utilities



} // gCP



#endif /* INCLUDE_UTILITIES_H_ */