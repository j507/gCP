#ifndef INCLUDE_UTILITIES_H_
#define INCLUDE_UTILITIES_H_

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/full_matrix.h>

#include <fstream>
#include <map>
#include <string>


namespace gCP
{



namespace Utilities
{



template<int dim>
void update_ghost_material_ids(dealii::DoFHandler<dim> &dof_handler)
{
  const unsigned int spacedim = dim;

  auto pack = [](
    const typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator &cell)->
      typename dealii::DoFHandler<dim,dim>::active_fe_index_type
      {
        return cell->material_id();
      };

  auto unpack = [&dof_handler](
    const typename dealii::DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    const typename dealii::DoFHandler<dim,dim>::active_fe_index_type      active_fe_index)->
      void
      {
        cell->set_material_id(active_fe_index);
      };

  dealii::GridTools::exchange_cell_data_to_ghosts<
    typename dealii::DoFHandler< dim, dim >::active_fe_index_type,
    dealii::DoFHandler<dim, spacedim>>(
      dof_handler,
      pack,
      unpack);
}



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

  void log_headers_to_terminal();

  void log_values_to_terminal();

  void add_break(const std::string message);

private:
  dealii::ConditionalOStream                      pcout;

  std::ofstream                                   output_filepath;

  std::map<std::string, std::pair<double, bool>>  data_map;

  std::vector<std::string>                        keys_order;
};



template <int dim>
std::string get_tensor_as_string(
  const dealii::Tensor<1,dim> tensor,
  const unsigned int          width = 15,
  const unsigned int          precision = 6,
  const bool                  scientific = false)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision); // set # places after decimal

  if (scientific)
    ss << std::scientific;

  for (unsigned int i = 0; i < dim; ++i)
    ss << std::setw(width) << tensor[i];

  return ss.str();
}



template <int dim>
std::string get_tensor_as_string(
  const dealii::Tensor<2,dim> tensor,
  const unsigned int          offset = 0,
  const unsigned int          width = 15,
  const unsigned int          precision = 6,
  const bool                  scientific = false)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision); // set # places after decimal

  if (scientific)
    ss << std::scientific;

  for (unsigned int i = 0; i < dim; ++i)
  {
    for (unsigned int j = 0; j < dim; ++j)
      ss << std::setw(width) << tensor[i][j];

    if (i != (dim-1))
      ss << "\n" << std::string(offset, ' ');
  }

  return ss.str();
}



template <int dim>
std::string get_tensor_as_string(
  const dealii::SymmetricTensor<2,dim>  tensor,
  const unsigned int                    offset = 0,
  const unsigned int                    width = 15,
  const unsigned int                    precision = 6,
  const bool                            scientific = false)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision);

  if (scientific)
    ss << std::scientific;

  for (unsigned int i = 0; i < dim; ++i)
  {
    for (unsigned int j = 0; j < dim; ++j)
      ss << std::setw(width) << tensor[i][j];

    if (i != (dim-1))
      ss << "\n" << std::string(offset, ' ');
  }

  return ss.str();
}



template <int dim>
std::string print_tetrad(
  const dealii::SymmetricTensor<4,dim> tetrad,
  const unsigned int                    offset = 0,
  const unsigned int                    width = 15,
  const unsigned int                    precision = 6,
  const bool                            scientific = false)
{
  std::stringstream ss;
  ss << std::fixed << std::left << std::showpos;
  ss.precision(precision);

  if (scientific)
    ss << std::scientific;

  std::vector<std::pair<unsigned int, unsigned int>> voigt_indices;

  switch (dim)
  {
    case 2:
      {
        voigt_indices.resize(3);

        voigt_indices[0] = std::make_pair<unsigned int, unsigned int>(0,0);
        voigt_indices[1] = std::make_pair<unsigned int, unsigned int>(1,1);
        voigt_indices[2] = std::make_pair<unsigned int, unsigned int>(0,1);
      }
      break;
    case 3:
      {
        voigt_indices.resize(6);

        voigt_indices[0] = std::make_pair<unsigned int, unsigned int>(0,0);
        voigt_indices[1] = std::make_pair<unsigned int, unsigned int>(1,1);
        voigt_indices[2] = std::make_pair<unsigned int, unsigned int>(2,2);
        voigt_indices[3] = std::make_pair<unsigned int, unsigned int>(1,2);
        voigt_indices[4] = std::make_pair<unsigned int, unsigned int>(0,2);
        voigt_indices[5] = std::make_pair<unsigned int, unsigned int>(0,1);
      }
      break;
    default:
      AssertThrow(false, dealii::ExcIndexRange(dim,2,3));
      break;
  }

  for (unsigned int i = 0; i < voigt_indices.size(); ++i)
  {
    for (unsigned int j = 0; j < voigt_indices.size(); ++j)
      ss << std::setw(width)
         << tetrad[voigt_indices[i].first][voigt_indices[i].second][voigt_indices[j].first][voigt_indices[j].second];

    if (i != (voigt_indices.size()-1))
      ss << "\n" << std::string(offset, ' ');
  }

  return ss.str();
}



std::string get_fullmatrix_as_string(
  const dealii::FullMatrix<double>  fullmatrix,
  const unsigned int                offset = 0,
  const unsigned int                width = 15,
  const unsigned int                precision = 6,
  const bool                        scientific = false);



}  // Utilities



} // gCP



#endif /* INCLUDE_UTILITIES_H_ */