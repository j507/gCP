#ifndef INCLUDE_FINITE_DOMAINS_H_
#define INCLUDE_FINITE_DOMAINS_H_



#include <deal.II/distributed/tria.h>



namespace gCP
{



namespace FiniteDomains
{



void plate_with_notches(
  dealii::parallel::distributed::Triangulation<2> &triangulation,
  const unsigned int n_slices,
  const double depth);



void plate_with_notches(
  dealii::parallel::distributed::Triangulation<3> &triangulation,
  const unsigned int n_slices,
  const double depth);



void polycristalline_unit_square(
  dealii::parallel::distributed::Triangulation<2> &triangulation,
  const std::string msh_file_path);



void polycristalline_unit_square(
  dealii::parallel::distributed::Triangulation<3> &triangulation,
  const std::string msh_file_path);



} // namespace FiniteDomains



} // namespace gCP



#endif /* INCLUDE_FINITE_DOMAINS_H_ */
