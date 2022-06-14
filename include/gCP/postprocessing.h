#ifndef INCLUDE_POSTPROCESSING_H_
#define INCLUDE_POSTPROCESSING_H_

#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/base/table_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/base/utilities.h>


namespace gCP
{



namespace Postprocessing
{



template <int dim>
class SimpleShear
{
public:
  SimpleShear(
    std::shared_ptr<FEField<dim>>         &fe_field,
    std::shared_ptr<dealii::Mapping<dim>> &mapping,
    const double                          shear_at_upper_boundary,
    const dealii::types::boundary_id      upper_boundary_id,
    const double                          width);

  void init(
    std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain,
    std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law);

  void compute_data(const double time);

  void output_data_to_file(std::ostream &file) const;

  const std::vector<dealii::LinearAlgebraTrilinos::MPI::Vector> &get_data() const;

  const dealii::DoFHandler<dim>& get_dof_handler() const;

private:
  dealii::TableHandler                                    table_handler;

  std::shared_ptr<const FEField<dim>>                     fe_field;

  const dealii::hp::MappingCollection<dim>                mapping_collection;

  dealii::hp::QCollection<dim>                            quadrature_collection;

  dealii::DoFHandler<dim>                                 dof_handler;

  dealii::IndexSet                                        locally_owned_dofs;

  dealii::IndexSet                                        locally_relevant_dofs;

  dealii::hp::FECollection<dim>                           fe_collection;

  dealii::AffineConstraints<double>                       hanging_node_constraints;

  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix        projection_matrix;

  dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG     preconditioner;

  std::vector<dealii::LinearAlgebraTrilinos::MPI::Vector> projection_rhs;

  std::vector<dealii::LinearAlgebraTrilinos::MPI::Vector> projected_data;

  std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain;

  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law;

  const double                                            shear_at_upper_boundary;

  const dealii::types::boundary_id                        upper_boundary_id;

  double                                                  average_stress_12;

  const double                                            width;

  bool                                                    flag_init_was_called;

  void compute_stress_12_at_boundary();

  void compute_strain_12();

  void assemble_projection_matrix();

  void assemble_local_projection_matrix(
    const typename dealii::DoFHandler<dim>::active_cell_iterator      &cell,
    gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<dim> &scratch,
    gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy         &data);

  void copy_local_to_global_projection_matrix(
    const gCP::AssemblyData::Postprocessing::ProjectionMatrix::Copy &data);

  void assemble_projection_rhs();

  void assemble_local_projection_rhs(
    const typename dealii::DoFHandler<dim>::active_cell_iterator    &cell,
    gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<dim>  &scratch,
    gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy          &data);

  void copy_local_to_global_projection_rhs(
    const gCP::AssemblyData::Postprocessing::ProjectionRHS::Copy &data);

  void project();
};



template <int dim>
inline const dealii::DoFHandler<dim> &
SimpleShear<dim>::get_dof_handler() const
{
  return (dof_handler);
}




template <int dim>
inline const std::vector<dealii::LinearAlgebraTrilinos::MPI::Vector> &
SimpleShear<dim>::get_data() const
{
  return (projected_data);
}


}  // Postprocessing



} // gCP



#endif /* INCLUDE_POSTPROCESSING_H_ */