#ifndef INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_
#define INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_

#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/quadrature_point_history.h>
#include <gCP/run_time_parameters.h>

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <memory>

namespace gCP
{



template<int dim>
class GradientCrystalPlasticitySolver
{
public:
  GradientCrystalPlasticitySolver(
    const RunTimeParameters::SolverParameters         &parameters,
    dealii::DiscreteTime                              &discrete_time,
    std::shared_ptr<FEField<dim>>                     &fe_field,
    std::shared_ptr<CrystalsData<dim>>                &crystals_data,
    const std::shared_ptr<dealii::Mapping<dim>>       external_mapping =
        std::shared_ptr<dealii::Mapping<dim>>(),
    const std::shared_ptr<dealii::ConditionalOStream> external_pcout =
        std::shared_ptr<dealii::ConditionalOStream>(),
    const std::shared_ptr<dealii::TimerOutput>        external_timer =
      std::shared_ptr<dealii::TimerOutput>());

  void init();

  void set_supply_term(
    std::shared_ptr<dealii::TensorFunction<1,dim>> supply_term);

  void solve();

  double get_residual_norm() const;

private:
  const RunTimeParameters::SolverParameters         &parameters;

  const dealii::DiscreteTime                        &discrete_time;

  std::shared_ptr<dealii::ConditionalOStream>       pcout;

  std::shared_ptr<dealii::TimerOutput>              timer_output;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::hp::MappingCollection<dim>                mapping_collection;

  dealii::hp::QCollection<dim>                      quadrature_collection;

  dealii::hp::QCollection<dim-1>                    face_quadrature_collection;

  std::shared_ptr<FEField<dim>>                     fe_field;

  std::shared_ptr<const CrystalsData<dim>>          crystals_data;

  std::shared_ptr<dealii::TensorFunction<1,dim>>    supply_term;

  std::shared_ptr<Kinematics::ElasticStrain<dim>>   elastic_strain;

  std::shared_ptr<ConstitutiveLaws::HookeLaw<dim>>  hooke_law;

  std::shared_ptr<ConstitutiveLaws::ResolvedShearStressLaw<dim>>
                                                    resolved_shear_stress_law;

  std::shared_ptr<ConstitutiveLaws::ScalarMicroscopicStressLaw<dim>>
                                                    scalar_microscopic_stress_law;

  std::shared_ptr<ConstitutiveLaws::VectorMicroscopicStressLaw<dim>>
                                                    vector_microscopic_stress_law;

  dealii::CellDataStorage<
    typename dealii::Triangulation<dim>::cell_iterator,
    QuadraturePointHistory<dim>>                    quadrature_point_history;

  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  jacobian;

  dealii::LinearAlgebraTrilinos::MPI::Vector        solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector        newton_update;

  dealii::LinearAlgebraTrilinos::MPI::Vector        residual;

  double                                            residual_norm;

  bool                                              flag_init_was_called;

  void init_quadrature_point_history();

  void assemble_jacobian();

  void assemble_local_jacobian(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Jacobian::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Jacobian::Copy                             &data);

  void copy_local_to_global_jacobian(
    const gCP::AssemblyData::Jacobian::Copy &data);

  void assemble_residual();

  void assemble_local_residual(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Residual::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Residual::Copy                             &data);

  void copy_local_to_global_residual(
    const gCP::AssemblyData::Residual::Copy &data);

  void update_quadrature_point_history();

  void update_local_quadrature_point_history(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>       &scratch,
    gCP::AssemblyData::QuadraturePointHistory::Copy               &data);

  void copy_local_to_global_quadrature_point_history(
    const gCP::AssemblyData::QuadraturePointHistory::Copy &){};
};



template <int dim>
inline double GradientCrystalPlasticitySolver<dim>::get_residual_norm() const
{
  return (residual_norm);
}



}  // namespace gCP



#endif /* INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_ */
