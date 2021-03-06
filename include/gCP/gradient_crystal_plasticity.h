#ifndef INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_
#define INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_

#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/quadrature_point_history.h>
#include <gCP/run_time_parameters.h>
#include <gCP/utilities.h>

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <memory>
#include <fstream>

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

  void solve_nonlinear_system();

  double get_residual_norm() const;

  std::shared_ptr<const Kinematics::ElasticStrain<dim>>
    get_elastic_strain_law() const;

  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>
    get_hooke_law() const;

  const dealii::Vector<float> &get_cell_is_at_grain_boundary_vector() const;

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

  std::shared_ptr<ConstitutiveLaws::MicroscopicTractionLaw<dim>>
                                                    microscopic_traction_law;

  dealii::CellDataStorage<
    typename dealii::Triangulation<dim>::cell_iterator,
    QuadraturePointHistory<dim>>                    quadrature_point_history;

  dealii::Vector<float>                             cell_is_at_grain_boundary;

  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  jacobian;

  dealii::LinearAlgebraTrilinos::MPI::Vector        trial_solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector        newton_update;

  dealii::LinearAlgebraTrilinos::MPI::Vector        residual;

  double                                            residual_norm;

  double                                            newton_update_norm;

  Utilities::Logger                                 nonlinear_solver_logger;

  bool                                              flag_init_was_called;

  void init_quadrature_point_history();

  void make_sparsity_pattern(
    dealii::TrilinosWrappers::SparsityPattern &sparsity_pattern);

  void distribute_constraints_to_trial_solution();

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

  void prepare_quadrature_point_history();

  void reset_and_update_quadrature_point_history();

  void update_local_quadrature_point_history(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>       &scratch,
    gCP::AssemblyData::QuadraturePointHistory::Copy               &data);

  void copy_local_to_global_quadrature_point_history(
    const gCP::AssemblyData::QuadraturePointHistory::Copy &){};

  void solve_linearized_system();

  void update_trial_solution(const double relaxation_parameter);

  void extrapolate_initial_trial_solution();
};



template <int dim>
inline std::shared_ptr<const Kinematics::ElasticStrain<dim>>
GradientCrystalPlasticitySolver<dim>::get_elastic_strain_law() const
{
  return (elastic_strain);
}



template <int dim>
inline std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>
GradientCrystalPlasticitySolver<dim>::get_hooke_law() const
{
  return (hooke_law);
}


template <int dim>
inline const dealii::Vector<float> &
GradientCrystalPlasticitySolver<dim>::get_cell_is_at_grain_boundary_vector() const
{
  return (cell_is_at_grain_boundary);
}



}  // namespace gCP



#endif /* INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_ */
