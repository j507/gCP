#ifndef INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_
#define INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_

#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/line_search.h>
#include <gCP/postprocessing.h>
#include <gCP/quadrature_point_history.h>
#include <gCP/run_time_parameters.h>
#include <gCP/utilities.h>

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/utilities.h>

#include <memory>
#include <fstream>



namespace gCP
{



/**
 * @brief Enum acting as row- or column-indicator in the block structure
 * of non-linear system.
 *
 * @details This enum aims to improve the intelligibility of some
 * methods' calls which only act on sub-entities of the block structure
 * of the non-linear system.
 *
 * @enum BlockIndex
 * @var BlockIndex::Macro
 * @brief Represents the block which derives from the linear impulse
 * balance.
 * @var BlockIndex::Micro
 * @brief Represents the block which derives from the pseudo-balance
 */
enum class BlockIndex: unsigned int
{
  Macro = 0,
  Micro = 1
};



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

  ~GradientCrystalPlasticitySolver();

  void init();

  void set_supply_term(
    std::shared_ptr<dealii::TensorFunction<1,dim>> supply_term);

  void set_neumann_boundary_condition(
    const dealii::types::boundary_id                      boundary_id,
    const std::shared_ptr<dealii::TensorFunction<1,dim>>  function);

  void set_macroscopic_strain(
    const dealii::SymmetricTensor<2,dim> macroscopic_strain);

  void solve_nonlinear_system(
    const bool flag_skip_extrapolation = false);

  std::shared_ptr<const Kinematics::ElasticStrain<dim>>
    get_elastic_strain_law() const;

  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>
    get_hooke_law() const;

  std::shared_ptr<const ConstitutiveLaws::CohesiveLaw<dim>>
    get_cohesive_law() const;

  /*!
   * @brief Returns a const reference to the @ref dof_handler
   */
  const dealii::DoFHandler<dim>& get_projection_dof_handler() const;

  const dealii::LinearAlgebraTrilinos::MPI::Vector &get_damage_at_grain_boundaries();

  const dealii::Vector<float> &get_cell_is_at_grain_boundary_vector() const;

  double get_macroscopic_damage();

  // Debugging
  void compute_difference_quotients_jacobian_approximation();

  /*!
   * @brief Temporary method
   *
   * @todo Docu
   */
  void output_data_to_file(std::ostream &file) const;

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

  std::vector<double>                               dimensionless_numbers;

  std::shared_ptr<Kinematics::ElasticStrain<dim>>   elastic_strain;

  std::shared_ptr<ConstitutiveLaws::HookeLaw<dim>>  hooke_law;

  std::shared_ptr<ConstitutiveLaws::ResolvedShearStressLaw<dim>>
                                                    resolved_shear_stress_law;

  std::shared_ptr<ConstitutiveLaws::ScalarMicrostressLaw<dim>>
                                                    scalar_microstress_law;

  std::shared_ptr<ConstitutiveLaws::VectorialMicrostressLaw<dim>>
                                                    vectorial_microstress_law;

  std::shared_ptr<ConstitutiveLaws::MicrotractionLaw<dim>>
                                                    microtraction_law;

  std::shared_ptr<ConstitutiveLaws::CohesiveLaw<dim>>
                                                    cohesive_law;

  std::shared_ptr<ConstitutiveLaws::DegradationFunction>
                                                    degradation_function;

  std::shared_ptr<ConstitutiveLaws::ContactLaw<dim>>
                                                    contact_law;

  dealii::CellDataStorage<
    typename dealii::Triangulation<dim>::cell_iterator,
    QuadraturePointHistory<dim>>                    quadrature_point_history;

  InterfaceDataStorage<
    typename dealii::Triangulation<dim>::cell_iterator,
    InterfaceQuadraturePointHistory<dim>>           interface_quadrature_point_history;

  dealii::Vector<float>                             cell_is_at_grain_boundary;

  dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix jacobian;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector   trial_solution;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector   initial_trial_solution;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector   tmp_trial_solution;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector   newton_update;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector   residual;

  dealii::SymmetricTensor<2,dim>                    macroscopic_strain;

  std::map<dealii::types::boundary_id,
           std::shared_ptr<dealii::TensorFunction<1,dim>>>
                                                    neumann_boundary_conditions;

  dealii::AffineConstraints<double>                 internal_newton_method_constraints;

  Utilities::Logger                                 nonlinear_solver_logger;

  dealii::TableHandler                              table_handler;
  /*!
   * @note Only for debugging purposes
   */
  gCP::Postprocessing::Postprocessor<dim>           postprocessor;

  bool                                              flag_init_was_called;

  void init_quadrature_point_history();

  template <typename SparsityPatternType>
  void make_sparsity_pattern(SparsityPatternType &sparsity_pattern);

  void assemble_linear_system();

  void assemble_jacobian();

  void assemble_local_jacobian(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Jacobian::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Jacobian::Copy                             &data);

  void copy_local_to_global_jacobian(
    const gCP::AssemblyData::Jacobian::Copy &data);

  double assemble_residual();

  void assemble_local_residual(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Residual::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Residual::Copy                             &data);

  void copy_local_to_global_residual(
    const gCP::AssemblyData::Residual::Copy &data);

  void reset_and_update_internal_variables();

  void prepare_quadrature_point_history();

  void reset_quadrature_point_history();

  void reset_and_update_quadrature_point_history();

  void update_local_quadrature_point_history(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>       &scratch,
    gCP::AssemblyData::QuadraturePointHistory::Copy               &data);

  void store_effective_opening_displacement_in_quadrature_history();

  void store_local_effective_opening_displacement(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::QuadraturePointHistory::Scratch<dim>       &scratch,
    gCP::AssemblyData::QuadraturePointHistory::Copy               &data);

  void copy_local_to_global_quadrature_point_history(
    const gCP::AssemblyData::QuadraturePointHistory::Copy &){};

  void monolithic_algorithm();

  void bouncing_algorithm();

  void embracing_algorihtm();

  unsigned int solve_linearized_system();


  unsigned int solve_decoupled_linearized_subsystem(
    const BlockIndex block_index);

  unsigned int solve_reduced_linearized_system();

  /**
   * @brief Updates the trial solution with the given relaxation
   * parameter.
   *
   * @details This function multiplies the increment by the
   * given relaxation parameter and adds the result to the trial
   * solution vector.
   *
   * @param relaxation_parameter The factor by which the increment is
   * multiplied by
   */
  void update_trial_solution(const double relaxation_parameter);

  /**
   * @brief Updates the trial solution with the given relaxation
   * parameter
   *
   * @details This function multiplies the increment by the
   * given relaxation parameter and adds the result to the trial
   * solution vector but only in the sub-block as indicated by the
   * second parameters
   *
   * @param relaxation_parameter The factor by which the increment is
   * multiplied by.
   * @param block The enum indicating which sub-vector should be updated
   */
  void update_trial_solution(
    const double relaxation_parameter,
    const BlockIndex block_index);

  void update_trial_solution(const std::vector<double>
    relaxation_parameter);

  double line_search_algorithm(
    const std::unique_ptr<gCP::LineSearch> &line_search);

  double line_search_algorithm(
    const std::unique_ptr<gCP::LineSearch> &line_search,
    const BlockIndex block_index);

  void store_trial_solution(
    const bool flag_store_initial_trial_solution = false);


  /**
   * @brief Resets the trial solution back to known value.
   *
   * @details This function resets the trial solution vector to either
   * the initial trial solution or the last accepted trial solution,
   * depending on the provided flags. Additionally, the function allows
   * for resetting only a specific sub-block of the trial solution,
   * indicated by the `block` parameter. By default, the entire trial
   * solution is reset.
   *
   * @param flag_reset_to_initial_trial_solution A flag indicating
   * whether to reset the trial solution to the initial trial solution
   * or the last accepted trial solution. Default value is false.
   * @param block An enum indicating which sub-block of the trial
   * solution should be reset. Default value is BlockIndex::Macro,
   * indicating the entire trial solution when the linear system is
   * solved monolithically.
   */
  void reset_trial_solution(
    const bool flag_reset_to_initial_trial_solution = false,
    const BlockIndex block_index = BlockIndex::Macro);

  void extrapolate_initial_trial_solution(
    const bool flag_skip_extrapolation = false);

  void distribute_affine_constraints_to_trial_solution();

  void update_and_output_nonlinear_solver_logger(
    const dealii::Vector<double>  residual_l2_norms);

  void update_and_output_nonlinear_solver_logger(
    const unsigned int           nonlinear_iteration,
    const unsigned int           n_krylov_iterations,
    const unsigned int           n_line_search_iterations,
    const dealii::Vector<double> newton_update_l2_norms,
    const dealii::Vector<double> residual_l2_norms,
    const double                 order_of_convergence,
    const double                 relaxation_parameter = 1.0);

  /*!
   * @note Only for debugging purposes
   */
  void debug_output();

  // Members and methods related to the L2 projection of the damage
  // variable
  dealii::DoFHandler<dim>                           projection_dof_handler;

  dealii::hp::FECollection<dim>                     projection_fe_collection;

  dealii::AffineConstraints<double>                 projection_hanging_node_constraints;

  dealii::LinearAlgebraTrilinos::MPI::Vector        lumped_projection_matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        projection_rhs;

  dealii::LinearAlgebraTrilinos::MPI::Vector        damage_variable_values;

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

  // Members and methods related to the trial microstress
  std::shared_ptr<FEField<dim>> trial_microstress;

  dealii::IndexSet                            locally_owned_active_set;

  dealii::IndexSet                            locally_owned_inactive_set;

  dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix
    trial_microstress_matrix;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    trial_microstress_lumped_matrix;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    trial_microstress_right_hand_side;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    slip_resistance;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    tmp_slip_resistance;

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    active_set;

  gCP::Postprocessing::SlipBasedPostprocessor<dim>
    trial_microstress_postprocessor;

  gCP::Postprocessing::SlipBasedPostprocessor<dim>
    slip_resistance_postprocessor;

  gCP::Postprocessing::SlipBasedPostprocessor<dim>
    active_set_postprocessor;

  using DoFInfo = std::pair<std::vector<
    dealii::types::global_dof_index>, unsigned int>;

  std::map<dealii::types::global_dof_index, DoFInfo>  dof_to_info;

  void reset_internal_newton_method_constraints();

  void compute_trial_microstress();

  void active_set_algorithm(bool &flag_compute_active_set);

  void determine_active_set();

  void determine_inactive_set();

  void reset_inactive_set_values();

  void assemble_trial_microstress_lumped_matrix();

  void assemble_local_trial_microstress_lumped_matrix(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::TrialMicrostress::Matrix::Scratch<dim>     &scratch,
    gCP::AssemblyData::TrialMicrostress::Matrix::Copy             &data);

  void copy_local_to_global_trial_microstress_lumped_matrix(
    const gCP::AssemblyData::TrialMicrostress::Matrix::Copy &data);

  void assemble_trial_microstress_right_hand_side();

  void assemble_local_trial_microstress_right_hand_side(
    const typename dealii::DoFHandler<dim>::active_cell_iterator      &cell,
    gCP::AssemblyData::TrialMicrostress::RightHandSide::Scratch<dim>  &scratch,
    gCP::AssemblyData::TrialMicrostress::RightHandSide::Copy          &data);

  void copy_local_to_global_trial_microstress_right_hand_side(
    const gCP::AssemblyData::TrialMicrostress::RightHandSide::Copy  &data);

  void init_dof_to_info_map();

  void store_slip_resistances();

  void reset_and_update_slip_resistances();

  DeclException1(
    ExcMaxIterations,
    unsigned int,
    << "The maximum number of iterations (" << arg1 << ") has been "
    << "reached.");
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
inline std::shared_ptr<const ConstitutiveLaws::CohesiveLaw<dim>>
GradientCrystalPlasticitySolver<dim>::get_cohesive_law() const
{
  return (cohesive_law);
}



template <int dim>
inline const dealii::DoFHandler<dim> &
GradientCrystalPlasticitySolver<dim>::get_projection_dof_handler() const
{
  return (projection_dof_handler);
}



template <int dim>
inline const dealii::Vector<float> &
GradientCrystalPlasticitySolver<dim>::get_cell_is_at_grain_boundary_vector() const
{
  return (cell_is_at_grain_boundary);
}



}  // namespace gCP



#endif /* INCLUDE_GRADIENT_CRYSTAL_PLASTICITY_H_ */
