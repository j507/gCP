#ifndef INCLUDE_ASSEMBLY_DATA_H_
#define INCLUDE_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

namespace gCP
{



namespace AssemblyData
{



struct CopyBase
{
  CopyBase(const unsigned int dofs_per_cell);

  unsigned int                                  dofs_per_cell;

  std::vector<dealii::types::global_cell_index> local_dof_indices;
};



template <int dim>
struct ScratchBase
{
  ScratchBase(const dealii::hp::QCollection<dim>  &quadrature_collection,
              const dealii::hp::FECollection<dim> &finite_element);

  ScratchBase(const ScratchBase<dim>  &data);

  const unsigned int  n_q_points;

  const unsigned int  dofs_per_cell;
};



namespace Jacobian
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  std::vector<dealii::types::global_cell_index>
                                          neighbour_cell_local_dof_indices;

  std::vector<std::vector<dealii::types::global_cell_index>>
                                          neighbour_cells_local_dof_indices;

  dealii::FullMatrix<double>              local_matrix;

  dealii::FullMatrix<double>              local_coupling_matrix;

  std::vector<dealii::FullMatrix<double>> local_coupling_matrices;

  bool                                    cell_is_at_grain_boundary;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::hp::MappingCollection<dim>  &mapping_collection,
          const dealii::hp::QCollection<dim>        &quadrature_collection,
          const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
          const dealii::hp::FECollection<dim>       &finite_element,
          const dealii::UpdateFlags                 update_flags,
          const dealii::UpdateFlags                 face_update_flags,
          const unsigned int                        n_slips);

  Scratch(const Scratch<dim>  &data);

  using GrainInteractionModuli =
    typename std::pair<std::vector<dealii::FullMatrix<double>>,
                       std::vector<dealii::FullMatrix<double>>>;

  dealii::hp::FEValues<dim>                       hp_fe_values;

  dealii::hp::FEFaceValues<dim>                   hp_fe_face_values;

  dealii::hp::FEFaceValues<dim>                   neighbour_hp_fe_face_values;

  const unsigned int                              n_face_q_points;

  const unsigned int                              n_slips;

  std::vector<dealii::Tensor<1,dim>>              normal_vector_values;

  std::vector<double>                             JxW_values;

  std::vector<double>                             face_JxW_values;

  std::vector<double>                             face_neighbor_JxW_values;

  dealii::SymmetricTensor<4,dim>                  stiffness_tetrad;

  std::vector<dealii::SymmetricTensor<2,dim>>     symmetrized_schmid_tensors;

  std::vector<std::vector<double>>                slip_values;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_gradient_values;

  std::vector<std::vector<double>>                old_slip_values;

  std::vector<std::vector<dealii::SymmetricTensor<2,dim>>>
                                                  vectorial_microstress_law_jacobian_values;

  std::vector<dealii::FullMatrix<double>>         scalar_microstress_law_jacobian_values;

  GrainInteractionModuli                          grain_interaction_moduli;

  std::vector<dealii::FullMatrix<double>>         intra_gateaux_derivative_values;

  std::vector<dealii::FullMatrix<double>>         inter_gateaux_derivative_values;

  std::vector<dealii::Tensor<1,dim>>              current_cell_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              neighbor_cell_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              current_cell_old_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              neighbor_cell_old_displacement_values;

  std::vector<double>                             damage_variable_values;

  std::vector<dealii::SymmetricTensor<2,dim>>     cohesive_law_jacobian_values;

  std::vector<dealii::SymmetricTensor<2,dim>>     contact_law_jacobian_values;

  std::vector<dealii::SymmetricTensor<2,dim>>     sym_grad_vector_phi;

  std::vector<std::vector<double>>                scalar_phi;

  std::vector<std::vector<dealii::Tensor<1,dim>>> grad_scalar_phi;

  std::vector<dealii::Tensor<1,dim>>              face_vector_phi;

  std::vector<dealii::Tensor<1,dim>>              neighbor_face_vector_phi;

  std::vector<std::vector<double>>                face_scalar_phi;

  std::vector<std::vector<double>>                neighbour_face_scalar_phi;
};



} // namespace Jacobian



namespace Residual
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::Vector<double>      local_rhs;

  dealii::FullMatrix<double>  local_matrix_for_inhomogeneous_bcs;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::hp::MappingCollection<dim>  &mapping_collection,
          const dealii::hp::QCollection<dim>        &quadrature_collection,
          const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
          const dealii::hp::FECollection<dim>       &finite_element,
          const dealii::UpdateFlags                 update_flags,
          const dealii::UpdateFlags                 face_update_flags,
          const unsigned int                        n_slips);

  Scratch(const Scratch<dim>  &data);

  using GrainInteractionModuli =
    typename std::pair<std::vector<dealii::FullMatrix<double>>,
                       std::vector<dealii::FullMatrix<double>>>;

  dealii::hp::FEValues<dim>                       hp_fe_values;

  dealii::hp::FEFaceValues<dim>                   hp_fe_face_values;

  dealii::hp::FEFaceValues<dim>                   neighbour_hp_fe_face_values;

  const unsigned int                              n_face_q_points;

  const unsigned int                              n_slips;

  std::vector<dealii::Tensor<1,dim>>              normal_vector_values;

  std::vector<double>                             JxW_values;

  std::vector<double>                             face_JxW_values;

  std::vector<double>                             face_neighbor_JxW_values;

  std::vector<dealii::SymmetricTensor<2,dim>>     strain_tensor_values;

  std::vector<dealii::SymmetricTensor<2,dim>>     elastic_strain_tensor_values;

  std::vector<dealii::SymmetricTensor<2,dim>>     stress_tensor_values;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_gradient_values;

  GrainInteractionModuli                          grain_interaction_moduli;

  std::vector<std::vector<dealii::Tensor<1,dim>>> vectorial_microstress_values;

  std::vector<std::vector<double>>                resolved_stress_values;

  std::vector<std::vector<double>>                slip_values;

  std::vector<std::vector<double>>                old_slip_values;

  std::vector<std::vector<double>>                scalar_microstress_values;

  std::vector<std::vector<double>>                microtraction_values;

  std::vector<dealii::Tensor<1,dim>>              supply_term_values;

  std::vector<dealii::Tensor<1,dim>>              neumann_boundary_values;

  std::vector<dealii::Tensor<1,dim>>              current_cell_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              neighbor_cell_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              current_cell_old_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              neighbor_cell_old_displacement_values;

  std::vector<dealii::Tensor<1,dim>>              cohesive_traction_values;

  std::vector<dealii::Tensor<1,dim>>              contact_traction_values;

  std::vector<double>                             damage_variable_values;

  std::vector<std::vector<double>>                face_slip_values;

  std::vector<std::vector<double>>                neighbour_face_slip_values;

  std::vector<dealii::Tensor<1,dim>>              vector_phi;

  std::vector<dealii::Tensor<1,dim>>              face_vector_phi;

  std::vector<dealii::SymmetricTensor<2,dim>>     sym_grad_vector_phi;

  std::vector<std::vector<double>>                scalar_phi;

  std::vector<std::vector<double>>                face_scalar_phi;

  std::vector<std::vector<dealii::Tensor<1,dim>>> grad_scalar_phi;
};




} // namespace Residual



namespace QuadraturePointHistory
{



struct Copy
{
  void reset(){}
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::hp::MappingCollection<dim>  &mapping_collection,
          const dealii::hp::QCollection<dim>        &quadrature_collection,
          const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
          const dealii::hp::FECollection<dim>       &finite_element_collection,
          const dealii::UpdateFlags                 update_flags,
          const dealii::UpdateFlags                 face_update_flags,
          const unsigned int                        n_slips);

  Scratch(const Scratch<dim>  &data);

  void reset();

  dealii::hp::FEValues<dim>           hp_fe_values;

  dealii::hp::FEFaceValues<dim>       hp_fe_face_values;

  dealii::hp::FEFaceValues<dim>       neighbor_hp_fe_face_values;

  const unsigned int                  n_face_q_points;

  unsigned int                        n_slips;

  std::vector<std::vector<double>>    slips_values;

  std::vector<std::vector<double>>    old_slips_values;

  std::vector<std::vector<double>>    face_slip_values;

  std::vector<std::vector<double>>    neighbor_face_slip_values;

  std::vector<double>                 effective_opening_displacement;

  std::vector<double>                 thermodynamic_force_values;

  std::vector<dealii::Tensor<1,dim>>  normal_vector_values;

  std::vector<dealii::Tensor<1,dim>>  current_cell_displacement_values;

  std::vector<dealii::Tensor<1,dim>>  neighbor_cell_displacement_values;

  std::vector<dealii::Tensor<1,dim>>  current_cell_old_displacement_values;

  std::vector<dealii::Tensor<1,dim>>  neighbor_cell_old_displacement_values;

  std::vector<dealii::Tensor<1,dim>>  cohesive_traction_values;
};




} // namespace QuadraturePointHistory



namespace TrialMicrostress
{



namespace Matrix
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::FullMatrix<double>  local_matrix;

  dealii::Vector<double>      local_lumped_matrix;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::hp::MappingCollection<dim>  &mapping_collection,
          const dealii::hp::QCollection<dim>        &quadrature_collection,
          const dealii::hp::FECollection<dim>       &finite_element_collection,
          const dealii::UpdateFlags                 update_flags,
          const unsigned int                        n_slips);

  Scratch(const Scratch<dim>  &data);

  dealii::hp::FEValues<dim>         hp_fe_values;

  const unsigned int                n_slips;

  std::vector<double>               JxW_values;

  std::vector<std::vector<double>>  test_function_values;
};



} // namespace Matrix



namespace RightHandSide
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::Vector<double>      local_right_hand_side;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const dealii::UpdateFlags                 face_update_flags,
  const unsigned int                        n_slips);

Scratch(const Scratch<dim>  &data);

const unsigned int                n_slips;

dealii::hp::FEValues<dim>         hp_fe_values;

std::vector<std::vector<double>>  test_function_values;

std::vector<std::vector<dealii::Tensor<1,dim>>>
                                  test_function_gradient_values;

std::vector<dealii::SymmetricTensor<2,dim>>
                                  linear_strain_values;

std::vector<dealii::SymmetricTensor<2,dim>>
                                  elastic_strain_values;

std::vector<dealii::SymmetricTensor<2,dim>>
                                  stress_values;

std::vector<std::vector<double>>  slip_values;

std::vector<std::vector<dealii::Tensor<1,dim>>>
                                  slip_gradient_values;

std::vector<std::vector<double>>  resolved_shear_stress_values;

std::vector<std::vector<dealii::Tensor<1,dim>>>
                                  vectorial_microstress_values;

std::vector<double>               JxW_values;

dealii::hp::FEFaceValues<dim>     hp_fe_face_values;

const unsigned int                n_face_quadrature_points;

std::vector<std::vector<double>>  test_function_face_values;

std::vector<std::vector<dealii::Tensor<1,dim>>>
                                  slip_gradient_face_values;

std::vector<std::vector<dealii::Tensor<1,dim>>>
                                  vectorial_microstress_face_values;

std::vector<dealii::Tensor<1,dim>>  normal_vector_values;

std::vector<double>               JxW_face_values;
};



} // namespace RightHandSide



} // namespace TrialMicrostress



namespace Postprocessing
{



namespace ProjectionMatrix
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::Vector<double>      local_lumped_projection_matrix;

  dealii::FullMatrix<double>  local_matrix_for_inhomogeneous_bcs;

  bool                        cell_is_at_grain_boundary;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::hp::MappingCollection<dim>  &mapping_collection,
          const dealii::hp::QCollection<dim>        &quadrature_collection,
          const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
          const dealii::hp::FECollection<dim>       &finite_element_collection,
          const dealii::UpdateFlags                 update_flags);

  Scratch(const Scratch<dim>  &data);

  dealii::hp::FEFaceValues<dim> hp_fe_face_values;

  const unsigned int            n_face_q_points;

  std::vector<double>           face_JxW_values;

  std::vector<double>           scalar_test_function;
};



} // namespace ProjectionMatrix



namespace ProjectionRHS
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::Vector<double>      local_rhs;

  dealii::FullMatrix<double>  local_matrix_for_inhomogeneous_bcs;

  bool                        cell_is_at_grain_boundary;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::hp::MappingCollection<dim>  &mapping_collection,
          const dealii::hp::QCollection<dim>        &quadrature_collection,
          const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
          const dealii::hp::FECollection<dim>       &finite_element_collection,
          const dealii::UpdateFlags                 update_flags);

  Scratch(const Scratch<dim>  &data);

  dealii::hp::FEFaceValues<dim>                   hp_fe_face_values;

  const unsigned int                              n_face_q_points;

  std::vector<double>                             face_JxW_values;

  std::vector<double>                             damage_variable_values;

  std::vector<double>                             scalar_test_function;
};



} // namespace ProjectionRHS



} // namespace Postprocessing



} // namespace AssemblyData



} // namespace gCP



#endif /* INCLUDE_ASSEMBLY_DATA_H_ */
