#include <gCP/assembly_data.h>



namespace gCP
{



namespace AssemblyData
{



CopyBase::CopyBase(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}



template <int dim>
ScratchBase<dim>::ScratchBase(
  const dealii::hp::QCollection<dim>  &quadrature_collection,
  const dealii::hp::FECollection<dim> &finite_element_collection)
:
n_q_points(quadrature_collection.max_n_quadrature_points()),
dofs_per_cell(finite_element_collection.max_dofs_per_cell())
{}



template <int dim>
ScratchBase<dim>::ScratchBase(const ScratchBase<dim> &data)
:
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell)
{}



namespace Jacobian
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
neighbour_cell_local_dof_indices(dofs_per_cell),
local_matrix(dofs_per_cell, dofs_per_cell),
local_coupling_matrix(dofs_per_cell, dofs_per_cell),
cell_is_at_grain_boundary(false)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const dealii::UpdateFlags                 face_update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
hp_fe_values(
  mapping_collection,
  finite_element_collection,
  quadrature_collection,
  update_flags),
hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
neighbour_hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
n_face_q_points(face_quadrature_collection.max_n_quadrature_points()),
n_slips(n_slips),
normal_vector_values(this->n_face_q_points),
JxW_values(this->n_q_points),
face_JxW_values(this->n_face_q_points),
face_neighbor_JxW_values(this->n_face_q_points),
symmetrized_schmid_tensors(n_slips),
slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
old_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
vectorial_microstress_law_jacobian_values(
  this->n_q_points,
  std::vector<dealii::SymmetricTensor<2,dim>>(n_slips)),
scalar_microstress_law_jacobian_values(
  this->n_q_points,
  dealii::FullMatrix<double>(n_slips)),
intra_gateaux_derivative_values(
  this->n_face_q_points,
  dealii::FullMatrix<double>(n_slips)),
inter_gateaux_derivative_values(
  this->n_face_q_points,
  dealii::FullMatrix<double>(n_slips)),
current_cell_displacement_values(this->n_face_q_points),
neighbor_cell_displacement_values(this->n_face_q_points),
current_cell_old_displacement_values(this->n_face_q_points),
neighbor_cell_old_displacement_values(this->n_face_q_points),
damage_variable_values(this->n_face_q_points),
cohesive_law_jacobian_values(this->n_face_q_points),
contact_law_jacobian_values(this->n_face_q_points),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
grad_scalar_phi(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell)),
face_vector_phi(this->dofs_per_cell),
neighbor_face_vector_phi(this->dofs_per_cell),
face_scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
neighbour_face_scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell))
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(
  data.hp_fe_values.get_mapping_collection(),
  data.hp_fe_values.get_fe_collection(),
  data.hp_fe_values.get_quadrature_collection(),
  data.hp_fe_values.get_update_flags()),
hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
neighbour_hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
n_slips(data.n_slips),
normal_vector_values(this->n_face_q_points),
JxW_values(this->n_q_points),
face_JxW_values(this->n_face_q_points),
face_neighbor_JxW_values(this->n_face_q_points),
symmetrized_schmid_tensors(n_slips),
slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
old_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
vectorial_microstress_law_jacobian_values(
  this->n_q_points,
  std::vector<dealii::SymmetricTensor<2,dim>>(n_slips)),
scalar_microstress_law_jacobian_values(
  this->n_q_points,
  dealii::FullMatrix<double>(n_slips)),
intra_gateaux_derivative_values(
  this->n_face_q_points,
  dealii::FullMatrix<double>(n_slips)),
inter_gateaux_derivative_values(
  this->n_face_q_points,
  dealii::FullMatrix<double>(n_slips)),
current_cell_displacement_values(this->n_face_q_points),
neighbor_cell_displacement_values(this->n_face_q_points),
current_cell_old_displacement_values(this->n_face_q_points),
neighbor_cell_old_displacement_values(this->n_face_q_points),
damage_variable_values(this->n_face_q_points),
cohesive_law_jacobian_values(this->n_face_q_points),
contact_law_jacobian_values(this->n_face_q_points),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
grad_scalar_phi(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell)),
face_vector_phi(this->dofs_per_cell),
neighbor_face_vector_phi(this->dofs_per_cell),
face_scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
neighbour_face_scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell))
{}



} // namespace Jacobian



namespace Residual
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_rhs(dofs_per_cell),
local_matrix_for_inhomogeneous_bcs(dofs_per_cell, dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const dealii::UpdateFlags                 face_update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
hp_fe_values(
  mapping_collection,
  finite_element_collection,
  quadrature_collection,
  update_flags),
hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
neighbour_hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
n_face_q_points(face_quadrature_collection.max_n_quadrature_points()),
n_slips(n_slips),
normal_vector_values(this->n_face_q_points),
JxW_values(this->n_q_points),
face_JxW_values(this->n_face_q_points),
face_neighbor_JxW_values(this->n_face_q_points),
strain_tensor_values(this->n_q_points),
elastic_strain_tensor_values(this->n_q_points),
stress_tensor_values(this->n_q_points),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
vectorial_microstress_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
resolved_stress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
old_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
scalar_microstress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
microtraction_values(
  n_slips,
  std::vector<double>(this->n_face_q_points)),
supply_term_values(this->n_q_points),
neumann_boundary_values(this->n_face_q_points),
current_cell_displacement_values(this->n_face_q_points),
neighbor_cell_displacement_values(this->n_face_q_points),
current_cell_old_displacement_values(this->n_face_q_points),
neighbor_cell_old_displacement_values(this->n_face_q_points),
cohesive_traction_values(this->n_face_q_points),
contact_traction_values(this->n_face_q_points),
damage_variable_values(this->n_face_q_points),
face_slip_values(
  n_slips,
  std::vector<double>(this->n_face_q_points)),
neighbour_face_slip_values(
  n_slips,
  std::vector<double>(this->n_face_q_points)),
vector_phi(this->dofs_per_cell),
face_vector_phi(this->dofs_per_cell),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
face_scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
grad_scalar_phi(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell))
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(
  data.hp_fe_values.get_mapping_collection(),
  data.hp_fe_values.get_fe_collection(),
  data.hp_fe_values.get_quadrature_collection(),
  data.hp_fe_values.get_update_flags()),
hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
neighbour_hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
n_slips(data.n_slips),
normal_vector_values(this->n_face_q_points),
JxW_values(this->n_q_points),
face_JxW_values(this->n_face_q_points),
face_neighbor_JxW_values(this->n_face_q_points),
strain_tensor_values(this->n_q_points),
elastic_strain_tensor_values(this->n_q_points),
stress_tensor_values(this->n_q_points),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
vectorial_microstress_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
resolved_stress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
old_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
scalar_microstress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
microtraction_values(
  n_slips,
  std::vector<double>(this->n_face_q_points)),
supply_term_values(this->n_q_points),
neumann_boundary_values(this->n_face_q_points),
current_cell_displacement_values(this->n_face_q_points),
neighbor_cell_displacement_values(this->n_face_q_points),
current_cell_old_displacement_values(this->n_face_q_points),
neighbor_cell_old_displacement_values(this->n_face_q_points),
cohesive_traction_values(this->n_face_q_points),
contact_traction_values(this->n_face_q_points),
damage_variable_values(this->n_face_q_points),
face_slip_values(
  n_slips,
  std::vector<double>(this->n_face_q_points)),
neighbour_face_slip_values(
  n_slips,
  std::vector<double>(this->n_face_q_points)),
vector_phi(this->dofs_per_cell),
face_vector_phi(this->dofs_per_cell),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
face_scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
grad_scalar_phi(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell))
{}



} // namespace Residual



namespace QuadraturePointHistory
{



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const dealii::UpdateFlags                 face_update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
hp_fe_values(
  mapping_collection,
  finite_element_collection,
  quadrature_collection,
  update_flags),
hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
neighbor_hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
n_face_q_points(face_quadrature_collection.max_n_quadrature_points()),
n_slips(n_slips),
slips_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
old_slips_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
face_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
neighbor_face_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
effective_opening_displacement(this->n_face_q_points),
thermodynamic_force_values(this->n_face_q_points),
normal_vector_values(this->n_face_q_points),
current_cell_displacement_values(this->n_face_q_points),
neighbor_cell_displacement_values(this->n_face_q_points),
current_cell_old_displacement_values(this->n_face_q_points),
neighbor_cell_old_displacement_values(this->n_face_q_points),
cohesive_traction_values(this->n_face_q_points)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(
  data.hp_fe_values.get_mapping_collection(),
  data.hp_fe_values.get_fe_collection(),
  data.hp_fe_values.get_quadrature_collection(),
  data.hp_fe_values.get_update_flags()),
hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
neighbor_hp_fe_face_values(
  data.neighbor_hp_fe_face_values.get_mapping_collection(),
  data.neighbor_hp_fe_face_values.get_fe_collection(),
  data.neighbor_hp_fe_face_values.get_quadrature_collection(),
  data.neighbor_hp_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
n_slips(data.n_slips),
slips_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
old_slips_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
face_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
neighbor_face_slip_values(
  n_slips,
  std::vector<double>(this->n_q_points, 0.0)),
effective_opening_displacement(this->n_face_q_points),
thermodynamic_force_values(this->n_face_q_points),
normal_vector_values(this->n_face_q_points),
current_cell_displacement_values(this->n_face_q_points),
neighbor_cell_displacement_values(this->n_face_q_points),
current_cell_old_displacement_values(this->n_face_q_points),
neighbor_cell_old_displacement_values(this->n_face_q_points),
cohesive_traction_values(this->n_face_q_points)
{}



template <int dim>
void Scratch<dim>::reset()
{
  const unsigned int n_q_points = slips_values.size();

  for (unsigned int slip_id = 0; slip_id < n_slips; ++slip_id)
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      slips_values[slip_id][q_point]     = 0.0;
      old_slips_values[slip_id][q_point] = 0.0;
    }
}



} // namespace QuadraturePointHistory



namespace TrialMicrostress
{



namespace Matrix
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_matrix(dofs_per_cell,dofs_per_cell),
local_lumped_matrix(dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
hp_fe_values(
  mapping_collection,
  finite_element_collection,
  quadrature_collection,
  update_flags),
n_slips(n_slips),
JxW_values(this->n_q_points),
test_function_values(
  n_slips,
  std::vector<double>(this->dofs_per_cell))
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(
  data.hp_fe_values.get_mapping_collection(),
  data.hp_fe_values.get_fe_collection(),
  data.hp_fe_values.get_quadrature_collection(),
  data.hp_fe_values.get_update_flags()),
n_slips(this->n_q_points),
JxW_values(this->n_q_points),
test_function_values(
  data.n_slips,
  std::vector<double>(this->dofs_per_cell))
{}



} // namespace Matrix



namespace RightHandSide
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_right_hand_side(dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const dealii::UpdateFlags                 face_update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
n_slips(n_slips),
hp_fe_values(
  mapping_collection,
  finite_element_collection,
  quadrature_collection,
  update_flags),
test_function_values(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
test_function_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell)),
linear_strain_values(this->n_q_points),
elastic_strain_values(this->n_q_points),
stress_values(this->n_q_points),
slip_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
resolved_shear_stress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
vectorial_microstress_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
JxW_values(this->n_q_points),
hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  face_update_flags),
n_face_quadrature_points(face_quadrature_collection.max_n_quadrature_points()),
test_function_face_values(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
slip_gradient_face_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_face_quadrature_points)),
vectorial_microstress_face_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_face_quadrature_points)),
normal_vector_values(this->n_face_quadrature_points),
JxW_face_values(this->n_face_quadrature_points)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
n_slips(data.n_slips),
hp_fe_values(
  data.hp_fe_values.get_mapping_collection(),
  data.hp_fe_values.get_fe_collection(),
  data.hp_fe_values.get_quadrature_collection(),
  data.hp_fe_values.get_update_flags()),
test_function_values(
  data.n_slips,
  std::vector<double>(this->dofs_per_cell)),
test_function_gradient_values(
  data.n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell)),
linear_strain_values(this->n_q_points),
elastic_strain_values(this->n_q_points),
stress_values(this->n_q_points),
slip_values(
  data.n_slips,
  std::vector<double>(this->n_q_points)),
slip_gradient_values(
  data.n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
resolved_shear_stress_values(
  data.n_slips,
  std::vector<double>(this->n_q_points)),
vectorial_microstress_values(
  data.n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
JxW_values(this->n_q_points),
hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
n_face_quadrature_points(data.n_face_quadrature_points),
test_function_face_values(
  data.n_slips,
  std::vector<double>(this->dofs_per_cell)),
slip_gradient_face_values(
  data.n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_face_quadrature_points)),
vectorial_microstress_face_values(
  data.n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_face_quadrature_points)),
normal_vector_values(this->n_face_quadrature_points),
JxW_face_values(this->n_face_quadrature_points)
{}



} // namespace RightHandSide



} // namespace TrialMicroStress


namespace Postprocessing
{



namespace ProjectionMatrix
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_lumped_projection_matrix(dofs_per_cell),
local_matrix_for_inhomogeneous_bcs(dofs_per_cell, dofs_per_cell),
cell_is_at_grain_boundary(false)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  update_flags),
n_face_q_points(face_quadrature_collection.max_n_quadrature_points()),
face_JxW_values(this->n_q_points),
scalar_test_function(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
face_JxW_values(this->n_face_q_points),
scalar_test_function(this->dofs_per_cell)
{}



} // ProjectionMatrix



namespace ProjectionRHS
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_rhs(dofs_per_cell),
local_matrix_for_inhomogeneous_bcs(dofs_per_cell, dofs_per_cell),
cell_is_at_grain_boundary(false)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags)
:
ScratchBase<dim>(
  quadrature_collection,
  finite_element_collection),
hp_fe_face_values(
  mapping_collection,
  finite_element_collection,
  face_quadrature_collection,
  update_flags),
n_face_q_points(face_quadrature_collection.max_n_quadrature_points()),
face_JxW_values(this->n_q_points),
damage_variable_values(this->n_q_points),
scalar_test_function(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_face_values(
  data.hp_fe_face_values.get_mapping_collection(),
  data.hp_fe_face_values.get_fe_collection(),
  data.hp_fe_face_values.get_quadrature_collection(),
  data.hp_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
face_JxW_values(this->n_q_points),
damage_variable_values(this->n_q_points),
scalar_test_function(this->dofs_per_cell)
{}



} // namespace ProjectionRHS



} // namespace Postprocessing



} // namespace AssemblyData



} // namespace gCP



template struct gCP::AssemblyData::ScratchBase<2>;
template struct gCP::AssemblyData::ScratchBase<3>;

template struct gCP::AssemblyData::Jacobian::Scratch<2>;
template struct gCP::AssemblyData::Jacobian::Scratch<3>;

template struct gCP::AssemblyData::Residual::Scratch<2>;
template struct gCP::AssemblyData::Residual::Scratch<3>;

template struct gCP::AssemblyData::QuadraturePointHistory::Scratch<2>;
template struct gCP::AssemblyData::QuadraturePointHistory::Scratch<3>;

template struct gCP::AssemblyData::TrialMicrostress::Matrix::Scratch<2>;
template struct gCP::AssemblyData::TrialMicrostress::Matrix::Scratch<3>;

template struct gCP::AssemblyData::TrialMicrostress::RightHandSide::Scratch<2>;
template struct gCP::AssemblyData::TrialMicrostress::RightHandSide::Scratch<3>;

template struct gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<2>;
template struct gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<3>;

template struct gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<2>;
template struct gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<3>;
