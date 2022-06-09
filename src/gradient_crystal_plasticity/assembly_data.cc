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
local_matrix(dofs_per_cell, dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(quadrature_collection, finite_element_collection),
hp_fe_values(mapping,
             finite_element_collection,
             quadrature_collection,
             update_flags),
n_slips(n_slips),
slip_id_alpha(0),
slip_id_beta(0),
reduced_gradient_hardening_tensors(n_slips),
symmetrized_schmid_tensors(n_slips),
JxW_values(this->n_q_points),
slip_values(n_slips, std::vector<double>(this->n_q_points)),
old_slip_values(n_slips, std::vector<double>(this->n_q_points)),
gateaux_derivative_values(this->n_q_points,
                          dealii::FullMatrix<double>(n_slips)),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(
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
hp_fe_values(data.hp_fe_values.get_mapping_collection(),
             data.hp_fe_values.get_fe_collection(),
             data.hp_fe_values.get_quadrature_collection(),
             data.hp_fe_values.get_update_flags()),
n_slips(data.n_slips),
slip_id_alpha(0),
slip_id_beta(0),
reduced_gradient_hardening_tensors(n_slips),
symmetrized_schmid_tensors(n_slips),
JxW_values(this->n_q_points),
slip_values(n_slips, std::vector<double>(this->n_q_points)),
old_slip_values(n_slips, std::vector<double>(this->n_q_points)),
gateaux_derivative_values(this->n_q_points,
                          dealii::FullMatrix<double>(n_slips)),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(
  n_slips,
  std::vector<double>(this->dofs_per_cell)),
grad_scalar_phi(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell))
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
  const dealii::hp::MappingCollection<dim>  &mapping,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::QCollection<dim-1>      &face_quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const dealii::UpdateFlags                 face_update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(quadrature_collection, finite_element_collection),
hp_fe_values(mapping,
             finite_element_collection,
             quadrature_collection,
             update_flags),
hp_fe_face_values(mapping,
                  finite_element_collection,
                  face_quadrature_collection,
                  face_update_flags),
n_face_q_points(face_quadrature_collection.max_n_quadrature_points()),
n_slips(n_slips),
JxW_values(this->n_q_points),
strain_tensor_values(this->n_q_points),
elastic_strain_tensor_values(this->n_q_points),
stress_tensor_values(this->n_q_points),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
vector_microscopic_stress_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
resolved_stress_values(n_slips, std::vector<double>(this->n_q_points)),
slip_values(n_slips, std::vector<double>(this->n_q_points)),
old_slip_values(n_slips, std::vector<double>(this->n_q_points)),
scalar_microscopic_stress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
supply_term_values(this->n_q_points),
neumann_boundary_values(n_face_q_points),
vector_phi(this->dofs_per_cell),
face_vector_phi(this->dofs_per_cell),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(n_slips, std::vector<double>(this->dofs_per_cell)),
face_scalar_phi(n_slips,
                std::vector<double>(this->dofs_per_cell)),
grad_scalar_phi(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->dofs_per_cell))
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(data.hp_fe_values.get_mapping_collection(),
             data.hp_fe_values.get_fe_collection(),
             data.hp_fe_values.get_quadrature_collection(),
             data.hp_fe_values.get_update_flags()),
hp_fe_face_values(data.hp_fe_face_values.get_mapping_collection(),
                  data.hp_fe_face_values.get_fe_collection(),
                  data.hp_fe_face_values.get_quadrature_collection(),
                  data.hp_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
n_slips(data.n_slips),
JxW_values(this->n_q_points),
strain_tensor_values(this->n_q_points),
elastic_strain_tensor_values(this->n_q_points),
stress_tensor_values(this->n_q_points),
slip_gradient_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
vector_microscopic_stress_values(
  n_slips,
  std::vector<dealii::Tensor<1,dim>>(this->n_q_points)),
resolved_stress_values(n_slips, std::vector<double>(this->n_q_points)),
slip_values(n_slips, std::vector<double>(this->n_q_points)),
old_slip_values(n_slips, std::vector<double>(this->n_q_points)),
scalar_microscopic_stress_values(
  n_slips,
  std::vector<double>(this->n_q_points)),
supply_term_values(this->n_q_points),
neumann_boundary_values(n_face_q_points),
vector_phi(this->dofs_per_cell),
face_vector_phi(this->dofs_per_cell),
sym_grad_vector_phi(this->dofs_per_cell),
scalar_phi(n_slips, std::vector<double>(this->dofs_per_cell)),
face_scalar_phi(n_slips,
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
  const dealii::hp::MappingCollection<dim>  &mapping,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags,
  const unsigned int                        n_slips)
:
ScratchBase<dim>(quadrature_collection, finite_element_collection),
hp_fe_values(mapping,
             finite_element_collection,
             quadrature_collection,
             update_flags),
n_slips(n_slips),
slips_values(n_slips, std::vector<double>(this->n_q_points, 0.0)),
old_slips_values(n_slips, std::vector<double>(this->n_q_points, 0.0))
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(data.hp_fe_values.get_mapping_collection(),
             data.hp_fe_values.get_fe_collection(),
             data.hp_fe_values.get_quadrature_collection(),
             data.hp_fe_values.get_update_flags()),
n_slips(data.n_slips),
slips_values(n_slips, std::vector<double>(this->n_q_points, 0.0)),
old_slips_values(n_slips, std::vector<double>(this->n_q_points, 0.0))
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



namespace Postprocessing
{



namespace ProjectionMatrix
{



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_matrix(dofs_per_cell, dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(
  const dealii::hp::MappingCollection<dim>  &mapping_collection,
  const dealii::hp::QCollection<dim>        &quadrature_collection,
  const dealii::hp::FECollection<dim>       &finite_element_collection,
  const dealii::UpdateFlags                 update_flags)
:
ScratchBase<dim>(quadrature_collection, finite_element_collection),
hp_fe_values(mapping_collection,
             finite_element_collection,
             quadrature_collection,
             update_flags),
JxW_values(this->n_q_points),
scalar_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
hp_fe_values(data.hp_fe_values.get_mapping_collection(),
             data.hp_fe_values.get_fe_collection(),
             data.hp_fe_values.get_quadrature_collection(),
             data.hp_fe_values.get_update_flags()),
JxW_values(this->n_q_points),
scalar_phi(this->dofs_per_cell)
{}



} // ProjectionMatrix



namespace ProjectionRHS
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
  const dealii::hp::FECollection<dim>       &scalar_finite_element_collection,
  const dealii::UpdateFlags                 scalar_update_flags,
  const dealii::hp::FECollection<dim>       &vector_finite_element_collection,
  const dealii::UpdateFlags                 vector_update_flags)
:
ScratchBase<dim>(quadrature_collection, scalar_finite_element_collection),
scalar_hp_fe_values(mapping_collection,
                    scalar_finite_element_collection,
                    quadrature_collection,
                    scalar_update_flags),
vector_hp_fe_values(mapping_collection,
                    vector_finite_element_collection,
                    quadrature_collection,
                    vector_update_flags),
JxW_values(this->n_q_points),
strain_tensor_values(this->n_q_points),
scalar_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
scalar_hp_fe_values(data.scalar_hp_fe_values.get_mapping_collection(),
                    data.scalar_hp_fe_values.get_fe_collection(),
                    data.scalar_hp_fe_values.get_quadrature_collection(),
                    data.scalar_hp_fe_values.get_update_flags()),
vector_hp_fe_values(data.vector_hp_fe_values.get_mapping_collection(),
                    data.vector_hp_fe_values.get_fe_collection(),
                    data.vector_hp_fe_values.get_quadrature_collection(),
                    data.vector_hp_fe_values.get_update_flags()),
JxW_values(this->n_q_points),
strain_tensor_values(this->n_q_points),
scalar_phi(this->dofs_per_cell)
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

template struct gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<2>;
template struct gCP::AssemblyData::Postprocessing::ProjectionMatrix::Scratch<3>;

template struct gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<2>;
template struct gCP::AssemblyData::Postprocessing::ProjectionRHS::Scratch<3>;
