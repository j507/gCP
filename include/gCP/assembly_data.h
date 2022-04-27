#ifndef INCLUDE_ASSEMBLY_DATA_H_
#define INCLUDE_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

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
  ScratchBase(const dealii::Quadrature<dim>     &quadrature_formula,
              const dealii::FiniteElement<dim>  &finite_element);

  ScratchBase(const ScratchBase<dim>  &data);

  const unsigned int  n_q_points;

  const unsigned int  dofs_per_cell;
};



namespace Jacobian
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::FullMatrix<double>  local_matrix;
};



template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const dealii::Mapping<dim>        &mapping,
          const dealii::Quadrature<dim>     &quadrature_formula,
          const dealii::FiniteElement<dim>  &finite_element,
          const dealii::UpdateFlags         update_flags);

  Scratch(const Scratch<dim>  &data);

  dealii::FEValues<dim>                       fe_values;

  std::vector<dealii::SymmetricTensor<2,dim>> sym_grad_phi;
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
  Scratch(const dealii::Mapping<dim>        &mapping,
          const dealii::Quadrature<dim>     &quadrature_formula,
          const dealii::Quadrature<dim-1>   &face_quadrature_formula,
          const dealii::FiniteElement<dim>  &finite_element,
          const dealii::UpdateFlags         update_flags,
          const dealii::UpdateFlags         face_update_flags);

  Scratch(const Scratch<dim>  &data);

  dealii::FEValues<dim>                       fe_values;

  dealii::FEFaceValues<dim>                   fe_face_values;

  const unsigned int                          n_face_q_points;

  std::vector<dealii::Tensor<1,dim>>          phi;

  std::vector<dealii::SymmetricTensor<2,dim>> sym_grad_phi;

  std::vector<dealii::Tensor<1,dim>>          face_phi;

  std::vector<dealii::SymmetricTensor<2,dim>> strain_tensor_values;

  std::vector<dealii::SymmetricTensor<2,dim>> stress_tensor_values;

  std::vector<dealii::Tensor<1,dim>>          supply_term_values;

  std::vector<dealii::Tensor<1,dim>>          neumann_boundary_values;
};




} // namespace Residual



} // namespace AssemblyData



} // namespace gCP



#endif /* INCLUDE_ASSEMBLY_DATA_H_ */
