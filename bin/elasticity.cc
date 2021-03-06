#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>



namespace Elasticity
{



namespace ConstitutiveEquations
{



template<int dim>
class StiffnessTetrad
{
public:
  StiffnessTetrad(const double youngs_modulus,
                  const double poissons_ratio);

  void compute_stiffness_tetrad();

  const dealii::SymmetricTensor<4,dim> &get_stiffness_tetrad() const;

  const dealii::SymmetricTensor<2,dim> compute_stress_tensor(
    const dealii::SymmetricTensor<2,dim> strain_tensor_values) const;

private:
  const double                          bulk_modulus;

  const double                          shear_modulus;

  const dealii::SymmetricTensor<4,dim>  spherical_projector;

  const dealii::SymmetricTensor<4,dim>  deviatoric_projector;

  dealii::SymmetricTensor<4,dim>        stiffness_tetrad;
};


template<int dim>
StiffnessTetrad<dim>::StiffnessTetrad(const double youngs_modulus,
                                      const double poissons_ratio)
:
bulk_modulus(youngs_modulus/(3.0 * (1.0 - 2.0 * poissons_ratio))),
shear_modulus(youngs_modulus/(2.0 * (1.0 + poissons_ratio))),
spherical_projector(1.0 / 3.0 * dealii::outer_product(
                                  dealii::unit_symmetric_tensor<dim>(),
                                  dealii::unit_symmetric_tensor<dim>())),
deviatoric_projector(dealii::identity_tensor<dim>() - spherical_projector)
{
  compute_stiffness_tetrad();
}



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&StiffnessTetrad<dim>::get_stiffness_tetrad() const
{
  return (stiffness_tetrad);
}



template<int dim>
void StiffnessTetrad<dim>::compute_stiffness_tetrad()
{
  stiffness_tetrad = 3.0 * bulk_modulus * spherical_projector
                      + 2.0 * shear_modulus * deviatoric_projector;
}



template<int dim>
const dealii::SymmetricTensor<2,dim> StiffnessTetrad<dim>::
compute_stress_tensor(const dealii::SymmetricTensor<2,dim> strain_tensor_values) const
{
  return stiffness_tetrad * strain_tensor_values;
}



} // namespace ConstitutiveEquations



struct CopyBase
{
  CopyBase(const unsigned int dofs_per_cell);

  unsigned int                                  dofs_per_cell;

  std::vector<dealii::types::global_cell_index> local_dof_indices;
};



CopyBase::CopyBase(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}



template <int dim>
struct ScratchBase
{
  ScratchBase(const dealii::Quadrature<dim>     &quadrature_formula,
              const dealii::FiniteElement<dim>  &finite_element);

  ScratchBase(const ScratchBase<dim>  &data);

  const unsigned int  n_q_points;

  const unsigned int  dofs_per_cell;
};



template <int dim>
ScratchBase<dim>::ScratchBase(
  const dealii::Quadrature<dim>     &quadrature_formula,
  const dealii::FiniteElement<dim>  &finite_element)
:
n_q_points(quadrature_formula.size()),
dofs_per_cell(finite_element.dofs_per_cell)
{}



template <int dim>
ScratchBase<dim>::ScratchBase(const ScratchBase<dim> &data)
:
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell)
{}




namespace Matrix
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::FullMatrix<double>  local_matrix;
};



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_matrix(dofs_per_cell, dofs_per_cell)
{}



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




template <int dim>
Scratch<dim>::Scratch(
  const dealii::Mapping<dim>        &mapping,
  const dealii::Quadrature<dim>     &quadrature_formula,
  const dealii::FiniteElement<dim>  &finite_element,
  const dealii::UpdateFlags         update_flags)
:
ScratchBase<dim>(quadrature_formula, finite_element),
fe_values(mapping,
          finite_element,
          quadrature_formula,
          update_flags),
sym_grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
sym_grad_phi(this->dofs_per_cell)
{}



} // namespace Matrix



namespace RightHandSide
{



struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  dealii::Vector<double>      local_rhs;

  dealii::FullMatrix<double>  local_matrix_for_inhomogeneous_bcs;
};



Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_rhs(dofs_per_cell),
local_matrix_for_inhomogeneous_bcs(dofs_per_cell, dofs_per_cell)
{}



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



template <int dim>
Scratch<dim>::Scratch(
  const dealii::Mapping<dim>        &mapping,
  const dealii::Quadrature<dim>     &quadrature_formula,
  const dealii::Quadrature<dim-1>   &face_quadrature_formula,
  const dealii::FiniteElement<dim>  &finite_element,
  const dealii::UpdateFlags         update_flags,
  const dealii::UpdateFlags         face_update_flags)
:
ScratchBase<dim>(quadrature_formula, finite_element),
fe_values(mapping,
          finite_element,
          quadrature_formula,
          update_flags),
fe_face_values(mapping,
               finite_element,
               face_quadrature_formula,
               face_update_flags),
n_face_q_points(face_quadrature_formula.size()),
phi(this->dofs_per_cell),
sym_grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell),
strain_tensor_values(this->n_q_points),
stress_tensor_values(this->n_q_points),
supply_term_values(this->n_q_points),
neumann_boundary_values(n_face_q_points)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
fe_face_values(data.fe_face_values.get_mapping(),
               data.fe_face_values.get_fe(),
               data.fe_face_values.get_quadrature(),
               data.fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
phi(this->dofs_per_cell),
sym_grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell),
strain_tensor_values(this->n_q_points),
stress_tensor_values(this->n_q_points),
supply_term_values(this->n_q_points),
neumann_boundary_values(n_face_q_points)
{}



} // namespace RightHandSide



template <int dim>
class DirichletBoundaryFunction : public dealii::Function<dim>
{
public:
  DirichletBoundaryFunction(const double time = 0.0);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
};



template<int dim>
DirichletBoundaryFunction<dim>::DirichletBoundaryFunction(const double time)
:
dealii::Function<dim>(dim, time)
{}



template<int dim>
void DirichletBoundaryFunction<dim>::vector_value(
  const dealii::Point<dim>  &point,
  dealii::Vector<double>    &return_vector) const
{
  const double x = point(0);
  const double y = point(1);

  return_vector[0] = x*0;
  return_vector[1] = y*0;

  switch (dim)
  {
    case 3:
      {
        const double z = point(2);
        return_vector[2] = z*0;
      }
      break;
    default:
      break;
  }
}



template <int dim>
class SupplyTermFunction : public dealii::TensorFunction<1,dim>
{
public:
  SupplyTermFunction(const double time = 0.0);

  virtual dealii::Tensor<1, dim> value(const dealii::Point<dim> &point) const override;

private:
};



template <int dim>
SupplyTermFunction<dim>::SupplyTermFunction(const double time)
:
dealii::TensorFunction<1, dim>(time)
{}



template <int dim>
dealii::Tensor<1, dim> SupplyTermFunction<dim>::value(
  const dealii::Point<dim> &point) const
{
  dealii::Tensor<1, dim> return_vector;

  const double x = point(0);
  const double y = point(1);

  return_vector[0] = 0.0*x*y;
  return_vector[1] = -1e-3;

  switch (dim)
  {
    case 3:
      {
        const double z = point(2);
        return_vector[2] = z*0;
      }
      break;
    default:
      break;
  }

  return return_vector;
}



template <int dim>
class NeumannBoundaryFunction : public dealii::TensorFunction<1,dim>
{
public:
  NeumannBoundaryFunction(const double time = 0.0);

  virtual dealii::Tensor<1, dim> value(const dealii::Point<dim> &point) const override;

private:
};



template <int dim>
NeumannBoundaryFunction<dim>::NeumannBoundaryFunction(
  const double time)
:
dealii::TensorFunction<1, dim>(time)
{}



template <int dim>
dealii::Tensor<1, dim> NeumannBoundaryFunction<dim>::value(
  const dealii::Point<dim> &point) const
{
  dealii::Tensor<1, dim> return_vector;

  const double x = point(0);
  const double y = point(1);

  return_vector[0] = 0.0*x;
  return_vector[1] = 0.0*y;

  switch (dim)
  {
    case 3:
      {
        const double z = point(2);
        return_vector[2] = z*0;
      }
      break;
    default:
      break;
  }

  return return_vector;
}




template<int dim>
class Elasticity
{
public:
  Elasticity();

  void run();

private:
  dealii::ConditionalOStream                        pcout;

  dealii::TimerOutput                               timer_output;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::FESystem<dim>                             finite_element;

  dealii::FEValuesExtractors::Vector                displacement_extractor;

  dealii::DoFHandler<dim>                           dof_handler;

  dealii::IndexSet                                  locally_owned_dofs;

  dealii::IndexSet                                  locally_relevant_dofs;

  dealii::AffineConstraints<double>                 hanging_node_constraints;

  dealii::AffineConstraints<double>                 newton_method_constraints;

  dealii::AffineConstraints<double>                 affine_constraints;

  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  system_matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        system_rhs;

  dealii::LinearAlgebraTrilinos::MPI::Vector        solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector        trial_solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector        newton_update;

  dealii::LinearAlgebraTrilinos::MPI::Vector        residual;

  DirichletBoundaryFunction<dim>                    dirichlet_boundary_function;

  NeumannBoundaryFunction<dim>                      neumann_boundary_function;

  SupplyTermFunction<dim>                           supply_term_function;

  double                                            relaxation_parameter;

  ConstitutiveEquations::StiffnessTetrad<dim>       stiffness_tetrad;

  void make_grid();

  void refine_grid();

  void set_boundary_values();

  void setup();

  void assemble_linear_system();

  void assemble_system_matrix();

  void assemble_local_system_matrix(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    Matrix::Scratch<dim>                                          &scratch,
    Matrix::Copy                                                  &data);

  void copy_local_to_global_system_matrix(const Matrix::Copy &data);

  void assemble_rhs();

  void assemble_local_system_rhs(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    RightHandSide::Scratch<dim>                                   &scratch,
    RightHandSide::Copy                                           &data);

  void copy_local_to_global_system_rhs(const RightHandSide::Copy &data);

  double compute_residual(const double relaxation_parameter);

  void assemble_local_residual(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    RightHandSide::Scratch<dim>                                   &scratch,
    RightHandSide::Copy                                           &datan);

  void copy_local_to_global_residual(const RightHandSide::Copy &data);

  void solve();

  void postprocessing();

  void data_output();
};



template<int dim>
Elasticity<dim>::Elasticity()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
timer_output(MPI_COMM_WORLD,
             pcout,
             dealii::TimerOutput::summary,
             dealii::TimerOutput::wall_times),
triangulation(MPI_COMM_WORLD,
               typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening)),
mapping(std::make_shared<dealii::MappingQ<dim>>(1)),
finite_element(dealii::FE_Q<dim>(2), dim),
displacement_extractor(0),
dof_handler(triangulation),
dirichlet_boundary_function(0.0),
neumann_boundary_function(0.0),
supply_term_function(0.0),
relaxation_parameter(1.0),
stiffness_tetrad(1e5, 0.3)
{}



template<int dim>
void Elasticity<dim>::make_grid()
{
  const double length = 25.0;
  const double height = 1.0;

  std::vector<unsigned int> repetitions(dim, 10);
  repetitions[0] = 250;

  switch (dim)
  {
  case 2:
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0),
      dealii::Point<dim>(length, height),
      true);
    break;
  case 3:
    {
      const double width  = 1.0;
      dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        repetitions,
        dealii::Point<dim>(0,0,0),
        dealii::Point<dim>(length, height, width),
        true);
    }
    break;
  default:
    Assert(false, dealii::ExcNotImplemented());
    break;
  }


  triangulation.refine_global(0);

  this->pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}




template<int dim>
void Elasticity<dim>::refine_grid()
{
  // Initiate the solution transfer object
  dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebraTrilinos::MPI::Vector>
  solution_transfer(dof_handler);

  // Scope of the refinement process
  {
    // Initiate vector storing the estimed error per cell
    dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    // Compute the estimated error
    dealii::KellyErrorEstimator<dim>::estimate(
      dof_handler,
      dealii::QGauss<dim - 1>(2),
      std::map<dealii::types::boundary_id, const dealii::Function<dim> *>(),
      solution,
      estimated_error_per_cell,
      dealii::ComponentMask(),
      nullptr,
      0,
      triangulation.locally_owned_subdomain());

    // Set refine or coarse flags in each cell
    dealii::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation,
      estimated_error_per_cell,
      0.3,
      0.03);

    // Prepare the triangulation for the coarsening and refining
    triangulation.prepare_coarsening_and_refinement();

    // Gather the pertinent vectors to prepare them for the coarsening
    // and refining
    std::vector<const dealii::LinearAlgebraTrilinos::MPI::Vector *>
      transfer_vectors(1);
    transfer_vectors[0] = &solution;

    solution_transfer.prepare_for_coarsening_and_refinement(transfer_vectors);

    // Execute the coarsening and refinement of the triangulation
    triangulation.execute_coarsening_and_refinement();
  }

  // Initialize linear algebra corresponding to the new triangulation
  setup();

  // Scope of the transfer process
  {
    // Initiate distributed vectors to recieve the vectors prior to the
    // coarsening and refineming
    dealii::LinearAlgebraTrilinos::MPI::Vector  distributed_solution(system_rhs);

    // Gather the pertinent distributed vectors in an std::vector to
    // interpolate them
    std::vector<dealii::LinearAlgebraTrilinos::MPI::Vector *> distributed_vectors(1);
    distributed_vectors[0] = &distributed_solution;

    solution_transfer.interpolate(distributed_vectors);

    // Apply the constraints of the field variables to the vectors
    affine_constraints.distribute(distributed_solution);

    // Finally, pass the distributed vectors to their ghost counterparts
    solution = distributed_solution;
  }
}



template<int dim>
void Elasticity<dim>::setup()
{
  dof_handler.distribute_dofs(finite_element);

  dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);

  hanging_node_constraints.clear();
  {
    hanging_node_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    hanging_node_constraints);
  }
  hanging_node_constraints.close();

  affine_constraints.clear();
  {
    affine_constraints.reinit(locally_relevant_dofs);
    affine_constraints.merge(hanging_node_constraints);
    dealii::VectorTools::interpolate_boundary_values(
      *mapping,
      dof_handler,
      0,
      DirichletBoundaryFunction<dim>(),
      affine_constraints);
  }
  affine_constraints.close();

  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(locally_relevant_dofs);
    newton_method_constraints.merge(hanging_node_constraints);
    dealii::VectorTools::interpolate_boundary_values(
      *mapping,
      dof_handler,
      0,
      dealii::Functions::ZeroFunction<dim>(dim),
      newton_method_constraints);
  }
  newton_method_constraints.close();

  dealii::TrilinosWrappers::SparsityPattern
    sparsity_pattern(locally_owned_dofs,
                      locally_owned_dofs,
                      locally_relevant_dofs,
                      MPI_COMM_WORLD);

  dealii::DoFTools::make_sparsity_pattern(
    dof_handler,
    sparsity_pattern,
    newton_method_constraints,
    false,
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

  sparsity_pattern.compress();

  system_matrix.reinit(sparsity_pattern);

  system_rhs.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     MPI_COMM_WORLD,
                     true);
  residual.reinit(system_rhs);

  solution.reinit(locally_relevant_dofs,
                   MPI_COMM_WORLD);
  newton_update.reinit(solution);
  trial_solution.reinit(solution);

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_vector;

  distributed_vector.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD,
                              true);

  distributed_vector = solution;

  affine_constraints.distribute(distributed_vector);

  solution = distributed_vector;

  distributed_vector = newton_update;

  newton_method_constraints.distribute(distributed_vector);

  distributed_vector = trial_solution;

  affine_constraints.distribute(distributed_vector);

  trial_solution = distributed_vector;


  this->pcout << "Spatial discretization:"
              << std::endl
              << " Number of degrees of freedom = "
              << dof_handler.n_dofs()
              << std::endl << std::endl;
}



template<int dim>
void Elasticity<dim>::assemble_linear_system()
{
  assemble_system_matrix();

  assemble_rhs();
}



template<int dim>
void Elasticity<dim>::assemble_system_matrix()
{
  system_matrix = 0.0;

  const dealii::QGauss<dim> quadrature_formula(3);

  using cell_iterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  auto worker = [this](const cell_iterator  &cell,
                       Matrix::Scratch<dim> &scratch,
                       Matrix::Copy         &data)
  {
    this->assemble_local_system_matrix(cell, scratch, data);
  };

  auto copier = [this](const Matrix::Copy &data)
  {
    this->copy_local_to_global_system_matrix(data);
  };

  dealii::UpdateFlags update_flags =  dealii::update_JxW_values |
                                      dealii::update_gradients |
                                      dealii::update_quadrature_points;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.end()),
    worker,
    copier,
    Matrix::Scratch<dim>(*mapping,
                         quadrature_formula,
                         finite_element,
                         update_flags),
    Matrix::Copy(finite_element.dofs_per_cell));

  system_matrix.compress(dealii::VectorOperation::add);
}



template<int dim>
void Elasticity<dim>::assemble_local_system_matrix(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  Matrix::Scratch<dim>                                          &scratch,
  Matrix::Copy                                                  &data)
{
  // Reset local data
  data.local_matrix = 0.0;

  // Reset finite element values to those of the current cell
  scratch.fe_values.reinit(cell);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.sym_grad_phi[i] =
        scratch.fe_values[displacement_extractor].symmetric_gradient(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
      {
        data.local_matrix(i,j) -=
          scratch.sym_grad_phi[i] * -1.0 *
          stiffness_tetrad.get_stiffness_tetrad() *
          scratch.sym_grad_phi[j] *
          scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void Elasticity<dim>::copy_local_to_global_system_matrix(
  const Matrix::Copy  &data)
{
  newton_method_constraints.distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    system_matrix);
}



template<int dim>
void Elasticity<dim>::assemble_rhs()
{
  system_rhs = 0.0;

  const dealii::QGauss<dim>   quadrature_formula(3);

  const dealii::QGauss<dim-1> face_quadrature_formula(3);

  using cell_iterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;


  auto worker = [this](const cell_iterator          &cell,
                       RightHandSide::Scratch<dim>  &scratch,
                       RightHandSide::Copy          &data)
  {
    this->assemble_local_system_rhs(cell, scratch, data);
  };

  auto copier = [this](const RightHandSide::Copy  &data)
  {
    this->copy_local_to_global_system_rhs(data);
  };

  const dealii::UpdateFlags update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points;

  const dealii::UpdateFlags face_update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_quadrature_points;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.end()),
    worker,
    copier,
    RightHandSide::Scratch<dim>(*mapping,
                                quadrature_formula,
                                face_quadrature_formula,
                                finite_element,
                                update_flags,
                                face_update_flags),
    RightHandSide::Copy(finite_element.dofs_per_cell));

  system_rhs.compress(dealii::VectorOperation::add);
}



template<int dim>
void Elasticity<dim>::assemble_local_system_rhs(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  RightHandSide::Scratch<dim>                                   &scratch,
  RightHandSide::Copy                                           &data)
{
  // Reset local data
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  // Update the FEValues instance with the values of the current cell
  scratch.fe_values.reinit(cell);

  // Compute the linear strain tensor at the quadrature points
  scratch.fe_values[displacement_extractor].get_function_symmetric_gradients(
    solution,
    scratch.strain_tensor_values);

  // Compute the supply term at the quadrature points
  supply_term_function.value_list(
    scratch.fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Compute the stress tensor at the quadrature points
    scratch.stress_tensor_values[q] =
      stiffness_tetrad.compute_stress_tensor(scratch.strain_tensor_values[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i] =
        scratch.fe_values[displacement_extractor].value(i,q);
      scratch.sym_grad_phi[i] =
        scratch.fe_values[displacement_extractor].symmetric_gradient(i,q);
    }

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
        (scratch.sym_grad_phi[i] *
         scratch.stress_tensor_values[q] * 0.0
         -
         scratch.phi[i] * -1.0 *
         scratch.supply_term_values[q]) *
        scratch.fe_values.JxW(q);
    } // Loop over the degrees of freedom
  } // Loop over quadrature points

  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == 1)
      {
        // Update the FEValues instance with the values of the current cell
        scratch.fe_face_values.reinit(cell, face);

        // Compute the Neumann boundary function values at the
        // quadrature points
        neumann_boundary_function.value_list(
          scratch.fe_face_values.get_quadrature_points(),
          scratch.neumann_boundary_values);

        // Loop over face quadrature points
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
        {
          // Extract the test function's values at the face quadrature points
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            scratch.face_phi[i] =
              scratch.fe_face_values[displacement_extractor].value(i,q);

          // Loop over degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            data.local_rhs(i) -=
              scratch.face_phi[i] *
              scratch.neumann_boundary_values[q] *
              scratch.fe_face_values.JxW(q);
        } // Loop over face quadrature points
      } // if (face->at_boundary() && face->boundary_id() == 3)
}



template<int dim>
void Elasticity<dim>::copy_local_to_global_system_rhs(
  const RightHandSide::Copy  &data)
{
  newton_method_constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    system_rhs,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
double Elasticity<dim>::compute_residual(const double alpha)
{
  residual = 0.0;

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_trial_solution.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD,
                              true);
  distributed_newton_update.reinit(distributed_trial_solution);

  distributed_trial_solution  = solution;
  distributed_newton_update   = newton_update;

  distributed_trial_solution.add(alpha, distributed_newton_update);

  trial_solution = distributed_trial_solution;

  const dealii::QGauss<dim>   quadrature_formula(3);

  const dealii::QGauss<dim-1> face_quadrature_formula(3);

  using cell_iterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;


  auto worker = [this](const cell_iterator          &cell,
                       RightHandSide::Scratch<dim>  &scratch,
                       RightHandSide::Copy          &data)
  {
    this->assemble_local_residual(cell, scratch, data);
  };

  auto copier = [this](const RightHandSide::Copy  &data)
  {
    this->copy_local_to_global_residual(data);
  };

  const dealii::UpdateFlags update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points;

  const dealii::UpdateFlags face_update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_quadrature_points;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.end()),
    worker,
    copier,
    RightHandSide::Scratch<dim>(*mapping,
                                quadrature_formula,
                                face_quadrature_formula,
                                finite_element,
                                update_flags,
                                face_update_flags),
    RightHandSide::Copy(finite_element.dofs_per_cell));

  residual.compress(dealii::VectorOperation::add);

  return residual.l2_norm();
}



template<int dim>
void Elasticity<dim>::assemble_local_residual(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  RightHandSide::Scratch<dim>                                   &scratch,
  RightHandSide::Copy                                           &data)
{
  // Reset local data
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  // Update the FEValues instance with the values of the current cell
  scratch.fe_values.reinit(cell);

  // Compute the linear strain tensor at the quadrature points
  scratch.fe_values[displacement_extractor].get_function_symmetric_gradients(
    trial_solution,
    scratch.strain_tensor_values);

  // Compute the supply term at the quadrature points
  supply_term_function.value_list(
    scratch.fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Compute the stress tensor at the quadrature points
    scratch.stress_tensor_values[q] =
      stiffness_tetrad.compute_stress_tensor(scratch.strain_tensor_values[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i] =
        scratch.fe_values[displacement_extractor].value(i,q);
      scratch.sym_grad_phi[i] =
        scratch.fe_values[displacement_extractor].symmetric_gradient(i,q);
    }

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
        (scratch.sym_grad_phi[i] *
         scratch.stress_tensor_values[q]
         -
         scratch.phi[i] *
         scratch.supply_term_values[q]) *
        scratch.fe_values.JxW(q);
    } // Loop over the degrees of freedom
  } // Loop over quadrature points

  // Area integrals
  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == 1)
      {
        // Update the FEValues instance with the values of the current cell
        scratch.fe_face_values.reinit(cell, face);

        // Compute the Neumann boundary values at the quadrature points
        neumann_boundary_function.value_list(
          scratch.fe_face_values.get_quadrature_points(),
          scratch.neumann_boundary_values);

        // Loop over face quadrature points
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
        {
          // Extract the test function's values at the face quadrature points
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            scratch.face_phi[i] =
              scratch.fe_face_values[displacement_extractor].value(i,q);

          // Loop over degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            data.local_rhs(i) -=
              scratch.face_phi[i] *
              scratch.neumann_boundary_values[q] *
              scratch.fe_face_values.JxW(q);
        } // Loop over face quadrature points
      } // if (face->at_boundary() && face->boundary_id() == 3)
}



template<int dim>
void Elasticity<dim>::copy_local_to_global_residual(
  const RightHandSide::Copy  &data)
{
  newton_method_constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    residual,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
void Elasticity<dim>::solve()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_solution.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD,
                              true);
  distributed_newton_update.reinit(distributed_solution);

  distributed_solution      = solution;
  distributed_newton_update = newton_update;

  dealii::SolverControl solver_control(
    1000,
    std::max(system_rhs.l2_norm() * 1e-6, 1e-8));

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;

  preconditioner.initialize(system_matrix);

  dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control);

  try
  {
    solver.solve(system_matrix,
                 distributed_newton_update,
                 system_rhs,
                 preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception in the solve method!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  newton_method_constraints.distribute(distributed_newton_update);

  distributed_solution.add(relaxation_parameter, distributed_newton_update);

  newton_update = distributed_newton_update;
  solution      = distributed_solution;
}



template<int dim>
void Elasticity<dim>::postprocessing()
{
  dealii::Vector<double>  point_value(dim);

  bool point_found = false;

  try
  {
    switch (dim)
    {
    case 2:
      dealii::VectorTools::point_value(*mapping,
                                       dof_handler,
                                       solution,
                                       dealii::Point<dim>(25.,.5),
                                       point_value);
      break;
    case 3:
      dealii::VectorTools::point_value(*mapping,
                                       dof_handler,
                                       solution,
                                       dealii::Point<dim>(25.,.5,.5),
                                       point_value);
      break;
    default:
      break;
    }

    point_found = true;
  }
  catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  const int n_procs = dealii::Utilities::MPI::sum(point_found ? 1 : 0,
                                                  MPI_COMM_WORLD);

  dealii::Utilities::MPI::sum(point_value,
                              MPI_COMM_WORLD,
                              point_value);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    point_value /= n_procs;

  dealii::Tensor<1, dim> point_value_tensor;
  for (unsigned i=0; i<dim; ++i)
    point_value_tensor[i] = point_value[i];

  this->pcout << "w = " << point_value_tensor[1] << std::endl;
}



template<int dim>
void Elasticity<dim>::data_output()
{
  // Explicit declaration of the velocity as a vector
  std::vector<std::string> displacement_names(dim, "Displacement");
  std::vector<std::string> newton_update_names(dim, "NewtonUpdate");
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

  dealii::DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler,
                           solution,
                           displacement_names,
                           component_interpretation);

  data_out.add_data_vector(dof_handler,
                           newton_update,
                           newton_update_names,
                           component_interpretation);

  data_out.build_patches(*mapping,
                         2,
                         dealii::DataOut<dim>::curved_inner_cells);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record("./",
                                      "Solution",
                                      out_index,
                                      MPI_COMM_WORLD,
                                      5);

  out_index++;
}



template<int dim>
void Elasticity<dim>::run()
{
  make_grid();

  setup();

  double last_residual_norm     = std::numeric_limits<double>::max();

  unsigned int refinement_cycle = 0;

  do
  {
    this->pcout << "Mesh refinement step " << refinement_cycle << std::endl;

    if (refinement_cycle != 0)
      refine_grid();

    this->pcout << "  Initial residual: " << compute_residual(0) << std::endl;

    for (unsigned int inner_iteration = 0;
         inner_iteration < 1000;
         ++inner_iteration)
    {
      assemble_linear_system();

      last_residual_norm = system_rhs.l2_norm();

      solve();

      data_output();
      postprocessing();

      return;

      double current_residual = compute_residual(0);

      this->pcout << "  Residual: " << current_residual << std::endl;

      if (current_residual < 1e-11)
        break;
    }

    data_output();

    ++refinement_cycle;

    this->pcout << std::endl;

    postprocessing();

    break;

  } while (last_residual_norm > 1e-8);
}



} // namespace Elasticity


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    Elasticity::Elasticity<2> problem;

    problem.run();

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}