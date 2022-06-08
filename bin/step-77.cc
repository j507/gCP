#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
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



namespace step77
{



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
  Scratch(
  const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point,
  const dealii::Mapping<dim>                        &mapping,
  const dealii::Quadrature<dim>                     &quadrature_formula,
  const dealii::FiniteElement<dim>                  &finite_element,
  const dealii::UpdateFlags                         update_flags);

  Scratch(const Scratch<dim>  &data);

  const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point;

  dealii::FEValues<dim>                             fe_values;

  std::vector<dealii::Tensor<1,dim>>                grad_phi;

  std::vector<dealii::Tensor<1,dim>>                old_solution_gradients;

  std::vector<double>                               coefficient;
};




template <int dim>
Scratch<dim>::Scratch(
  const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point,
  const dealii::Mapping<dim>                        &mapping,
  const dealii::Quadrature<dim>                     &quadrature_formula,
  const dealii::FiniteElement<dim>                  &finite_element,
  const dealii::UpdateFlags                         update_flags)
:
ScratchBase<dim>(quadrature_formula, finite_element),
evaluation_point(evaluation_point),
fe_values(mapping,
          finite_element,
          quadrature_formula,
          update_flags),
grad_phi(this->dofs_per_cell),
old_solution_gradients(this->n_q_points),
coefficient(this->n_q_points)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
evaluation_point(data.evaluation_point),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
grad_phi(this->dofs_per_cell),
old_solution_gradients(this->n_q_points),
coefficient(this->n_q_points)
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
  Scratch(
    const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point,
    const dealii::Mapping<dim>                        &mapping,
    const dealii::Quadrature<dim>                     &quadrature_formula,
    const dealii::Quadrature<dim-1>                   &face_quadrature_formula,
    const dealii::FiniteElement<dim>                  &finite_element,
    const dealii::UpdateFlags                         update_flags,
    const dealii::UpdateFlags                         face_update_flags);

  Scratch(const Scratch<dim>  &data);

  const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point;

  dealii::FEValues<dim>                             fe_values;

  dealii::FEFaceValues<dim>                         fe_face_values;

  const unsigned int                                n_face_q_points;

  std::vector<double>                               supply_term_values;

  std::vector<double>                               neumann_bc_values;

  std::vector<dealii::Tensor<1,dim>>                old_solution_gradients;

  std::vector<double>                               coefficient;

  std::vector<double>                               phi;

  std::vector<double>                               face_phi;

  std::vector<dealii::Tensor<1,dim>>                grad_phi;
};



template <int dim>
Scratch<dim>::Scratch(
  const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point,
  const dealii::Mapping<dim>                        &mapping,
  const dealii::Quadrature<dim>                     &quadrature_formula,
  const dealii::Quadrature<dim-1>                   &face_quadrature_formula,
  const dealii::FiniteElement<dim>                  &finite_element,
  const dealii::UpdateFlags                         update_flags,
  const dealii::UpdateFlags                         face_update_flags)
:
ScratchBase<dim>(quadrature_formula, finite_element),
evaluation_point(evaluation_point),
fe_values(mapping,
          finite_element,
          quadrature_formula,
          update_flags),
fe_face_values(mapping,
               finite_element,
               face_quadrature_formula,
               face_update_flags),
n_face_q_points(face_quadrature_formula.size()),
supply_term_values(this->n_q_points),
neumann_bc_values(n_face_q_points),
old_solution_gradients(this->n_q_points),
coefficient(this->n_q_points),
phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
evaluation_point(data.evaluation_point),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
fe_face_values(data.fe_face_values.get_mapping(),
               data.fe_face_values.get_fe(),
               data.fe_face_values.get_quadrature(),
               data.fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
supply_term_values(this->n_q_points),
neumann_bc_values(n_face_q_points),
old_solution_gradients(this->n_q_points),
coefficient(this->n_q_points),
phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



} // namespace RightHandSide



template <int dim>
class DirichletBoundaryFunction : public dealii::Function<dim>
{
public:
  DirichletBoundaryFunction();

  virtual double value(const dealii::Point<dim> &ppoint,
                       const unsigned int component = 0) const override;

private:
};



template<int dim>
DirichletBoundaryFunction<dim>::DirichletBoundaryFunction()
:
dealii::Function<dim>(1,dim)
{}



template<int dim>
double DirichletBoundaryFunction<dim>::value(
  const dealii::Point<dim> &point,
  const unsigned int /*component*/) const
{
  const double x = point(0);
  const double y = point(1);

  return std::sin(2.*M_PI*(x+y));
}



template<int dim>
class step77
{
public:
  step77();

  void run();

private:
  dealii::ConditionalOStream                        pcout;

  dealii::TimerOutput                               timer_output;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::FE_Q<dim>                                 finite_element;

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

  double                                            relaxation_parameter;

  void make_grid();

  void refine_grid();

  void set_boundary_values();

  void setup();

  void assemble_linear_system();

  void assemble_system_matrix(
    const dealii::LinearAlgebraTrilinos::MPI::Vector &evaluation_point);

  void assemble_local_system_matrix(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    Matrix::Scratch<dim>                                          &scratch,
    Matrix::Copy                                                  &data);

  void copy_local_to_global_system_matrix(const Matrix::Copy &data);

  void assemble_rhs(
    const dealii::LinearAlgebraTrilinos::MPI::Vector &evaluation_point);

  void assemble_local_system_rhs(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    RightHandSide::Scratch<dim>                                   &scratch,
    RightHandSide::Copy                                           &data);

  void copy_local_to_global_system_rhs(
    const RightHandSide::Copy                   &data);

  double compute_residual(const double relaxation_parameter);

  void assemble_local_residual(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    RightHandSide::Scratch<dim>                                   &scratch,
    RightHandSide::Copy                                           &datan);

  void copy_local_to_global_residual(
    const RightHandSide::Copy                   &data);

  void solve(
    const dealii::LinearAlgebraTrilinos::MPI::Vector  &rhs,
    dealii::LinearAlgebraTrilinos::MPI::Vector        &solution_vector);

  void postprocessing();

  void data_output(const unsigned int refinement_cycle);
};



template<int dim>
step77<dim>::step77()
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
finite_element(2),
dof_handler(triangulation),
relaxation_parameter(0.1)
{}



template<int dim>
void step77<dim>::make_grid()
{
  dealii::GridGenerator::hyper_ball(triangulation);
  triangulation.refine_global(2);

  this->pcout << "Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;
}




template<int dim>
void step77<dim>::refine_grid()
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
      dealii::QGauss<dim - 1>(finite_element.degree + 1),
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

    // Store the pertinent vectors in the solution transfer object
    std::vector<const dealii::LinearAlgebraTrilinos::MPI::Vector *>
      transfer_vectors(1);
    transfer_vectors[0] = &solution;

    // Prepare the perinent vectors for the coarsening and refining
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

    // Store the pertinent distributed vectors in an std::vector
    std::vector<dealii::LinearAlgebraTrilinos::MPI::Vector *> distributed_vectors(1);
    distributed_vectors[0] = &distributed_solution;

    // Interpolate the vectors prior to the coarsening and refinement
    // into the distributed vectors
    solution_transfer.interpolate(distributed_vectors);

    // Apply the constraints of the field variables to the vectors
    affine_constraints.distribute(distributed_solution);

    // Finally, pass the distributed vectors to their counterparts with
    // locally relevant dofs
    solution = distributed_solution;
  }
}



template<int dim>
void step77<dim>::setup()
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
      dealii::Functions::ZeroFunction<dim>(),
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
}



template<int dim>
void step77<dim>::assemble_linear_system()
{
  assemble_system_matrix(solution);

  assemble_rhs(solution);
}



template<int dim>
void step77<dim>::assemble_system_matrix(
  const dealii::LinearAlgebraTrilinos::MPI::Vector  &evaluation_point)
{
  system_matrix = 0.0;

  const dealii::QGauss<dim> quadrature_formula(
                              finite_element.get_degree() + 1);

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
                                      dealii::update_values |
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
    Matrix::Scratch<dim>(
      evaluation_point,
      *mapping,
      quadrature_formula,
      finite_element,
      update_flags),
    Matrix::Copy(finite_element.dofs_per_cell));

  system_matrix.compress(dealii::VectorOperation::add);
}



template<int dim>
void step77<dim>::assemble_local_system_matrix(
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

  scratch.fe_values.get_function_gradients(
    scratch.evaluation_point,
    scratch.old_solution_gradients);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    scratch.coefficient[q] = 1.0 / std::sqrt(1.0 +
                            scratch.old_solution_gradients[q] *
                            scratch.old_solution_gradients[q]);


    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.grad_phi[i]  = scratch.fe_values.shape_grad(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
      {
        data.local_matrix(i,j) +=
          (scratch.coefficient[q] *
           scratch.grad_phi[i] *
           scratch.grad_phi[j]
           -
           std::pow(scratch.coefficient[q], 3) *
           (scratch.old_solution_gradients[q] *
            scratch.grad_phi[i]) *
           (scratch.old_solution_gradients[q] *
            scratch.grad_phi[j])) *
          scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void step77<dim>::copy_local_to_global_system_matrix(
  const Matrix::Copy  &data)
{
  newton_method_constraints.distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    system_matrix);
}



template<int dim>
void step77<dim>::assemble_rhs(
  const dealii::LinearAlgebraTrilinos::MPI::Vector &evaluation_point)
{
  system_rhs  = 0.0;

  const dealii::QGauss<dim>   quadrature_formula(
                                finite_element.get_degree() + 1);

  const dealii::QGauss<dim-1> face_quadrature_formula(
                                finite_element.get_degree() + 1);

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
    dealii::update_JxW_values;

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
    RightHandSide::Scratch<dim>(evaluation_point,
                                *mapping,
                                quadrature_formula,
                                face_quadrature_formula,
                                finite_element,
                                update_flags,
                                face_update_flags),
    RightHandSide::Copy(finite_element.dofs_per_cell));

  system_rhs.compress(dealii::VectorOperation::add);
}



template<int dim>
void step77<dim>::assemble_local_system_rhs(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  RightHandSide::Scratch<dim>                                   &scratch,
  RightHandSide::Copy                                           &data)
{
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  scratch.fe_values.reinit(cell);

  //supply_term.value_list(scratch.fe_values.get_quadrature_points(),
  //                       scratch.supply_term_values);

  scratch.fe_values.get_function_gradients(scratch.evaluation_point,
                                           scratch.old_solution_gradients);

  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    scratch.coefficient[q] = 1.0 / std::sqrt(1.0 +
                        scratch.old_solution_gradients[q] *
                        scratch.old_solution_gradients[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values.shape_value(i,q);
      scratch.grad_phi[i] = scratch.fe_values.shape_grad(i,q);
    }

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) -=
        scratch.coefficient[q] *
        scratch.grad_phi[i] *
        scratch.old_solution_gradients[q] *
        scratch.fe_values.JxW(q);
    } // Loop over the degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void step77<dim>::copy_local_to_global_system_rhs(
  const RightHandSide::Copy  &data)
{
  newton_method_constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    system_rhs,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
double step77<dim>::compute_residual(const double alpha)
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

  const dealii::QGauss<dim>   quadrature_formula(
                                finite_element.get_degree() + 1);

  const dealii::QGauss<dim-1> face_quadrature_formula(
                                finite_element.get_degree() + 1);

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
    dealii::update_JxW_values;

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
    RightHandSide::Scratch<dim>(solution,
                                *mapping,
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
void step77<dim>::assemble_local_residual(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  RightHandSide::Scratch<dim>                                   &scratch,
  RightHandSide::Copy                                           &data)
{
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  scratch.fe_values.reinit(cell);

  scratch.fe_values.get_function_gradients(trial_solution,
                                           scratch.old_solution_gradients);

  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    scratch.coefficient[q] = 1.0 / std::sqrt(1.0 +
                        scratch.old_solution_gradients[q] *
                        scratch.old_solution_gradients[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values.shape_value(i,q);
      scratch.grad_phi[i] = scratch.fe_values.shape_grad(i,q);
    }

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) -=
        scratch.coefficient[q] *
        scratch.grad_phi[i] *
        scratch.old_solution_gradients[q] *
        scratch.fe_values.JxW(q);
    } // Loop over the degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void step77<dim>::copy_local_to_global_residual(
  const RightHandSide::Copy  &data)
{
  newton_method_constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    residual,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
void step77<dim>::solve(
  const dealii::LinearAlgebraTrilinos::MPI::Vector  &rhs,
  dealii::LinearAlgebraTrilinos::MPI::Vector        &solution_vector)
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_solution.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD,
                              true);
  distributed_newton_update.reinit(distributed_solution);

  distributed_solution      = solution_vector;
  distributed_newton_update = newton_update;

  dealii::SolverControl solver_control(rhs.size(),
                                       rhs.l2_norm() * 1e-6);

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  dealii::LinearAlgebraTrilinos::MPI::PreconditionSSOR preconditioner;

  preconditioner.initialize(system_matrix);

  try
  {
    solver.solve(system_matrix,
                 distributed_newton_update,
                 rhs,
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

  newton_update   = distributed_newton_update;
  solution_vector = distributed_solution;
}



template<int dim>
void step77<dim>::postprocessing()
{

}



template<int dim>
void step77<dim>::data_output(const unsigned int refinement_cycle)
{
  (void)refinement_cycle;

  dealii::DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler,
                           solution,
                           "Solution");

  data_out.add_data_vector(dof_handler,
                           newton_update,
                           "Update");

  data_out.build_patches(*mapping,
                         finite_element.get_degree(),
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
void step77<dim>::run()
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
         inner_iteration < 5;
         ++inner_iteration)
    {
      assemble_linear_system();

      last_residual_norm = system_rhs.l2_norm();

      solve(residual, solution);

      this->pcout << "  Residual: " << compute_residual(0) << std::endl;
    }

    data_output(refinement_cycle);
    ++refinement_cycle;
    this->pcout << std::endl;

  } while (last_residual_norm > 1e-3);
}



} // namespace step77


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    //dealii::deallog.depth_console(2);


    /*AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage(
                  "This program can only be run in serial"));
    */
    step77::step77<2> problem;

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