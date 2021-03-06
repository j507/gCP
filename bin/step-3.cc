#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
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
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>



namespace step3
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
  Scratch(const dealii::Mapping<dim>        &mapping,
          const dealii::Quadrature<dim>     &quadrature_formula,
          const dealii::FiniteElement<dim>  &finite_element,
          const dealii::UpdateFlags         update_flags);

  Scratch(const Scratch<dim>  &data);

  dealii::FEValues<dim>               fe_values;

  std::vector<dealii::Tensor<1,dim>>  grad_phi;
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
grad_phi(this->dofs_per_cell)

{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
grad_phi(this->dofs_per_cell)
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

  dealii::FEValues<dim>               fe_values;

  dealii::FEFaceValues<dim>           fe_face_values;

  const unsigned int                  n_face_q_points;

  std::vector<double>                 supply_term_values;

  std::vector<double>                 neumann_bc_values;

  std::vector<double>                 phi;

  std::vector<double>                 face_phi;

  std::vector<dealii::Tensor<1,dim>>  grad_phi;
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
supply_term_values(this->n_q_points),
neumann_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
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
supply_term_values(this->n_q_points),
neumann_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



} // namespace RightHandSide



template <int dim>
class SupplyTerm : public dealii::Function<dim>
{
public:
  SupplyTerm();

  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
};



template<int dim>
SupplyTerm<dim>::SupplyTerm()
:
dealii::Function<dim>(1,dim)
{}



template<int dim>
double SupplyTerm<dim>::value(const dealii::Point<dim> &p,
                              const unsigned int component) const
{
  (void)p;
  (void)component;

  return 1.0;
}



template<int dim>
class step3
{
public:
  step3();

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

  dealii::AffineConstraints<double>                 affine_constraints;

  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  system_matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        system_rhs;

  dealii::LinearAlgebraTrilinos::MPI::Vector        solution;

  SupplyTerm<dim>                                   supply_term;

  void make_grid();

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

  void solve();

  void postprocessing();

  void data_output();
};



template<int dim>
step3<dim>::step3()
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
finite_element(1),
dof_handler(triangulation)
{}



template<int dim>
void step3<dim>::make_grid()
{
  dealii::GridGenerator::hyper_cube(triangulation,-1,1);
  triangulation.refine_global(8);

  std::cout << "Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
}



template<int dim>
void step3<dim>::setup()
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
      dealii::Functions::ZeroFunction<dim>(),
      affine_constraints);
  }
  affine_constraints.close();

  dealii::TrilinosWrappers::SparsityPattern
    sparsity_pattern_(locally_owned_dofs,
                      locally_owned_dofs,
                      locally_relevant_dofs,
                      MPI_COMM_WORLD);

  dealii::DoFTools::make_sparsity_pattern(
    dof_handler,
    sparsity_pattern_,
    affine_constraints,
    false,
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

  sparsity_pattern_.compress();

  system_matrix.reinit(sparsity_pattern_);

  system_rhs.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     MPI_COMM_WORLD,
                     true);
  solution.reinit(locally_relevant_dofs,
                   MPI_COMM_WORLD);
}



template<int dim>
void step3<dim>::assemble_linear_system()
{
  assemble_system_matrix();

  assemble_rhs();
}



template<int dim>
void step3<dim>::assemble_system_matrix()
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
                                      dealii::update_gradients;

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
void step3<dim>::assemble_local_system_matrix(
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
      scratch.grad_phi[i]  = scratch.fe_values.shape_grad(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
      {
        data.local_matrix(i,j) +=
          scratch.grad_phi[i] *
          scratch.grad_phi[j] *
          scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void step3<dim>::copy_local_to_global_system_matrix(
  const Matrix::Copy  &data)
{
  affine_constraints.distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    system_matrix);
}



template<int dim>
void step3<dim>::assemble_rhs()
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
void step3<dim>::assemble_local_system_rhs(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  RightHandSide::Scratch<dim>                                   &scratch,
  RightHandSide::Copy                                           &data)
{
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  scratch.fe_values.reinit(cell);

  supply_term.value_list(scratch.fe_values.get_quadrature_points(),
                         scratch.supply_term_values);

  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i] = scratch.fe_values.shape_value(i,q);
    }

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
        scratch.supply_term_values[q] *
        scratch.phi[i] *
        scratch.fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (affine_constraints.is_inhomogeneously_constrained(
        data.local_dof_indices[i]))
      {
        // Extract test function values at the quadrature points
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          scratch.grad_phi[j] = scratch.fe_values.shape_grad(j,q);

        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          data.local_matrix_for_inhomogeneous_bcs(j,i) +=
            scratch.grad_phi[j] *
            scratch.grad_phi[i] *
            scratch.fe_values.JxW(q);
        }
      } // Loop over the i-th column's rows of the local matrix
    } // Loop over the degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void step3<dim>::copy_local_to_global_system_rhs(
  const RightHandSide::Copy  &data)
{
  affine_constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    system_rhs,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
void step3<dim>::solve()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

  distributed_solution.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD,
                              true);

  distributed_solution = solution;

  dealii::SolverControl                     solver_control(1000, 1e-12);

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;

  preconditioner.initialize(system_matrix);

  try
  {
    solver.solve(system_matrix,
                 distributed_solution,
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

  affine_constraints.distribute(distributed_solution);

  solution = distributed_solution;
}



template<int dim>
void step3<dim>::postprocessing()
{

}



template<int dim>
void step3<dim>::data_output()
{
  dealii::DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler,
                           solution,
                           "Solution");

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
void step3<dim>::run()
{
  make_grid();
  setup();
  assemble_linear_system();
  solve();
  postprocessing();
  data_output();
}



} // namespace step3


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    dealii::deallog.depth_console(2);


    /*AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage(
                  "This program can only be run in serial"));
    */
    std::cout << "Hello world" << std::endl;
    step3::step3<2> problem;

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