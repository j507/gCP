#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/run_time_parameters.h>

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

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

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
#include <string>

#ifndef __has_include
  static_assert(false, "__has_include not supported");
#else
#  if __cplusplus >= 201703L && __has_include(<filesystem>)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
     namespace fs = boost::filesystem;
#  endif
#endif


namespace gCP
{



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
class GradientCrystalPlasticitySolver
{
public:
  GradientCrystalPlasticitySolver(
    const RunTimeParameters::Parameters &parameters);

  void run();

private:
  const RunTimeParameters::Parameters               parameters;

  dealii::ConditionalOStream                        pcout;

  dealii::TimerOutput                               timer_output;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::hp::MappingCollection<dim>                mapping_collection;

  std::shared_ptr<FEField<dim>>                     fe_field;

  std::shared_ptr<CrystalsData<dim>>                crystals_data;

  dealii::AffineConstraints<double>                 newton_method_constraints;

  dealii::AffineConstraints<double>                 affine_constraints;

  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  system_matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        system_rhs;

  dealii::LinearAlgebraTrilinos::MPI::Vector        trial_solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector        newton_update;

  dealii::LinearAlgebraTrilinos::MPI::Vector        residual;

  DirichletBoundaryFunction<dim>                    dirichlet_boundary_function;

  NeumannBoundaryFunction<dim>                      neumann_boundary_function;

  SupplyTermFunction<dim>                           supply_term_function;

  ConstitutiveLaws::HookeLaw<dim>                   stiffness_tetrad;

  double                                            relaxation_parameter;

  void make_grid();

  void set_boundary_values();

  void setup();

  void assemble_linear_system();

  void assemble_system_matrix();

  void assemble_local_system_matrix(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Jacobian::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Jacobian::Copy                             &data);

  void copy_local_to_global_system_matrix(const gCP::AssemblyData::Jacobian::Copy &data);

  void assemble_rhs();

  void assemble_local_system_rhs(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Residual::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Residual::Copy                             &data);

  void copy_local_to_global_system_rhs(const gCP::AssemblyData::Residual::Copy &data);

  double compute_residual(const double relaxation_parameter);

  void assemble_local_residual(
    const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
    gCP::AssemblyData::Residual::Scratch<dim>                     &scratch,
    gCP::AssemblyData::Residual::Copy                             &datan);

  void copy_local_to_global_residual(const gCP::AssemblyData::Residual::Copy &data);

  void solve();

  void postprocessing();

  void data_output();
};



template<int dim>
GradientCrystalPlasticitySolver<dim>::GradientCrystalPlasticitySolver(
  const RunTimeParameters::Parameters &parameters)
:
parameters(parameters),
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
mapping(std::make_shared<dealii::MappingQ<dim>>(parameters.mapping_degree)),
mapping_collection(*mapping),
fe_field(std::make_shared<FEField<dim>>(triangulation,
                                        parameters.fe_degree_displacements,
                                        parameters.fe_degree_slips)),
crystals_data(std::make_shared<CrystalsData<dim>>()),
dirichlet_boundary_function(0.0),
neumann_boundary_function(0.0),
supply_term_function(0.0),
stiffness_tetrad(1e5, 0.3),
relaxation_parameter(1.0)
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    if (fs::exists(parameters.graphical_output_directory))
    {
      for (const auto& entry : fs::directory_iterator(parameters.graphical_output_directory))
        if (entry.path().extension() == ".vtu" ||
            entry.path().extension() == ".pvtu")
          fs::remove(entry.path());
    }
    else
    {
      try
      {
        fs::create_directories(parameters.graphical_output_directory);
      }
      catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception in the creation of the output directory: "
                  << std::endl
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
        std::cerr << "Unknown exception in the creation of the output directory!"
                  << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
    }
  }

  stiffness_tetrad.init();
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::make_grid()
{
  dealii::TimerOutput::Scope  t(timer_output, "Discrete domain");

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

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_material_id(0);


  this->pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}


template<int dim>
void GradientCrystalPlasticitySolver<dim>::setup()
{
  dealii::TimerOutput::Scope  t(timer_output, "Solver: Setup");

  crystals_data->init(triangulation,
                      "input/euler_angles",
                      "input/slip_directions",
                      "input/slip_normals");

  fe_field->setup_extractors(crystals_data->get_n_crystals(),
                             crystals_data->get_n_slips());
  fe_field->setup_dofs();

  // The finite element collection contains the finite element systems
  // corresponding to each crystal
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(0);

  // Initiate the actual constraints – boundary conditions – of the problem
  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());
    dealii::VectorTools::interpolate_boundary_values(
      *mapping,
      fe_field->get_dof_handler(),
      0,
      DirichletBoundaryFunction<dim>(),
      affine_constraints,
      fe_field->get_fe_collection().component_mask(
        fe_field->get_displacement_extractor(0)));
  }
  affine_constraints.close();

  // Inhomogeneous constraints are zero-ed out for the Newton Rhapson method
  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(fe_field->get_locally_relevant_dofs());
    newton_method_constraints.merge(fe_field->get_hanging_node_constraints());
    dealii::VectorTools::interpolate_boundary_values(
      *mapping,
      fe_field->get_dof_handler(),
      0,
      dealii::Functions::ZeroFunction<dim>(dim),
      newton_method_constraints,
      fe_field->get_fe_collection().component_mask(
      fe_field->get_displacement_extractor(0)));
  }
  newton_method_constraints.close();

  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(newton_method_constraints);

  fe_field->setup_vectors();

  // Initiate the solver's matrices
  {
    dealii::TrilinosWrappers::SparsityPattern
      sparsity_pattern(fe_field->get_locally_owned_dofs(),
                       fe_field->get_locally_owned_dofs(),
                       fe_field->get_locally_relevant_dofs(),
                       MPI_COMM_WORLD);

    dealii::DoFTools::make_sparsity_pattern(
      fe_field->get_dof_handler(),
      sparsity_pattern,
      fe_field->get_newton_method_constraints(),
      false,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    sparsity_pattern.compress();

    system_matrix.reinit(sparsity_pattern);
  }

  // Initiate the solver's vectors
  {
    system_rhs.reinit(fe_field->distributed_vector);
    residual.reinit(system_rhs);

    newton_update.reinit(fe_field->solution);
    trial_solution.reinit(fe_field->solution);
  }

  // Distribute the constraints to the pertinenet vectors
  // The distribute() method only works on distributed vectors.
  {
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_vector;

    distributed_vector.reinit(fe_field->distributed_vector);

    fe_field->get_affine_constraints().distribute(distributed_vector);

    fe_field->solution  = distributed_vector;

    distributed_vector  = newton_update;

    fe_field->get_newton_method_constraints().distribute(distributed_vector);

    newton_update       = distributed_vector;

    distributed_vector  = trial_solution;

    fe_field->get_affine_constraints().distribute(distributed_vector);

    trial_solution      = distributed_vector;
  }

  this->pcout << "Spatial discretization:"
              << std::endl
              << " Number of degrees of freedom = "
              << fe_field->n_dofs()
              << std::endl << std::endl;
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_linear_system()
{
  assemble_system_matrix();

  assemble_rhs();
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_system_matrix()
{
  dealii::TimerOutput::Scope  t(timer_output, "Solver: Jacobian assembly");

  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Reset data
  system_matrix = 0.0;

  // Create instances of the quadrature collection classes
  dealii::hp::QCollection<dim>    quadrature_collection;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;

  // Initiate the quadrature formula for exact numerical integration
  const dealii::QGauss<dim>       quadrature_formula(3);

  const dealii::QGauss<dim-1>     face_quadrature_formula(3);

  // Add the initiated quadrature formulas to the collections
  quadrature_collection.push_back(quadrature_formula);

  face_quadrature_collection.push_back(face_quadrature_formula);

  // Set up the lambda function for the local assembly operation
  auto worker = [this](const CellIterator                         &cell,
                       gCP::AssemblyData::Jacobian::Scratch<dim>  &scratch,
                       gCP::AssemblyData::Jacobian::Copy          &data)
  {
    this->assemble_local_system_matrix(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](const gCP::AssemblyData::Jacobian::Copy &data)
  {
    this->copy_local_to_global_system_matrix(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags =
    dealii::update_JxW_values |
    dealii::update_gradients |
    dealii::update_quadrature_points;

  //dealii::UpdateFlags face_update_flags = dealii::update_JxW_values;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    worker,
    copier,
    gCP::AssemblyData::Jacobian::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      fe_field->get_fe_collection(),
      update_flags),
    gCP::AssemblyData::Jacobian::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  // Compress global data
  system_matrix.compress(dealii::VectorOperation::add);
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_system_matrix(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Jacobian::Scratch<dim>                     &scratch,
  gCP::AssemblyData::Jacobian::Copy                             &data)
{
  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Reset local data
  data.local_matrix = 0.0;

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values = scratch.hp_fe_values.get_present_fe_values();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.sym_grad_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q);
    }

    // Mapped quadrature weight
    double dv = fe_values.JxW(q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
      {
        data.local_matrix(i,j) -=
          scratch.sym_grad_phi[i] * -1.0 *
          stiffness_tetrad.get_stiffness_tetrad() *
          scratch.sym_grad_phi[j] *
          dv;
      } // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::copy_local_to_global_system_matrix(
  const gCP::AssemblyData::Jacobian::Copy  &data)
{
  fe_field->get_newton_method_constraints().distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    system_matrix);
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_rhs()
{
  dealii::TimerOutput::Scope  t(timer_output, "Solver: Residual assembly");

  // Set up local aliases
  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  // Reset data
  system_rhs = 0.0;

  // Create instances of the quadrature collection classes
  dealii::hp::QCollection<dim>    quadrature_collection;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;

  // Initiate the quadrature formula for exact numerical integration
  const dealii::QGauss<dim>       quadrature_formula(3);

  const dealii::QGauss<dim-1>     face_quadrature_formula(3);

  // Add the initiated quadrature formulas to the collections
  quadrature_collection.push_back(quadrature_formula);

  face_quadrature_collection.push_back(face_quadrature_formula);

  // Set up the lambda function for the local assembly operation
  auto worker = [this](const CellIterator                         &cell,
                       gCP::AssemblyData::Residual::Scratch<dim>  &scratch,
                       gCP::AssemblyData::Residual::Copy          &data)
  {
    this->assemble_local_system_rhs(cell, scratch, data);
  };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this](const gCP::AssemblyData::Residual::Copy  &data)
  {
    this->copy_local_to_global_system_rhs(data);
  };

  // Define the update flags for the FEValues instances
  const dealii::UpdateFlags update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points;

  const dealii::UpdateFlags face_update_flags  =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_quadrature_points;

  // Assemble using the WorkStream approach
  dealii::WorkStream::run(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    worker,
    copier,
    gCP::AssemblyData::Residual::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      face_quadrature_collection,
      fe_field->get_fe_collection(),
      update_flags,
      face_update_flags),
    gCP::AssemblyData::Residual::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  // Compress global data
  system_rhs.compress(dealii::VectorOperation::add);
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_system_rhs(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Residual::Scratch<dim>                                   &scratch,
  gCP::AssemblyData::Residual::Copy                                           &data)
{
  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Reset local data
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  // Update the hp::FEValues instance to the values of the current cell
  scratch.hp_fe_values.reinit(cell);

  const dealii::FEValues<dim> &fe_values = scratch.hp_fe_values.get_present_fe_values();

  // Compute the linear strain tensor at the quadrature points
  fe_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
    fe_field->solution,
    scratch.strain_tensor_values);

  // Compute the supply term at the quadrature points
  supply_term_function.value_list(
    fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Compute the stress tensor at the quadrature points
    scratch.stress_tensor_values[q] =
      stiffness_tetrad.get_stress_tensor(scratch.strain_tensor_values[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q);

      scratch.sym_grad_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q);
    }

    // Mapped quadrature weight
    double dv = fe_values.JxW(q);

    // Loop over the degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
        (scratch.sym_grad_phi[i] *
         scratch.stress_tensor_values[q] * 0.0
         -
         scratch.phi[i] * -1.0 *
         scratch.supply_term_values[q]) *
         dv;
    } // Loop over the degrees of freedom
  } // Loop over quadrature points

  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == 1)
      {
        // Update the hp::FEFaceValues instance to the values of the current cell
        scratch.hp_fe_face_values.reinit(cell, face);

        const dealii::FEFaceValues<dim> &fe_face_values =
          scratch.hp_fe_face_values.get_present_fe_values();

        // Compute the Neumann boundary function values at the
        // quadrature points
        neumann_boundary_function.value_list(
          fe_face_values.get_quadrature_points(),
          scratch.neumann_boundary_values);

        // Loop over face quadrature points
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
        {
          // Extract the test function's values at the face quadrature points
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            scratch.face_phi[i] =
              fe_face_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q);

          // Mapped quadrature weight
          double da = fe_face_values.JxW(q);

          // Loop over degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            data.local_rhs(i) -=
              scratch.face_phi[i] *
              scratch.neumann_boundary_values[q] *
              da;
        } // Loop over face quadrature points
      } // if (face->at_boundary() && face->boundary_id() == 3)
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::copy_local_to_global_system_rhs(
  const gCP::AssemblyData::Residual::Copy  &data)
{
  fe_field->get_newton_method_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    system_rhs,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
double GradientCrystalPlasticitySolver<dim>::compute_residual(const double alpha)
{
  residual = 0.0;

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_trial_solution.reinit(fe_field->distributed_vector);
  distributed_newton_update.reinit(distributed_trial_solution);

  distributed_trial_solution  = fe_field->solution;
  distributed_newton_update   = newton_update;

  distributed_trial_solution.add(alpha, distributed_newton_update);

  trial_solution = distributed_trial_solution;

  dealii::hp::QCollection<dim>    quadrature_collection;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;

  const dealii::QGauss<dim>       quadrature_formula(3);

  const dealii::QGauss<dim-1>     face_quadrature_formula(3);

  quadrature_collection.push_back(quadrature_formula);

  face_quadrature_collection.push_back(face_quadrature_formula);

  using CellIterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;


  auto worker = [this](const CellIterator                         &cell,
                       gCP::AssemblyData::Residual::Scratch<dim>  &scratch,
                       gCP::AssemblyData::Residual::Copy          &data)
  {
    this->assemble_local_residual(cell, scratch, data);
  };

  auto copier = [this](const gCP::AssemblyData::Residual::Copy  &data)
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
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    worker,
    copier,
    gCP::AssemblyData::Residual::Scratch<dim>(
      mapping_collection,
      quadrature_collection,
      face_quadrature_collection,
      fe_field->get_fe_collection(),
      update_flags,
      face_update_flags),
    gCP::AssemblyData::Residual::Copy(
      fe_field->get_fe_collection().max_dofs_per_cell()));

  residual.compress(dealii::VectorOperation::add);

  return residual.l2_norm();
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::assemble_local_residual(
  const typename dealii::DoFHandler<dim>::active_cell_iterator  &cell,
  gCP::AssemblyData::Residual::Scratch<dim>                                   &scratch,
  gCP::AssemblyData::Residual::Copy                                           &data)
{
  // Get the crystal identifier for the current cell
  const unsigned int crystal_id = cell->active_fe_index();

  // Reset local data
  data.local_rhs                          = 0.0;
  data.local_matrix_for_inhomogeneous_bcs = 0.0;

  // Local to global mapping of the indices of the degrees of freedom
  cell->get_dof_indices(data.local_dof_indices);

  // Update the hp::FEValues instance to those of the current cell
  scratch.hp_fe_values.reinit(cell);

  // Get the pertinent FEValues instance
  const dealii::FEValues<dim> &fe_values = scratch.hp_fe_values.get_present_fe_values();

  // Compute the linear strain tensor at the quadrature points
  fe_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
    trial_solution,
    scratch.strain_tensor_values);

  // Compute the supply term at the quadrature points
  supply_term_function.value_list(
    fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Compute the stress tensor at the quadrature points
    scratch.stress_tensor_values[q] =
      stiffness_tetrad.get_stress_tensor(scratch.strain_tensor_values[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q);
      scratch.sym_grad_phi[i] =
        fe_values[fe_field->get_displacement_extractor(crystal_id)].symmetric_gradient(i,q);
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
        fe_values.JxW(q);
    } // Loop over the degrees of freedom
  } // Loop over quadrature points

  // Area integrals
  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == 1)
      {
        // Update the hp::FEFaceValues instance to the values of the current cell
        scratch.hp_fe_face_values.reinit(cell, face);

        const dealii::FEFaceValues<dim> &fe_face_values =
          scratch.hp_fe_face_values.get_present_fe_values();

        // Compute the Neumann boundary values at the quadrature points
        neumann_boundary_function.value_list(
          fe_face_values.get_quadrature_points(),
          scratch.neumann_boundary_values);

        // Loop over face quadrature points
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
        {
          // Extract the test function's values at the face quadrature points
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            scratch.face_phi[i] =
              fe_face_values[fe_field->get_displacement_extractor(crystal_id)].value(i,q);

          // Loop over degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            data.local_rhs(i) -=
              scratch.face_phi[i] *
              scratch.neumann_boundary_values[q] *
              fe_face_values.JxW(q);
        } // Loop over face quadrature points
      } // if (face->at_boundary() && face->boundary_id() == 3)
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::copy_local_to_global_residual(
  const gCP::AssemblyData::Residual::Copy  &data)
{
  fe_field->get_newton_method_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    residual,
    data.local_matrix_for_inhomogeneous_bcs);
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::solve()
{
  dealii::TimerOutput::Scope  t(timer_output, "Solver: Solve ");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_solution.reinit(fe_field->distributed_vector);
  distributed_newton_update.reinit(distributed_solution);

  distributed_solution      = fe_field->solution;
  distributed_newton_update = newton_update;

  // The solver's tolerances are passed to the SolverControl instance
  // used to initialize the solver
  dealii::SolverControl solver_control(
    parameters.n_maximum_iterations,
    std::max(system_rhs.l2_norm() * parameters.relative_tolerance,
             parameters.absolute_tolerance));

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  // The preconditioner is instanciated and initialized
  dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;

  preconditioner.initialize(system_matrix);

  // try-catch scope for the solve() call
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
  /*
  // Zero out the Dirichlet boundary conditions
  fe_field->get_newton_method_constraints().distribute(
    distributed_newton_update);

  // Compute the updated solution
  distributed_solution.add(relaxation_parameter,
                           distributed_newton_update);*/

  fe_field->get_affine_constraints().distribute(
    distributed_solution);

  // Pass the distributed vectors to their ghosted counterpart
  newton_update       = distributed_newton_update;
  fe_field->solution  = distributed_solution;
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::postprocessing()
{
  dealii::TimerOutput::Scope  t(timer_output, "Postprocessing: Point evaluation");

  dealii::Vector<double>  point_value(dim);

  bool point_found = false;

  try
  {
    switch (dim)
    {
    case 2:
      dealii::VectorTools::point_value(*mapping,
                                       fe_field->get_dof_handler(),
                                       fe_field->solution,
                                       dealii::Point<dim>(25.,.5),
                                       point_value);
      break;
    case 3:
      dealii::VectorTools::point_value(*mapping,
                                       fe_field->get_dof_handler(),
                                       fe_field->solution,
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
void GradientCrystalPlasticitySolver<dim>::data_output()
{
  dealii::TimerOutput::Scope  t(timer_output, "Solver: Data output");

  // Explicit declaration of the velocity as a vector
  std::vector<std::string> displacement_names(dim, "Displacement");
  std::vector<std::string> newton_update_names(dim, "NewtonUpdate");
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

  dealii::DataOut<dim> data_out;

  data_out.add_data_vector(fe_field->get_dof_handler(),
                           fe_field->solution,
                           displacement_names,
                           component_interpretation);

  data_out.add_data_vector(fe_field->get_dof_handler(),
                           newton_update,
                           newton_update_names,
                           component_interpretation);

  data_out.build_patches(*mapping,
                         fe_field->get_displacement_fe_degree(),
                         dealii::DataOut<dim>::curved_inner_cells);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(parameters.graphical_output_directory,
                                      "solution",
                                      out_index,
                                      MPI_COMM_WORLD,
                                      5);

  out_index++;
}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::run()
{
  make_grid();

  setup();

  double last_residual_norm     = std::numeric_limits<double>::max();

  unsigned int refinement_cycle = 0;

  do
  {
    this->pcout << "Mesh refinement step " << refinement_cycle << std::endl;

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



} // namespace gCP


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    gCP::RunTimeParameters::Parameters  parameters("input/prm.prm");

    gCP::GradientCrystalPlasticitySolver<2> problem(parameters);

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