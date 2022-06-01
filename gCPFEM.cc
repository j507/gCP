#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/gradient_crystal_plasticity.h>
#include <gCP/run_time_parameters.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


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
  DirichletBoundaryFunction(const unsigned int  n_components = 3,
                            const double        time = 0.0);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
};



template<int dim>
DirichletBoundaryFunction<dim>::DirichletBoundaryFunction(
  const unsigned int  n_components,
  const double        time)
:
dealii::Function<dim>(n_components, time)
{}



template<int dim>
void DirichletBoundaryFunction<dim>::vector_value(
  const dealii::Point<dim>  &/*point*/,
  dealii::Vector<double>    &return_vector) const
{
  return_vector = 0.0;
}



template <int dim>
class DisplacementControl : public dealii::Function<dim>
{
public:
  DisplacementControl(const unsigned int  n_components = 3,
                      const double        time = 0.0);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
};



template<int dim>
DisplacementControl<dim>::DisplacementControl(
  const unsigned int  n_components,
  const double        time)
:
dealii::Function<dim>(n_components, time)
{}



template<int dim>
void DisplacementControl<dim>::vector_value(
  const dealii::Point<dim>  &/*point*/,
  dealii::Vector<double>    &return_vector) const
{
  const double t = this->get_time();

  return_vector = 0.0;

  return_vector[0] = t * 1e-3;
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

  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  return_vector[0] = 0.0*x*y;
  return_vector[1] = 0.0 * t;

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
class ProblemClass
{
public:
  ProblemClass(
    const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  const RunTimeParameters::ProblemParameters        parameters;

  std::shared_ptr<dealii::ConditionalOStream>       pcout;

  std::shared_ptr<dealii::TimerOutput>              timer_output;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::DiscreteTime                              discrete_time;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  std::shared_ptr<FEField<dim>>                     fe_field;

  std::shared_ptr<CrystalsData<dim>>                crystals_data;

  GradientCrystalPlasticitySolver<dim>              gCP_solver;

  std::unique_ptr<DirichletBoundaryFunction<dim>>   dirichlet_boundary_function;

  std::unique_ptr<DisplacementControl<dim>>         displacement_control;

  NeumannBoundaryFunction<dim>                      neumann_boundary_function;

  std::shared_ptr<SupplyTermFunction<dim>>          supply_term_function;

  void make_grid();

  void update_dirichlet_boundary_conditions();

  void setup();

  void solve();

  void postprocessing();

  void data_output();
};



template<int dim>
ProblemClass<dim>::ProblemClass(
  const RunTimeParameters::ProblemParameters &parameters)
:
parameters(parameters),
pcout(std::make_shared<dealii::ConditionalOStream>(
  std::cout,
  dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
timer_output(std::make_shared<dealii::TimerOutput>(
  MPI_COMM_WORLD,
  *pcout,
  dealii::TimerOutput::summary,
  dealii::TimerOutput::wall_times)),
mapping(std::make_shared<dealii::MappingQ<dim>>(
  parameters.mapping_degree)),
discrete_time(parameters.start_time,
              parameters.end_time,
              parameters.time_step_size),
triangulation(
  MPI_COMM_WORLD,
  typename dealii::Triangulation<dim>::MeshSmoothing(
  dealii::Triangulation<dim>::smoothing_on_refinement |
  dealii::Triangulation<dim>::smoothing_on_coarsening)),
fe_field(std::make_shared<FEField<dim>>(
  triangulation,
  parameters.fe_degree_displacements,
  parameters.fe_degree_slips)),
crystals_data(std::make_shared<CrystalsData<dim>>()),
gCP_solver(parameters.solver_parameters,
           discrete_time,
           fe_field,
           crystals_data,
           mapping,
           pcout,
           timer_output),
neumann_boundary_function(parameters.start_time),
supply_term_function(
  std::make_shared<SupplyTermFunction<dim>>(parameters.start_time))
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
}



template<int dim>
void ProblemClass<dim>::make_grid()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Discrete domain");

  const double height = 1.0;
  const double width  = 1.0;

  std::vector<unsigned int> repetitions(dim, 10);
  repetitions[1] = height/width * 10;

  switch (dim)
  {
  case 2:
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0),
      dealii::Point<dim>(width, height),
      true);
    break;
  case 3:
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        repetitions,
        dealii::Point<dim>(0,0,0),
        dealii::Point<dim>(width, height, width),
        true);
    }
    break;
  default:
    Assert(false, dealii::ExcNotImplemented());
    break;
  }

  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  dealii::GridTools::collect_periodic_faces(triangulation,
                                            0,
                                            1,
                                            0,
                                            periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  triangulation.refine_global(2);

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_material_id(0);


  *pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}


template<int dim>
void ProblemClass<dim>::setup()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Setup");

  crystals_data->init(triangulation,
                      parameters.euler_angles_pathname,
                      parameters.slips_directions_pathname,
                      parameters.slips_normals_pathname);

  fe_field->setup_extractors(crystals_data->get_n_crystals(),
                             crystals_data->get_n_slips());
  fe_field->setup_dofs();

  displacement_control =
    std::make_unique<DisplacementControl<dim>>(
      fe_field->get_n_components(),
      discrete_time.get_start_time());

  dirichlet_boundary_function =
    std::make_unique<DirichletBoundaryFunction<dim>>(
      fe_field->get_n_components(),
      discrete_time.get_start_time());


  // The finite element collection contains the finite element systems
  // corresponding to each crystal
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(0);

  std::vector<
    dealii::GridTools::
    PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
      periodicity_vector;

  dealii::GridTools::collect_periodic_faces(
    fe_field->get_dof_handler(),
    0,
    1,
    0,
    periodicity_vector);

  // Initiate the actual constraints – boundary conditions – of the problem
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    // Displacements' Dirichlet boundary conditions
    {
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        2,
        *dirichlet_boundary_function,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_displacement_extractor(0)));

      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        3,
        *displacement_control,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_displacement_extractor(0)));
    }

    // Slips' Dirichlet boundary conditions
    std::map<dealii::types::boundary_id,
             const dealii::Function<dim> *> function_map;

    function_map[2] = dirichlet_boundary_function.get();
    function_map[3] = dirichlet_boundary_function.get();

    for(unsigned int slip_id = 0;
        slip_id < crystals_data->get_n_slips();
        ++slip_id)
    {
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        function_map,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_slip_extractor(0, slip_id)));
    }

    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      affine_constraints);
  }
  affine_constraints.close();

  // Inhomogeneous constraints are zero-ed out for the Newton Rhapson method
  dealii::AffineConstraints<double> newton_method_constraints;

  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(fe_field->get_locally_relevant_dofs());
    newton_method_constraints.merge(fe_field->get_hanging_node_constraints());

    dealii::Functions::ZeroFunction<dim> zero_function(
                                          fe_field->get_n_components());

    std::map<dealii::types::boundary_id,
             const dealii::Function<dim> *> function_map;

    function_map[2] = &zero_function;
    function_map[3] = &zero_function;

    // Displacements' Dirichlet boundary conditions
    {
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        function_map,
        newton_method_constraints,
        fe_field->get_fe_collection().component_mask(
        fe_field->get_displacement_extractor(0)));
    }

    // Slips' Dirichlet boundary conditions
    for(unsigned int slip_id = 0;
        slip_id < crystals_data->get_n_slips();
        ++slip_id)
    {
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        function_map,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_slip_extractor(0, slip_id)));
    }

    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      newton_method_constraints);
  }
  newton_method_constraints.close();

  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(newton_method_constraints);

  fe_field->setup_vectors();

  *pcout << "Spatial discretization:"
              << std::endl
              << " Number of degrees of freedom = "
              << fe_field->n_dofs()
              << std::endl << std::endl;
}



template<int dim>
void ProblemClass<dim>::update_dirichlet_boundary_conditions()
{
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    dealii::VectorTools::interpolate_boundary_values(
      *mapping,
      fe_field->get_dof_handler(),
      3,
      *displacement_control,
      affine_constraints,
      fe_field->get_fe_collection().component_mask(
        fe_field->get_displacement_extractor(0)));
  }
  affine_constraints.close();

  fe_field->set_affine_constraints(affine_constraints);

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_vector;

  distributed_vector.reinit(fe_field->distributed_vector);

  fe_field->get_affine_constraints().distribute(distributed_vector);

  fe_field->solution = distributed_vector;
}



template<int dim>
void ProblemClass<dim>::postprocessing()
{

}



template<int dim>
void ProblemClass<dim>::data_output()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Data output");

  // Explicit declaration of the velocity as a vector
  std::vector<std::string> displacement_names(dim, "displacement");
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips();
       ++slip_id)
  {
    displacement_names.emplace_back("slip_" + std::to_string(slip_id));
    component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);
  }

  dealii::DataOut<dim> data_out;

  data_out.add_data_vector(fe_field->get_dof_handler(),
                           fe_field->solution,
                           displacement_names,
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
void ProblemClass<dim>::run()
{
  make_grid();

  setup();

  gCP_solver.init();

  gCP_solver.set_supply_term(supply_term_function);

  while(discrete_time.get_current_time() < discrete_time.get_end_time())
  {
    supply_term_function->set_time(discrete_time.get_next_time());

    displacement_control->set_time(discrete_time.get_next_time());

    update_dirichlet_boundary_conditions();

    gCP_solver.solve_nonlinear_system();

    gCP_solver.update_quadrature_point_history();

    fe_field->update_solution_vectors();

    discrete_time.advance_time();

    postprocessing();

    if (discrete_time.get_step_number() %
         parameters.graphical_output_frequency == 0 ||
        discrete_time.get_current_time() ==
          discrete_time.get_end_time())
      data_output();
  }
}



} // namespace gCP


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    gCP::RunTimeParameters::ProblemParameters parameters("input/prm.prm");

    gCP::ProblemClass<2> problem(parameters);

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