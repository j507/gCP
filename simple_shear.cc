#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/gradient_crystal_plasticity.h>
#include <gCP/postprocessing.h>
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
  DisplacementControl(const double        shear_strain_at_upper_boundary,
                      const double        height,
                      const unsigned int  n_components = 3,
                      const double        time = 0.0);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
  const double shear_strain_at_upper_boundary;

  const double height;
};



template<int dim>
DisplacementControl<dim>::DisplacementControl(
  const double        shear_strain_at_upper_boundary,
  const double        height,
  const unsigned int  n_components,
  const double        time)
:
dealii::Function<dim>(n_components, time),
shear_strain_at_upper_boundary(shear_strain_at_upper_boundary),
height(height)
{}



template<int dim>
void DisplacementControl<dim>::vector_value(
  const dealii::Point<dim>  &/*point*/,
  dealii::Vector<double>    &return_vector) const
{
  const double t = this->get_time();

  return_vector = 0.0;

  return_vector[0] = t * height * shear_strain_at_upper_boundary;
}



template<int dim>
class SimpleShearProblem
{
public:
  SimpleShearProblem(
    const RunTimeParameters::SimpleShearParameters &parameters);

  void run();

private:
  const RunTimeParameters::SimpleShearParameters    parameters;

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

  Postprocessing::SimpleShear<dim>                  simple_shear;

  const double                                      string_width;

  void make_grid();

  void update_dirichlet_boundary_conditions();

  void setup();

  void setup_constraints();

  void solve();

  void postprocessing();

  void triangulation_output();

  void data_output();
};



template<int dim>
SimpleShearProblem<dim>::SimpleShearProblem(
  const RunTimeParameters::SimpleShearParameters &parameters)
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
simple_shear(fe_field,
             mapping,
             parameters.shear_strain_at_upper_boundary,
             3,
             parameters.width),
string_width((std::to_string((unsigned int)(
              (parameters.end_time - parameters.start_time) /
              parameters.time_step_size)) +
             "Step ").size())
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    if (fs::exists(parameters.graphical_output_directory + "paraview/"))
    {
      *pcout
        << "Deleting *.vtu and *.pvtu files inside the output folder... "
        << std::flush;

      for (const auto& entry : fs::directory_iterator(parameters.graphical_output_directory + "paraview/"))
        if (entry.path().extension() == ".vtu" ||
            entry.path().extension() == ".pvtu")
          fs::remove(entry.path());

      *pcout
        << "done!\n\n";
    }
    else
    {
      try
      {
        fs::create_directories(parameters.graphical_output_directory + "paraview/");
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
void SimpleShearProblem<dim>::make_grid()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem - Triangulation");

  std::vector<std::vector<double>> step_sizes(dim, std::vector<double>());

  step_sizes[0] = std::vector<double>(5, parameters.width / 5);

  const double factor       = .50;

  unsigned int n_divisions  = 6;

  for (unsigned int i = 0; i < n_divisions; ++i)
  {
    if (i == 0)
      step_sizes[1].push_back((0.5*parameters.height)*std::pow(1-factor, n_divisions - 1 -  i));
    else
      step_sizes[1].push_back((0.5*parameters.height)*std::pow(1-factor, n_divisions - 1 - i)*factor);
  }

  std::vector<double> mirror_vector(step_sizes[1]);
  std::reverse(mirror_vector.begin(), mirror_vector.end());
  step_sizes[1].insert(step_sizes[1].end(),
                       mirror_vector.begin(),
                       mirror_vector.end());

  std::vector<unsigned int> repetitions(2, 10);
  repetitions[1] = 100;

  switch (dim)
  {
  case 2:
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0),
      dealii::Point<dim>(parameters.width, parameters.height),
      true);
    break;
  case 3:
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        repetitions,
        dealii::Point<dim>(0,0,0),
        dealii::Point<dim>(parameters.width, parameters.height, parameters.width),
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

  triangulation.refine_global(parameters.n_global_refinements);

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_material_id(0);

  *pcout << "Triangulation:"
              << std::endl
              << " Number of active cells = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}


template<int dim>
void SimpleShearProblem<dim>::setup()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem - Setup");

  // Initiates the crystals' data (Slip directions, normals, orthogonals,
  // Schmid-Tensor and symmetrized Schmid-Tensors)
  crystals_data->init(triangulation,
                      parameters.euler_angles_pathname,
                      parameters.slips_directions_pathname,
                      parameters.slips_normals_pathname);

  // Sets up the FEValuesExtractor instances
  fe_field->setup_extractors(crystals_data->get_n_crystals(),
                             crystals_data->get_n_slips());

  // Set the active finite elemente index of each cell
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(0);

  // Sets up the degrees of freedom
  fe_field->setup_dofs();

  // Instantiates the external functions, whose number of components
  // depends on the number of crystals and slips
  displacement_control =
    std::make_unique<DisplacementControl<dim>>(
      parameters.shear_strain_at_upper_boundary,
      parameters.height,
      fe_field->get_n_components(),
      discrete_time.get_start_time());

  dirichlet_boundary_function =
    std::make_unique<DirichletBoundaryFunction<dim>>(
      fe_field->get_n_components(),
      discrete_time.get_start_time());

  // Sets up the problem's constraints
  setup_constraints();

  // Sets up the solution vectors
  fe_field->setup_vectors();

  // Terminal output
  *pcout
    << "Crystals data:" << std::endl
    << " Number of crystals = " << crystals_data->get_n_crystals() << std::endl
    << " Number of slips    = " << crystals_data->get_n_slips() << std::endl
    << std::endl;

  *pcout
    << "Spatial discretization:" << std::endl
    << " Number of degrees of freedom = " << fe_field->n_dofs()
    << std::endl << std::endl;
}



template<int dim>
void SimpleShearProblem<dim>::setup_constraints()
{
  const unsigned int lower_boundary_id = 2;

  const unsigned int upper_boundary_id = 3;

  // Initiate the entity needed for periodic boundary conditions
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
      std::map<dealii::types::boundary_id,
              const dealii::Function<dim> *> function_map;

      function_map[lower_boundary_id] = dirichlet_boundary_function.get();
      function_map[upper_boundary_id] = displacement_control.get();

      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        function_map,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_displacement_extractor(0)));
    }

    // Slips' Dirichlet boundary conditions
    std::map<dealii::types::boundary_id,
             const dealii::Function<dim> *> function_map;

    function_map[lower_boundary_id] = dirichlet_boundary_function.get();
    function_map[upper_boundary_id] = dirichlet_boundary_function.get();

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

    function_map[lower_boundary_id] = &zero_function;
    function_map[upper_boundary_id] = &zero_function;

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
        newton_method_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_slip_extractor(0, slip_id)));
    }

    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      newton_method_constraints);
  }
  newton_method_constraints.close();

  // The constraints are now passed to the FEField instance
  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(newton_method_constraints);
}



template<int dim>
void SimpleShearProblem<dim>::update_dirichlet_boundary_conditions()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem - Update boundary conditions");

  // Instantiate the temporary AffineConstraint instance
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    // Interpolate the updated values of the time-dependent boundary
    // conditions
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
}



template<int dim>
void SimpleShearProblem<dim>::postprocessing()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem - Postprocessing");

  simple_shear.compute_data(discrete_time.get_current_time());

  const fs::path path{parameters.graphical_output_directory};

  fs::path filename = path / "stress12_vs_shear_strain_at_boundary.txt";

  try
  {
    std::ofstream fstream(filename.string());
    simple_shear.output_data_to_file(fstream);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the creation of the output file: "
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
    std::cerr << "Unknown exception in the creation of the output file!"
              << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
}



template<int dim>
void SimpleShearProblem<dim>::triangulation_output()
{
  dealii::Vector<float> locally_owned_subdomain(triangulation.n_active_cells());

  dealii::Vector<float> material_id(triangulation.n_active_cells());

  dealii::Vector<float> active_fe_index(triangulation.n_active_cells());

  // Fill the dealii::Vector<float> instances for visualizatino of the
  // cell properties
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
    {
      locally_owned_subdomain(cell->active_cell_index()) =
        triangulation.locally_owned_subdomain();
      material_id(cell->active_cell_index()) =
        cell->material_id();
      active_fe_index(cell->active_cell_index()) =
        cell->active_fe_index();
    }

  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(fe_field->get_dof_handler());

  data_out.add_data_vector(locally_owned_subdomain,
                           "locally_owned_subdomain");

  data_out.add_data_vector(material_id,
                           "material_id");

  data_out.add_data_vector(active_fe_index,
                           "active_fe_index");

  data_out.add_data_vector(gCP_solver.get_cell_is_at_grain_boundary_vector(),
                           "cell_is_at_grain_boundary");

  data_out.build_patches();

  data_out.write_vtu_in_parallel(
    parameters.graphical_output_directory + "triangulation.vtu",
    MPI_COMM_WORLD);
}

template<int dim>
void SimpleShearProblem<dim>::data_output()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem - Data output");

  // Explicit declaration of the velocity as a vector
  std::vector<std::string> displacement_names(dim, "displacement");
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

  // Explicit declaration of the slips as scalars
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

  data_out.add_data_vector(simple_shear.get_dof_handler(),
                           simple_shear.get_data()[0],
                           std::vector<std::string>(1, "2e12"));

  data_out.add_data_vector(simple_shear.get_dof_handler(),
                           simple_shear.get_data()[1],
                           std::vector<std::string>(1, "s12"));

  data_out.build_patches(*mapping,
                         1/*fe_field->get_displacement_fe_degree()*/,
                         dealii::DataOut<dim>::curved_inner_cells);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    parameters.graphical_output_directory + "paraview/",
    "solution",
    out_index,
    MPI_COMM_WORLD,
    5);

  out_index++;
}



template<int dim>
void SimpleShearProblem<dim>::run()
{
  // Generate/Read triangulation (Material ids have to be set here)
  make_grid();

  // Setup CrystalsData<dim>, FEField<dim>, boundary conditions,
  // and assigns the FECollection id of each cell according to the
  // material id
  setup();

  // Initiate the solver
  gCP_solver.init();

  // Output the triangulation data (Partition, Material id, etc.)
  triangulation_output();

  // Initiate the benchmark data
  simple_shear.init(gCP_solver.get_elastic_strain_law(),
                    gCP_solver.get_hooke_law());

  // Time loop. The current time at the beggining of each loop
  // corresponds to t^{n-1}
  while(discrete_time.get_current_time() < discrete_time.get_end_time())
  {
    if (parameters.verbose)
      *pcout << std::setw(string_width) << std::left
             << "Step " +
                std::to_string(discrete_time.get_step_number() + 1)
             << " | "
             << std::setprecision(5) << std::fixed << std::scientific
             << "Solving for t = "
             << std::to_string(discrete_time.get_next_time())
             << " with dt = "
             << discrete_time.get_next_step_size()
             << std::endl;

    // Update the internal time variable of all time-dependant functions
    // to t^{n}
    displacement_control->set_time(discrete_time.get_next_time());

    // Update the Dirichlet boundary conditions values to t^{n}
    update_dirichlet_boundary_conditions();

    // Solve the nonlinear system. After the call fe_field->solution
    // corresponds to the solution at t^n
    gCP_solver.solve_nonlinear_system();

    // Update the solution vectors, i.e.,
    // fe_field->old_solution = fe_field->solution
    fe_field->update_solution_vectors();

    // Advance the DiscreteTime instance to t^{n}
    discrete_time.advance_time();

    // Call to the postprocessing method
    postprocessing();

    // Call to the data output method
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

    std::string parameters_filepath;

    // The switch statement verifies that the filepath includes the
    // .prm extension. That if, a filepath was even passed to the
    // executable. The existance of the filepath is checked by the
    // gCP::RunTimeParameters::ProblemParameters class
    switch (argc)
    {
    case 1:
      parameters_filepath = "input/simple_shear.prm";
      break;
    case 2:
      {
        const std::string arg(argv[1]);

        if (arg.find_last_of(".") != std::string::npos)
        {
          if (arg.substr(arg.find_last_of(".")+1) == "prm")
            parameters_filepath = arg;
          else
            AssertThrow(false,
                        dealii::ExcMessage(
                          "The filepath to the parameters file has to "
                          "be passed with its .prm extension."));
        }
        else
          AssertThrow(false,
                      dealii::ExcMessage(
                        "The filepath to the parameters file has to be "
                        "passed with its .prm extension."));
      }
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage(
                    "More than one argument are being passed to the "
                    "executable. Only one argument (the filepath to "
                    "the parameters file) is currently supported."));
      break;
    }

    gCP::RunTimeParameters::SimpleShearParameters parameters(parameters_filepath);

    gCP::SimpleShearProblem<2> problem(parameters);

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