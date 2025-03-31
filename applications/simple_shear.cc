#include <gCP/assembly_data.h>
#include <gCP/boundary_conditions.h>
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
#include <deal.II/numerics/data_out_faces.h>
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



template<int dim>
class SimpleShearProblem
{
public:
  SimpleShearProblem(
    const RunTimeParameters::InfiniteStripProblem &parameters);

  void run();

private:
  const RunTimeParameters::InfiniteStripProblem     parameters;

  std::shared_ptr<dealii::ConditionalOStream>       pcout;

  std::shared_ptr<dealii::TimerOutput>              timer_output;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::DiscreteTime                              discrete_time;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  std::shared_ptr<FEField<dim>>                     fe_field;

  std::shared_ptr<CrystalsData<dim>>                crystals_data;

  GradientCrystalPlasticitySolver<dim>              gCP_solver;

  //std::shared_ptr<NeumannBoundaryFunction<dim>>     neumann_boundary_function;

  std::unique_ptr<BoundaryConditions::DisplacementControl<dim>>
                                                    displacement_control;

  Postprocessing::Homogenization<dim>               homogenization;

  Postprocessing::Postprocessor<dim>                postprocessor;

  Postprocessing::SimpleShear<dim>                  simple_shear;

  const double                                      string_width;

  void make_grid();

  void update_dirichlet_boundary_conditions();

  void setup();

  void setup_constraints();

  void initialize_calls();

  void solve();

  void postprocessing();

  void triangulation_output();

  void data_output();
};



template<int dim>
SimpleShearProblem<dim>::SimpleShearProblem(
  const RunTimeParameters::InfiniteStripProblem &parameters)
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
  parameters.spatial_discretization.mapping_degree)),
discrete_time(
  parameters.simple_loading.start_time,
  parameters.simple_loading.end_time,
  parameters.simple_loading.time_step_size),
triangulation(
  MPI_COMM_WORLD,
  typename dealii::Triangulation<dim>::MeshSmoothing(
  dealii::Triangulation<dim>::smoothing_on_refinement |
  dealii::Triangulation<dim>::smoothing_on_coarsening)),
fe_field(std::make_shared<FEField<dim>>(
  triangulation,
  parameters.spatial_discretization.fe_degree_displacements,
  parameters.spatial_discretization.fe_degree_slips,
  parameters.solver_parameters.allow_decohesion,
  parameters.solver_parameters.solution_algorithm ==
    RunTimeParameters::SolutionAlgorithm::Monolithic &&
      (parameters.solver_parameters.monolithic_algorithm_parameters.
        monolithic_system_solver_parameters.
          krylov_parameters.solver_type ==
            RunTimeParameters::SolverType::DirectSolver ||
       parameters.solver_parameters.monolithic_algorithm_parameters.
        monolithic_preconditioner ==
        RunTimeParameters::MonolithicPreconditioner::BuiltIn))),
crystals_data(std::make_shared<CrystalsData<dim>>()),
gCP_solver(
  parameters.solver_parameters,
  discrete_time,
  fe_field,
  crystals_data,
  mapping,
  pcout,
  timer_output),
/*neumann_boundary_function(
  std::make_shared<NeumannBoundaryFunction<dim>>(
  parameters.max_shear_strain_at_upper_boundary * 10000.0,
  parameters.min_shear_strain_at_upper_boundary * 10000.0,
  parameters.temporal_discretization_parameters.period,
  parameters.temporal_discretization_parameters.initial_loading_time,
  parameters.temporal_discretization_parameters.loading_type,
  discrete_time.get_start_time())),*/
homogenization(
  fe_field,
  mapping),
postprocessor(
  fe_field,
  crystals_data,
  parameters.solver_parameters.dimensionless_form_parameters,
  false,
  parameters.output.flag_output_dimensionless_quantities),
simple_shear(
  fe_field,
  mapping,
  parameters.simple_loading.max_load,
  parameters.simple_loading.min_load,
  parameters.simple_loading.period,
  parameters.simple_loading.duration_loading_and_unloading_phase,
  parameters.simple_loading.loading_type,
  3,
  1. / parameters.n_elements_in_y_direction),
string_width(
  (std::to_string((unsigned int)(
  (parameters.simple_loading.end_time -
   parameters.simple_loading.start_time) /
  parameters.simple_loading.time_step_size)) +
  "Step ").size())
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    if (fs::exists(parameters.output.output_directory + "paraview/"))
    {
      *pcout
        << "Deleting *.vtu and *.pvtu files inside the output folder... "
        << std::flush;

      for (const auto& entry : fs::directory_iterator(parameters.output.output_directory + "paraview/"))
      {
        if (entry.path().extension() == ".vtu" ||
            entry.path().extension() == ".pvtu")
        {
          fs::remove(entry.path());
        }
      }

      *pcout
        << "done!\n\n";
    }
  }
}



template<int dim>
void SimpleShearProblem<dim>::make_grid()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Triangulation");

  std::vector<unsigned int> repetitions(2, 1);
  repetitions[1] = parameters.n_elements_in_y_direction;

  switch (dim)
  {
  case 2:
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0),
      dealii::Point<dim>(1. / parameters.n_elements_in_y_direction, parameters.height),
      true);
    break;
  case 3:
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        repetitions,
        dealii::Point<dim>(0,0,0),
        dealii::Point<dim>(1./parameters.n_elements_in_y_direction, 1./parameters.n_elements_in_y_direction),
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

  triangulation.refine_global(
    parameters.spatial_discretization.n_global_refinements);

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      for (unsigned int i = 1;
           i <= parameters.n_equal_sized_crystals; ++i)
      {
        if (std::fabs(cell->center()[1]) <
            i * parameters.height / parameters.n_equal_sized_crystals)
        {
          cell->set_material_id(i-1);
          break;
        }
      }
    }
  }

  *pcout << "Triangulation:"
              << std::endl
              << " Number of active cells = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}


template<int dim>
void SimpleShearProblem<dim>::setup()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Setup");

  // Initiates the crystals' data (Slip directions, normals, orthogonals,
  // Schmid-Tensor and symmetrized Schmid-Tensors)
  crystals_data->init(triangulation,
                      parameters.input.euler_angles_pathname,
                      parameters.input.slips_directions_pathname,
                      parameters.input.slips_normals_pathname);

  // Sets up the FEValuesExtractor instances
  fe_field->setup_extractors(crystals_data->get_n_crystals(),
                             crystals_data->get_n_slips());

  // Update the material ids of ghost cells
  fe_field->update_ghost_material_ids();

  // Set the active finite elemente index of each cell
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      cell->set_active_fe_index(cell->material_id());
    }
  }

  // Sets up the degrees of freedom
  fe_field->setup_dofs();

  // Instantiates the external functions, whose number of components
  // depends on the number of crystals and slips
  displacement_control =
    std::make_unique<BoundaryConditions::DisplacementControl<dim>>(
      parameters.simple_loading,
      fe_field->get_n_components(),
      crystals_data->get_n_crystals(),
      parameters.height,
      fe_field->is_decohesion_allowed(),
      parameters.solver_parameters.dimensionless_form_parameters.
        characteristic_quantities.displacement);

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
    << " Number of total degrees of freedom         = "
    << fe_field->n_dofs() << std::endl
    << " Number of displacement degrees of freedom  = "
    << fe_field->get_n_displacement_dofs() << std::endl
    << " Number of plastic slips degrees of freedom = "
    << fe_field->get_n_plastic_slip_dofs() << std::endl << std::endl;
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

  dealii::Functions::ZeroFunction<dim> zero_function(
                                        fe_field->get_n_components());

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    // Displacements' Dirichlet boundary conditions
    {
      std::map<dealii::types::boundary_id,
              const dealii::Function<dim> *> function_map;

      function_map[lower_boundary_id] = &zero_function;
      function_map[upper_boundary_id] = displacement_control.get();

      for (unsigned int crystal_id = 0;
           crystal_id < crystals_data->get_n_crystals();
           ++crystal_id)
      {
        dealii::VectorTools::interpolate_boundary_values(
          *mapping,
          fe_field->get_dof_handler(),
          function_map,
          affine_constraints,
          fe_field->get_fe_collection().component_mask(
            fe_field->get_displacement_extractor(crystal_id)));
      }
    }

    // Slips' Dirichlet boundary conditions
    {
      std::map<dealii::types::boundary_id,
              const dealii::Function<dim> *> function_map;

      function_map[lower_boundary_id] = &zero_function;
      function_map[upper_boundary_id] = &zero_function;

      for (unsigned int crystal_id = 0;
           crystal_id < crystals_data->get_n_crystals();
           ++crystal_id)
      {
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
              fe_field->get_slip_extractor(crystal_id, slip_id)));
        }
      }
    }

    if (parameters.solver_parameters.boundary_conditions_at_grain_boundaries ==
          RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microhard)
    {
      std::vector<dealii::types::global_dof_index> local_face_dof_indices(
        fe_field->get_fe_collection().max_dofs_per_face());

      for (const auto &cell :
           fe_field->get_dof_handler().active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          for (const auto &face_index : cell->face_indices())
          {
            if (!cell->face(face_index)->at_boundary() &&
                cell->material_id() !=
                  cell->neighbor(face_index)->material_id())
            {
              AssertThrow(
                cell->neighbor(face_index)->active_fe_index() ==
                  cell->neighbor(face_index)->material_id(),
                dealii::ExcMessage(
                  "The active finite element index and the material "
                  " identifier of the cell have to coincide!"));

              const unsigned int crystal_id = cell->material_id();

              cell->face(face_index)->get_dof_indices(
                local_face_dof_indices,
                crystal_id);

              for (unsigned int i = 0;
                   i < local_face_dof_indices.size(); ++i)
              {
                if (fe_field->get_global_component(crystal_id, i) >= dim)
                {
                  affine_constraints.add_line(
                    local_face_dof_indices[i]);
                }
              }
            }
          }
        }
      }
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
    for (unsigned int crystal_id = 0;
          crystal_id < crystals_data->get_n_crystals();
          ++crystal_id)
    {
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        function_map,
        newton_method_constraints,
        fe_field->get_fe_collection().component_mask(
        fe_field->get_displacement_extractor(crystal_id)));
    }

    // Slips' Dirichlet boundary conditions
    for (unsigned int crystal_id = 0;
          crystal_id < crystals_data->get_n_crystals();
          ++crystal_id)
    {
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
            fe_field->get_slip_extractor(crystal_id, slip_id)));
      }
    }

    if (parameters.solver_parameters.boundary_conditions_at_grain_boundaries ==
          RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microhard)
    {
      std::vector<dealii::types::global_dof_index> local_face_dof_indices(
        fe_field->get_fe_collection().max_dofs_per_face());

      for (const auto &cell :
           fe_field->get_dof_handler().active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          for (const auto &face_index : cell->face_indices())
          {
            if (!cell->face(face_index)->at_boundary() &&
                cell->material_id() !=
                  cell->neighbor(face_index)->material_id())
            {
              AssertThrow(
                cell->neighbor(face_index)->active_fe_index() ==
                  cell->neighbor(face_index)->material_id(),
                dealii::ExcMessage(
                  "The active finite element index and the material "
                  " identifier of the cell have to coincide!"));

              const unsigned int crystal_id = cell->material_id();

              cell->face(face_index)->get_dof_indices(
                local_face_dof_indices,
                crystal_id);

              for (unsigned int i = 0;
                   i < local_face_dof_indices.size(); ++i)
              {
                if (fe_field->get_global_component(crystal_id, i) >= dim)
                {
                  newton_method_constraints.add_line(
                    local_face_dof_indices[i]);
                }
              }
            }
          }
        }
      }
    }

    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      newton_method_constraints);
  }
  newton_method_constraints.close();

  // The constraints are now passed to the FEField instance
  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(newton_method_constraints);

  // Neumann boundary conditions
  //gCP_solver.set_neumann_boundary_condition(
  //  upper_boundary_id, neumann_boundary_function);
}

template<int dim>
void SimpleShearProblem<dim>::initialize_calls()
{
  // Initiate the solver
  gCP_solver.init();

  const fs::path output_directory{parameters.output.output_directory};

  fs::path path_to_ouput_file =
    output_directory / "homogenization.txt";

  std::ofstream ofstream(path_to_ouput_file.string());

  // Initiate the benchmark data
  homogenization.init(gCP_solver.get_elastic_strain_law(),
                      gCP_solver.get_hooke_law(),
                      ofstream);

  simple_shear.init(gCP_solver.get_elastic_strain_law(),
                    gCP_solver.get_hooke_law());

  postprocessor.init(gCP_solver.get_hooke_law());
}


template<int dim>
void SimpleShearProblem<dim>::update_dirichlet_boundary_conditions()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Update boundary conditions");

  // Instantiate the temporary AffineConstraint instance
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    // Interpolate the updated values of the time-dependent boundary
    // conditions
    for (unsigned int crystal_id = 0;
          crystal_id < crystals_data->get_n_crystals();
          ++crystal_id)
    {
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        3,
        *displacement_control,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_displacement_extractor(crystal_id)));
    }
  }
  affine_constraints.close();

  fe_field->set_affine_constraints(affine_constraints);
}



template<int dim>
void SimpleShearProblem<dim>::postprocessing()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Postprocessing");

  if (parameters.homogenization.flag_compute_homogenized_quantities &&
      (discrete_time.get_step_number() %
         parameters.homogenization.homogenization_frequency == 0 ||
        discrete_time.get_current_time() ==
          discrete_time.get_end_time()))
  {
    homogenization.compute_macroscopic_quantities(
      discrete_time.get_current_time());
  }

  simple_shear.compute_data(discrete_time.get_current_time());

  const fs::path path{parameters.output.output_directory};

  fs::path filename = path / "stress12_vs_shear_strain_at_boundary.txt";

  fs::path filename_ = path / "solution_algorithm_statistics.txt";

  try
  {
    std::ofstream fstream(filename.string());

    std::ofstream fstream_(filename_.string());

    simple_shear.output_data_to_file(fstream);

    gCP_solver.output_data_to_file(fstream_);
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

  locally_owned_subdomain = -1.0;

  material_id             = -1.0;

  active_fe_index         = -1.0;

  // Fill the dealii::Vector<float> instances for visualizatino of the
  // cell properties
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      locally_owned_subdomain(cell->active_cell_index()) =
        triangulation.locally_owned_subdomain();
      material_id(cell->active_cell_index()) =
        cell->material_id();
      active_fe_index(cell->active_cell_index()) =
        cell->active_fe_index();
    }
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
    parameters.output.output_directory + "triangulation.vtu",
    MPI_COMM_WORLD);
}

template<int dim>
void SimpleShearProblem<dim>::data_output()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Data output");

  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(fe_field->get_dof_handler());

  data_out.add_data_vector(fe_field->solution, postprocessor);

  data_out.build_patches(*mapping,
                         0,//fe_field->get_displacement_fe_degree(),
                         dealii::DataOut<dim>::curved_inner_cells);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    parameters.output.output_directory + "paraview/",
    "solution",
    out_index,
    MPI_COMM_WORLD,
    5);

  if (parameters.output.flag_output_damage_variable)
  {
    dealii::DataOutFaces<dim>        data_out_face(false);

    std::vector<std::string> face_name(1, "damage");

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(1, dealii::DataComponentInterpretation::component_is_scalar);

    data_out_face.add_data_vector(
      gCP_solver.get_projection_dof_handler(),
      gCP_solver.get_damage_at_grain_boundaries(),
      face_name,
      face_component_type);

    data_out_face.build_patches(2);
    data_out_face.write_vtu_with_pvtu_record(
      parameters.output.output_directory + "paraview/",
      "damage",
      out_index,
      MPI_COMM_WORLD,
      5);
  }

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

  // Call the init() methods of the class' members
  initialize_calls();

  // Output the triangulation data (Partition, Material id, etc.)
  triangulation_output();

  // Time loop. The current time at the beggining of each loop
  // corresponds to t^{n-1}
  while(discrete_time.get_current_time() < discrete_time.get_end_time())
  {
    discrete_time.set_desired_next_step_size(
      parameters.simple_loading.get_next_time_step_size(
        discrete_time.get_step_number()));

    if (parameters.verbose)
    {
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
    }

    // Update the internal time variable of all time-dependant functions
    // to t^{n}
    displacement_control->set_time(discrete_time.get_next_time());

    // Update the Dirichlet boundary conditions values to t^{n}
    update_dirichlet_boundary_conditions();

    // Solve the nonlinear system. After the call fe_field->solution
    // corresponds to the solution at t^n
    gCP_solver.solve_nonlinear_system(
      parameters.simple_loading.skip_extrapolation(
        discrete_time.get_step_number()));

    // Update the solution vectors, i.e.,
    // fe_field->old_solution = fe_field->solution
    fe_field->update_solution_vectors();

    // Advance the DiscreteTime instance to t^{n}
    discrete_time.advance_time();

    // Call to the postprocessing method
    postprocessing();

    // Call to the data output method
    if (discrete_time.get_step_number() %
         parameters.output.graphical_output_frequency == 0 ||
        discrete_time.get_current_time() ==
          discrete_time.get_end_time())
    {
      data_output();
    }
  }

  homogenization.output_macroscopic_quantities_to_file();
}



} // namespace gCP


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    std::string parameters_filepath;

    // The following switch mainly checks if the filepath includes the
    // .prm extension. The existance of the filepath is checked by the
    // constructor of the problem's parameter struct.
    switch (argc)
    {
      case 1:
      {
        parameters_filepath = "input/parameter_files/simple_shear.prm";
      }
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
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      "More than one argument are being passed to the "
                      "executable. Only one argument (the filepath to "
                      "the parameters file) is currently supported."));
      }
      break;
    }

    { // Terminal output of the parameter file's filepath
      int rank;

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == 0)
      {
        std::cout << "Running with \""
                  << parameters_filepath << "\"" << "\n";
      }
    }

    gCP::RunTimeParameters::InfiniteStripProblem parameters(parameters_filepath);

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