#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/gradient_crystal_plasticity.h>
#include <gCP/postprocessing.h>
#include <gCP/run_time_parameters.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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
class NeumannBoundaryFunction : public dealii::TensorFunction<1,dim>
{
public:
  NeumannBoundaryFunction(
    const double                          max_traction_at_upper_boundary,
    const double                          min_traction_at_upper_boundary,
    const double                          period,
    const double                          initial_loading_time,
    const RunTimeParameters::LoadingType  loading_type,
    const double                          start_time);

  virtual dealii::Tensor<1, dim> value(
    const dealii::Point<dim>  &point) const override;

private:

  const double                          max_traction_at_upper_boundary;

  const double                          min_traction_at_upper_boundary;

  const double                          period;

  const double                          initial_loading_time;

  const RunTimeParameters::LoadingType  loading_type;
};



template<int dim>
NeumannBoundaryFunction<dim>::NeumannBoundaryFunction(
  const double                          max_traction_at_upper_boundary,
  const double                          min_traction_at_upper_boundary,
  const double                          period,
  const double                          initial_loading_time,
  const RunTimeParameters::LoadingType  loading_type,
  const double                          start_time)
:
dealii::TensorFunction<1, dim>(start_time),
max_traction_at_upper_boundary(max_traction_at_upper_boundary),
min_traction_at_upper_boundary(min_traction_at_upper_boundary),
period(period),
initial_loading_time(initial_loading_time),
loading_type(loading_type)
{}



template<int dim>
dealii::Tensor<1, dim> NeumannBoundaryFunction<dim>::value(
  const dealii::Point<dim>  &/*point*/) const
{
  const double t = this->get_time();

  dealii::Tensor<1, dim> return_vector;

  double traction_load = 0.0;

  switch (loading_type)
  {
    case RunTimeParameters::LoadingType::Monotonic:
      {
        traction_load =
          t * max_traction_at_upper_boundary;
      }
      break;
    case RunTimeParameters::LoadingType::Cyclic:
      {
        if (t >= initial_loading_time)
          traction_load =
            (max_traction_at_upper_boundary -
             min_traction_at_upper_boundary) / 2.0 *
            std::cos(2.0 * M_PI / period * (t - initial_loading_time)) +
            (max_traction_at_upper_boundary +
             min_traction_at_upper_boundary) / 2.0;
        else
          traction_load =
            max_traction_at_upper_boundary *
            std::sin(M_PI / 2.0 / initial_loading_time * t);
      }
      break;
    default:
      Assert(false, dealii::ExcNotImplemented());
  }

  return_vector[1] = traction_load;

  return return_vector;
}



template <int dim>
class DisplacementControl : public dealii::Function<dim>
{
public:
  DisplacementControl(
    const unsigned int                    n_crystals,
    const double                          height,
    const double                          max_shear_strain_at_upper_boundary,
    const double                          min_shear_strain_at_upper_boundary,
    const double                          period,
    const double                          initial_loading_time,
    const RunTimeParameters::LoadingType  loading_type,
    const bool                            flag_is_decohesion_allowed,
    const unsigned int                    n_components,
    const double                          start_time);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
  const unsigned int                    n_crystals;

  const double                          height;

  const double                          max_shear_strain_at_upper_boundary;

  const double                          min_shear_strain_at_upper_boundary;

  const double                          period;

  const double                          initial_loading_time;

  const RunTimeParameters::LoadingType  loading_type;

  const bool                            flag_is_decohesion_allowed;
};



template<int dim>
DisplacementControl<dim>::DisplacementControl(
  const unsigned int                    n_crystals,
  const double                          height,
  const double                          max_shear_strain_at_upper_boundary,
  const double                          min_shear_strain_at_upper_boundary,
  const double                          period,
  const double                          initial_loading_time,
  const RunTimeParameters::LoadingType  loading_type,
  const bool                            flag_is_decohesion_allowed,
  const unsigned int                    n_components,
  const double                          start_time)
:
dealii::Function<dim>(n_components, start_time),
n_crystals(n_crystals),
height(height),
max_shear_strain_at_upper_boundary(max_shear_strain_at_upper_boundary),
min_shear_strain_at_upper_boundary(min_shear_strain_at_upper_boundary),
period(period),
initial_loading_time(initial_loading_time + start_time),
loading_type(loading_type),
flag_is_decohesion_allowed(flag_is_decohesion_allowed)
{}



template<int dim>
void DisplacementControl<dim>::vector_value(
  const dealii::Point<dim>  &/*point*/,
  dealii::Vector<double>    &return_vector) const
{
  const double t = this->get_time();

  return_vector = 0.0;

  double displacement_load = 0.0;

  switch (loading_type)
  {
    case RunTimeParameters::LoadingType::Monotonic:
      {
        displacement_load =
          t * height * max_shear_strain_at_upper_boundary;
      }
      break;
    case RunTimeParameters::LoadingType::Cyclic:
      {
        if (t >= initial_loading_time)
          displacement_load =
            (max_shear_strain_at_upper_boundary -
             min_shear_strain_at_upper_boundary) / 2.0 *
            std::cos(2.0 * M_PI / period * (t - initial_loading_time)) +
            (max_shear_strain_at_upper_boundary +
             min_shear_strain_at_upper_boundary) / 2.0;
        else
          displacement_load =
            max_shear_strain_at_upper_boundary *
            std::sin(M_PI / 2.0 / initial_loading_time * t);
      }
      break;
    default:
      Assert(false, dealii::ExcNotImplemented());
  }

  return_vector[0] = displacement_load;

  if (flag_is_decohesion_allowed)
    for (unsigned int i = 1; i < n_crystals; ++i)
      return_vector[i*dim] = displacement_load;
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

  std::shared_ptr<NeumannBoundaryFunction<dim>>     neumann_boundary_function;

  std::unique_ptr<DisplacementControl<dim>>         displacement_control;

  Postprocessing::Postprocessor<dim>                postprocessor;

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
discrete_time(
  parameters.temporal_discretization_parameters.start_time,
  parameters.temporal_discretization_parameters.end_time,
  parameters.temporal_discretization_parameters.time_step_size),
triangulation(
  MPI_COMM_WORLD,
  typename dealii::Triangulation<dim>::MeshSmoothing(
  dealii::Triangulation<dim>::smoothing_on_refinement |
  dealii::Triangulation<dim>::smoothing_on_coarsening)),
fe_field(std::make_shared<FEField<dim>>(
  triangulation,
  parameters.fe_degree_displacements,
  parameters.fe_degree_slips,
  parameters.solver_parameters.allow_decohesion)),
crystals_data(std::make_shared<CrystalsData<dim>>()),
gCP_solver(parameters.solver_parameters,
           discrete_time,
           fe_field,
           crystals_data,
           mapping,
           pcout,
           timer_output,
           parameters.temporal_discretization_parameters.loading_type),
neumann_boundary_function(
  std::make_shared<NeumannBoundaryFunction<dim>>(
  parameters.max_shear_strain_at_upper_boundary * 10000.0,
  parameters.min_shear_strain_at_upper_boundary * 10000.0,
  parameters.temporal_discretization_parameters.period,
  parameters.temporal_discretization_parameters.initial_loading_time,
  parameters.temporal_discretization_parameters.loading_type,
  discrete_time.get_start_time())),
postprocessor(fe_field,
              crystals_data),
simple_shear(fe_field,
             mapping,
             parameters.max_shear_strain_at_upper_boundary,
             parameters.min_shear_strain_at_upper_boundary,
             parameters.temporal_discretization_parameters.period,
             parameters.temporal_discretization_parameters.initial_loading_time,
             parameters.temporal_discretization_parameters.loading_type,
             3,
             parameters.width),
string_width(
  (std::to_string((unsigned int)(
  (parameters.temporal_discretization_parameters.end_time -
   parameters.temporal_discretization_parameters.start_time) /
  parameters.temporal_discretization_parameters.time_step_size)) +
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

  std::vector<unsigned int> repetitions(2, 10);
  repetitions[1] = 100;

  switch (dim)
  {
  case 2:
    {
      /*
      dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        repetitions,
        dealii::Point<dim>(0,0),
        dealii::Point<dim>(parameters.width, parameters.height),
        true);*/

      dealii::GridIn<dim> grid_in;

      grid_in.attach_triangulation(triangulation);

      std::ifstream input_file("tests/tess.msh");

      grid_in.read_msh(input_file);
    }
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
  /*
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  dealii::GridTools::collect_periodic_faces(triangulation,
                                            0,
                                            1,
                                            0,
                                            periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);
  */
  triangulation.refine_global(parameters.n_global_refinements);
  /*
  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      for (unsigned int i = 1;
           i <= parameters.n_equal_sized_divisions; ++i)
        if (std::fabs(cell->center()[1]) <
            i * parameters.height / parameters.n_equal_sized_divisions)
        {
          cell->set_material_id(i-1);
          break;
        }
    }
  */

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell->set_material_id(cell->material_id() - 1);
    }

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          if (face->vertex(0)[1] == 0.0)
            face->set_boundary_id(2);
          else if (face->vertex(0)[1] == 1.0)
            face->set_boundary_id(3);
          else
            face->set_boundary_id(0);
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
      cell->set_active_fe_index(cell->material_id());

  // Sets up the degrees of freedom
  fe_field->setup_dofs();

  // Instantiates the external functions, whose number of components
  // depends on the number of crystals and slips
  displacement_control =
    std::make_unique<DisplacementControl<dim>>(
      crystals_data->get_n_crystals(),
      parameters.height,
      parameters.max_shear_strain_at_upper_boundary,
      parameters.min_shear_strain_at_upper_boundary,
      parameters.temporal_discretization_parameters.period,
      parameters.temporal_discretization_parameters.initial_loading_time,
      parameters.temporal_discretization_parameters.loading_type,
      fe_field->is_decohesion_allowed(),
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
  /*
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
  */
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

      for (unsigned int crystal_id = 0;
           crystal_id < crystals_data->get_n_crystals();
           ++crystal_id)
        dealii::VectorTools::interpolate_boundary_values(
          *mapping,
          fe_field->get_dof_handler(),
          function_map,
          affine_constraints,
          fe_field->get_fe_collection().component_mask(
            fe_field->get_displacement_extractor(crystal_id)));
    }

    // Slips' Dirichlet boundary conditions
    {
      std::map<dealii::types::boundary_id,
              const dealii::Function<dim> *> function_map;

      function_map[lower_boundary_id] = dirichlet_boundary_function.get();
      function_map[upper_boundary_id] = dirichlet_boundary_function.get();

      for (unsigned int crystal_id = 0;
           crystal_id < crystals_data->get_n_crystals();
           ++crystal_id)
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
  /*
    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      affine_constraints);**/
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
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        function_map,
        newton_method_constraints,
        fe_field->get_fe_collection().component_mask(
        fe_field->get_displacement_extractor(crystal_id)));

    // Slips' Dirichlet boundary conditions
    for (unsigned int crystal_id = 0;
          crystal_id < crystals_data->get_n_crystals();
          ++crystal_id)
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
    /*
    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      newton_method_constraints);*/
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
    for (unsigned int crystal_id = 0;
          crystal_id < crystals_data->get_n_crystals();
          ++crystal_id)
      dealii::VectorTools::interpolate_boundary_values(
        *mapping,
        fe_field->get_dof_handler(),
        3,
        *displacement_control,
        affine_constraints,
        fe_field->get_fe_collection().component_mask(
          fe_field->get_displacement_extractor(crystal_id)));
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

  locally_owned_subdomain = -1.0;
  material_id             = -1.0;
  active_fe_index         = -1.0;

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

  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(fe_field->get_dof_handler());

  data_out.add_data_vector(fe_field->solution, postprocessor);

  data_out.build_patches(*mapping,
                         fe_field->get_displacement_fe_degree()+2/*,
                         dealii::DataOut<dim>::curved_inner_cells*/);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    parameters.graphical_output_directory + "paraview/",
    "solution",
    out_index,
    MPI_COMM_WORLD,
    5);

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
    parameters.graphical_output_directory + "paraview/",
    "damage",
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

  postprocessor.init(gCP_solver.get_hooke_law());

  if (parameters.temporal_discretization_parameters.loading_type ==
        RunTimeParameters::LoadingType::Cyclic)
    discrete_time.set_desired_next_step_size(
      parameters.temporal_discretization_parameters.time_step_size_in_loading_phase);

  // Time loop. The current time at the beggining of each loop
  // corresponds to t^{n-1}
  while(discrete_time.get_current_time() < discrete_time.get_end_time())
  {
    if (parameters.temporal_discretization_parameters.loading_type ==
        RunTimeParameters::LoadingType::Cyclic &&
        discrete_time.get_current_time() ==
        parameters.temporal_discretization_parameters.initial_loading_time)
      discrete_time.set_desired_next_step_size(
        parameters.temporal_discretization_parameters.time_step_size);

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
      argc, argv, 1);

    std::string parameters_filepath;

    // The switch statement verifies that the filepath includes the
    // .prm extension. That if, a filepath was even passed to the
    // executable. The existance of the filepath is checked by the
    // gCP::RunTimeParameters::ProblemParameters class
    switch (argc)
    {
    case 1:
      parameters_filepath = "input/simple_shear_poly.prm";
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

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
      std::cout << "Running with \""
                << parameters_filepath << "\"" << "\n";

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