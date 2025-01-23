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



template <int dim>
class SlipFields : public dealii::Function<dim>
{
public:
  SlipFields(const unsigned int n_components)
  :
  dealii::Function<dim>(n_components)
  {
    //std::cout << n_components << std::endl;
  }

  virtual void vector_value(
    const dealii::Point<dim> &point,
    dealii::Vector<double> &return_vector) const override
  {
    return_vector = 0.0;

    const std::vector<double> nodal_values =
      {0.12, 1.68, 0.86, 1.64};

    const double xi = point[0], eta = point[1];

    const double slip_value =
      nodal_values[0] * (xi - 1.) * (eta - 1.) +
      nodal_values[1] * (xi - xi * eta) +
      nodal_values[2] * (eta - xi * eta) +
      nodal_values[3] * (xi * eta);

    for (unsigned int i = dim; i < this->n_components; ++i)
    {
      return_vector(i) = slip_value;
    }
  }
};



template <int dim>
class OldSlipFields : public dealii::Function<dim>
{
public:
  OldSlipFields(const unsigned int n_components)
  :
  dealii::Function<dim>(n_components)
  {}

  virtual void vector_value(
    const dealii::Point<dim> &point,
    dealii::Vector<double> &return_vector) const override
  {
    return_vector = 0.0;

    const std::vector<double> nodal_values =
    //{0.12, 1.68, 0.86, 1.64};
    {0.77, 1.00, 1.04, 0.41};

    const double xi = point[0], eta = point[1];

    const double slip_value =
      nodal_values[0] * (xi - 1.) * (eta - 1.) +
      nodal_values[1] * (xi - xi * eta) +
      nodal_values[2] * (eta - xi * eta) +
      nodal_values[3] * (xi * eta);

    for (unsigned int i = dim; i < this->n_components; ++i)
    {
      return_vector(i) = slip_value;
    }
  }
};



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

  const double                                      string_width;

  void make_grid();

  void setup();

  void initialize_calls();

  void solve();
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
string_width(
  (std::to_string((unsigned int)(
  (parameters.simple_loading.end_time -
   parameters.simple_loading.start_time) /
  parameters.simple_loading.time_step_size)) +
  "Step ").size())
{}



template<int dim>
void SimpleShearProblem<dim>::make_grid()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Triangulation");

  dealii::GridGenerator::hyper_rectangle(
    triangulation,
    dealii::Point<dim>(0.,0.),
    dealii::Point<dim>(1.,1.),
    true);

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
void SimpleShearProblem<dim>::initialize_calls()
{
  // Initiate the solver
  gCP_solver.init();
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

  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());
  }
  affine_constraints.close();

  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(affine_constraints);

  // Call the init() methods of the class' members
  initialize_calls();

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_vector =
      fe_field->get_distributed_vector_instance(
        fe_field->solution);

  dealii::VectorTools::interpolate(
    fe_field->get_dof_handler(),
    SlipFields<dim>(fe_field->get_n_components()),
    distributed_vector,
    fe_field->get_fe_collection().component_mask(
      fe_field->get_slip_extractor(0, 0)));

  dealii::VectorTools::interpolate(
    fe_field->get_dof_handler(),
    SlipFields<dim>(fe_field->get_n_components()),
    distributed_vector,
    fe_field->get_fe_collection().component_mask(
      fe_field->get_slip_extractor(0, 1)));

  fe_field->get_affine_constraints().distribute(distributed_vector);

  fe_field->solution = distributed_vector;

  distributed_vector = 0.0;

  dealii::VectorTools::interpolate(
    fe_field->get_dof_handler(),
    OldSlipFields<dim>(fe_field->get_n_components()),
    distributed_vector,
    fe_field->get_fe_collection().component_mask(
      fe_field->get_slip_extractor(0, 0)));

  dealii::VectorTools::interpolate(
    fe_field->get_dof_handler(),
    OldSlipFields<dim>(fe_field->get_n_components()),
    distributed_vector,
    fe_field->get_fe_collection().component_mask(
      fe_field->get_slip_extractor(0, 1)));

  fe_field->get_affine_constraints().distribute(distributed_vector);

  fe_field->old_solution = distributed_vector;

  gCP_solver.compute_difference_quotients_jacobian_approximation();
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
        parameters_filepath = "input/parameter_files/jacobian_parameters.prm";
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