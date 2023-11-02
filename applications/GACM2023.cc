#include <gCP/assembly_data.h>
#include <gCP/constitutive_laws.h>
#include <gCP/fe_field.h>
#include <gCP/gradient_crystal_plasticity.h>
#include <gCP/postprocessing.h>
#include <gCP/run_time_parameters.h>

#include <deal.II/distributed/solution_transfer.h>

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
class TractionVector : public dealii::TensorFunction<1,dim>
{
public:

  TractionVector(
    const gCP::RunTimeParameters::SemicoupledParameters parameters);

  virtual dealii::Tensor<1, dim> value(
    const dealii::Point<dim>  &point) const override;

private:

  const double                    period;

  const double                    mean_value;

  const double                    amplitude;

  const double                    n_cycles;

  const double                    start_of_cyclic_phase;

  const double                    start_of_unloading_phase;

  const double                    unloading_and_unloading_phase_duration;
};



template<int dim>
TractionVector<dim>::TractionVector(
  const gCP::RunTimeParameters::SemicoupledParameters parameters)
:
dealii::TensorFunction<1, dim>(
  parameters.temporal_discretization_parameters.start_time),
period(parameters.temporal_discretization_parameters.period),
mean_value(100),
amplitude(25),
n_cycles(parameters.temporal_discretization_parameters.n_cycles),
start_of_cyclic_phase(parameters.temporal_discretization_parameters.start_of_cyclic_phase),
start_of_unloading_phase(parameters.temporal_discretization_parameters.start_of_unloading_phase),
unloading_and_unloading_phase_duration(parameters.temporal_discretization_parameters.unloading_and_unloading_phase_duration)
{}



template<int dim>
dealii::Tensor<1, dim> TractionVector<dim>::value(
  const dealii::Point<dim>  &point) const
{
  const double time = this->get_time();

  dealii::Tensor<1, dim> return_vector;

  if (point[1] > 3.247595 || point[1] < 2.814585) //< 2.59808)
  {
    return return_vector;
  }

  double traction_load = 0.0;

  if (time < start_of_cyclic_phase)
  {
    const double factor = time / unloading_and_unloading_phase_duration;

    traction_load = factor * mean_value;
  }
  else if (time < start_of_unloading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI / period * (time - start_of_cyclic_phase));

    traction_load = mean_value + factor * amplitude;
  }
  else
  {
    const double factor =
      1.0 - (time - start_of_unloading_phase) /
        unloading_and_unloading_phase_duration ;

    traction_load = factor * mean_value;
  }

  return_vector[0] = traction_load;
  //return_vector[1] = -2.0 * traction_load;

  return return_vector;
}



template <int dim>
class LinearDisplacement : public dealii::Function<dim>
{
public:
  LinearDisplacement(
    const unsigned int                                  n_crystals,
    const unsigned int                                  n_components,
    const bool                                          flag_is_decohesion_allowed,
    const gCP::RunTimeParameters::SemicoupledParameters parameters);

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override;

private:
  const unsigned int              n_crystals;

  double                          time;

  const double                    period;

  const double                    mean_value;

  const double                    amplitude;

  const double                    n_cycles;

  const double                    start_of_loading_phase;

  const double                    start_of_cyclic_phase;

  const double                    start_of_unloading_phase;

  const double                    preloading_phase_duration;

  const double                    unloading_and_unloading_phase_duration;

  const bool                      flag_is_decohesion_allowed;
};



template <int dim>
LinearDisplacement<dim>::LinearDisplacement(
  const unsigned int                                  n_crystals,
  const unsigned int                                  n_components,
  const bool                                          flag_is_decohesion_allowed,
  const gCP::RunTimeParameters::SemicoupledParameters parameters)
:
dealii::Function<dim>(
  n_components,
  parameters.temporal_discretization_parameters.start_time),
n_crystals(n_crystals),
time(parameters.temporal_discretization_parameters.start_time),
period(parameters.temporal_discretization_parameters.period),
mean_value(0.1),
amplitude(0.025),
n_cycles(parameters.temporal_discretization_parameters.n_cycles),
start_of_loading_phase(parameters.temporal_discretization_parameters.start_of_loading_phase),
start_of_cyclic_phase(parameters.temporal_discretization_parameters.start_of_cyclic_phase),
start_of_unloading_phase(parameters.temporal_discretization_parameters.start_of_unloading_phase),
preloading_phase_duration(parameters.temporal_discretization_parameters.preloading_phase_duration),
unloading_and_unloading_phase_duration(parameters.temporal_discretization_parameters.unloading_and_unloading_phase_duration),
flag_is_decohesion_allowed(flag_is_decohesion_allowed)
{}



template<int dim>
void LinearDisplacement<dim>::vector_value(
  const dealii::Point<dim>  &point,
  dealii::Vector<double>    &return_vector) const
{
  (void)point;

  const double time = this->get_time();

  return_vector = 0.0;

  /*if (point[1] < 2.59808)
  {
    return;
  }*/

  double displacement;

  displacement = 0.;

  if (time < start_of_loading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI * time / (preloading_phase_duration * 2));

    displacement = factor;
  }
  else if (time < start_of_cyclic_phase)
  {
    const double factor = time / unloading_and_unloading_phase_duration;

    displacement = factor * mean_value;
  }
  else if (time < start_of_unloading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI / period * (time - start_of_cyclic_phase));

    displacement = mean_value + factor * amplitude;
  }
  else
  {
    const double factor =
      1.0 - (time - start_of_unloading_phase) /
        unloading_and_unloading_phase_duration ;

    displacement = factor * mean_value;
  }

  // Return
  return_vector[0] = displacement;

  if (flag_is_decohesion_allowed)
  {
    for (unsigned int i = 1; i < n_crystals; ++i)
    {
      return_vector[i*dim]      = displacement;
    }
  }
}



template<int dim>
class SemicoupledProblem
{
public:
  SemicoupledProblem(
    const RunTimeParameters::SemicoupledParameters &parameters);

  void run();

private:
  const RunTimeParameters::SemicoupledParameters    parameters;

  std::shared_ptr<dealii::ConditionalOStream>       pcout;

  std::shared_ptr<dealii::TimerOutput>              timer_output;

  std::shared_ptr<dealii::Mapping<dim>>             mapping;

  dealii::DiscreteTime                              discrete_time;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  std::shared_ptr<FEField<dim>>                     fe_field;

  std::shared_ptr<CrystalsData<dim>>                crystals_data;

  GradientCrystalPlasticitySolver<dim>              gCP_solver;

  std::unique_ptr<LinearDisplacement<dim>>          linear_displacement;

  std::shared_ptr<TractionVector<dim>>              traction_vector;

  Postprocessing::Postprocessor<dim>                postprocessor;

  const double                                      string_width;

  const unsigned int                                x_lower_boundary_id = 0;

  const unsigned int                                x_upper_boundary_id = 1;

  const unsigned int                                y_lower_boundary_id = 2;

  const unsigned int                                y_upper_boundary_id = 3;

  void make_grid();

  void update_dirichlet_boundary_conditions();

  void setup();

  void setup_constraints();

  void initialize_calls();

  void solve();

  void postprocessing();

  void checkpoint();

  void triangulation_output();

  void data_output();
};



template<int dim>
SemicoupledProblem<dim>::SemicoupledProblem(
  const RunTimeParameters::SemicoupledParameters &parameters)
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
  parameters.spatial_discretization.fe_degree_displacements,
  parameters.spatial_discretization.fe_degree_slips,
  parameters.solver_parameters.allow_decohesion)),
crystals_data(std::make_shared<CrystalsData<dim>>()),
gCP_solver(parameters.solver_parameters,
           parameters.temporal_discretization_parameters,
           discrete_time,
           fe_field,
           crystals_data,
           mapping,
           pcout,
           timer_output),
traction_vector(
  std::make_shared<TractionVector<dim>>(parameters)),
postprocessor(fe_field,
              crystals_data,
              parameters.output.flag_output_fluctuations),
string_width(
  (std::to_string((unsigned int)(
  (parameters.temporal_discretization_parameters.end_time -
   parameters.temporal_discretization_parameters.start_time) /
   parameters.temporal_discretization_parameters.time_step_size)) +
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
        if (entry.path().extension() == ".vtu" ||
            entry.path().extension() == ".pvtu")
          fs::remove(entry.path());

      *pcout
        << "done!\n\n";
    }

    if (fs::exists(parameters.output.output_directory + "checkpoints/"))
    {
      *pcout
        << "Deleting checkpoints files inside the checkpoints folder... "
        << std::flush;

      for (const auto& entry : fs::directory_iterator(parameters.output.output_directory + "checkpoints/"))
        fs::remove(entry.path());

      *pcout
        << "done!\n\n";
    }
  }
}



template<int dim>
void SemicoupledProblem<dim>::make_grid()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Triangulation");

  // Read mesh from file
  dealii::GridIn<dim> grid_in;

  grid_in.attach_triangulation(triangulation);

  std::ifstream input_file(parameters.msh_file_pathname);

  grid_in.read_msh(input_file);

  // Identify boundaries
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          if (face->center()[0] == 0.0)
            face->set_boundary_id(x_lower_boundary_id);
          else if (face->center()[0] == 4.5)
            face->set_boundary_id(x_upper_boundary_id);
          else if (face->center()[1] == 0.0)
            face->set_boundary_id(y_lower_boundary_id);
          else if (face->center()[1] > 3.0)
            face->set_boundary_id(y_upper_boundary_id);
        }

  this->triangulation.refine_global(
    parameters.spatial_discretization.n_global_refinements);

  // Terminal output
  *pcout << "Triangulation:"
         << std::endl
         << " Number of active cells = "
         << triangulation.n_global_active_cells()
         << std::endl << std::endl;
}


template<int dim>
void SemicoupledProblem<dim>::setup()
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

  // Set material ids
  for (const auto &active_cell : triangulation.active_cell_iterators())
  {
    if (active_cell->is_locally_owned())
    {
      active_cell->set_material_id(active_cell->material_id() - 1);
    }
  }

  // Update the material ids of ghost cells
  fe_field->update_ghost_material_ids();

  // Set the active finite elemente index of each cell
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(cell->material_id());

  // Sets up the degrees of freedom
  fe_field->setup_dofs();

  linear_displacement =
    std::make_unique<LinearDisplacement<dim>>(
      crystals_data->get_n_crystals(),
      fe_field->get_n_components(),
      fe_field->is_decohesion_allowed(),
      parameters);

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
void SemicoupledProblem<dim>::setup_constraints()
{
  // Initiate a ZeroFunction object
  dealii::Functions::ZeroFunction<dim> zero_function(
                                        fe_field->get_n_components());

  // Initiate a dealii::Function map
  std::map<dealii::types::boundary_id,
        const dealii::Function<dim> *> function_map;

  // Initiate the actual constraints – boundary conditions – of the problem
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    // Displacements' Dirichlet boundary conditions
    {
      function_map.clear();

      function_map[x_lower_boundary_id] = &zero_function;
      //function_map[x_upper_boundary_id] = linear_displacement.get();

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

      function_map.clear();

      function_map[y_lower_boundary_id] = &zero_function;
      function_map[y_upper_boundary_id] = &zero_function;

      for (unsigned int crystal_id = 0;
            crystal_id < crystals_data->get_n_crystals();
            ++crystal_id)
      {
        dealii::ComponentMask component_mask(
          fe_field->get_n_components(),
          false);

        if (fe_field->is_decohesion_allowed())
        {
          component_mask.set(crystal_id*dim + 1, true);
        }
        else
        {
          component_mask.set(1, true);
        }

        dealii::VectorTools::interpolate_boundary_values(
          *mapping,
          fe_field->get_dof_handler(),
          function_map,
          affine_constraints,
          component_mask);
      }
    }

    // Slips' Dirichlet boundary conditions
    {

    }

    // Set microhard boundary conditions if specified.
    if (parameters.solver_parameters.boundary_conditions_at_grain_boundaries ==
          RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microhard)
    {
      std::vector<dealii::types::global_dof_index> local_face_dof_indices(
        fe_field->get_fe_collection().max_dofs_per_face());

      // Loop over locally owned faces at the grain boundaries
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

              // Get the global degrees of freedom of the face element
              cell->face(face_index)->get_dof_indices(
                local_face_dof_indices,
                crystal_id);

              // Loop over the global degrees of freedom
              for (unsigned int i = 0;
                   i < local_face_dof_indices.size(); ++i)
              {
                // If the global degree of freedom corresponds to a
                // slips, constrain it homogeneously
                if (fe_field->get_global_component(crystal_id, i) >= dim)
                {
                  affine_constraints.add_line(local_face_dof_indices[i]);
                }
              }
            }
          }
        }
      }
    }
  }
  affine_constraints.close();

  // Inhomogeneous constraints are zero-ed out for the Newton Rhapson method
  dealii::AffineConstraints<double> newton_method_constraints;

  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(fe_field->get_locally_relevant_dofs());
    newton_method_constraints.merge(fe_field->get_hanging_node_constraints());

    // Displacements' Dirichlet boundary conditions
    {
      function_map.clear();

      function_map[x_lower_boundary_id] = &zero_function;
      //function_map[x_upper_boundary_id] = &zero_function;

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

      function_map.clear();

      function_map[y_lower_boundary_id] = &zero_function;
      function_map[y_upper_boundary_id] = &zero_function;

      for (unsigned int crystal_id = 0;
            crystal_id < crystals_data->get_n_crystals();
            ++crystal_id)
      {
        dealii::ComponentMask component_mask(
          fe_field->get_n_components(),
          false);

        if (fe_field->is_decohesion_allowed())
        {
          component_mask.set(crystal_id*dim + 1, true);
        }
        else
        {
          component_mask.set(1, true);
        }

        dealii::VectorTools::interpolate_boundary_values(
          *mapping,
          fe_field->get_dof_handler(),
          function_map,
          newton_method_constraints,
          component_mask);
      }
    }

    // Slips' Dirichlet boundary conditions
    {

    }

    // Set microhard boundary conditions if specified.
    if (parameters.solver_parameters.boundary_conditions_at_grain_boundaries ==
          RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microhard)
    {
      std::vector<dealii::types::global_dof_index> local_face_dof_indices(
        fe_field->get_fe_collection().max_dofs_per_face());

      // Loop over locally owned faces at the grain boundaries
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

              // Get the global degrees of freedom of the face element
              cell->face(face_index)->get_dof_indices(
                local_face_dof_indices,
                crystal_id);

              // Loop over the global degrees of freedom
              for (unsigned int i = 0;
                   i < local_face_dof_indices.size(); ++i)
              {
                // If the global degree of freedom corresponds to a
                // slips, constrain it homogeneously
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
  }
  newton_method_constraints.close();

  // The constraints are now passed to the FEField instance
  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(newton_method_constraints);

  // Neumann boundary conditions
  gCP_solver.set_neumann_boundary_condition(
    x_upper_boundary_id, traction_vector);
}

template<int dim>
void SemicoupledProblem<dim>::initialize_calls()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Initialize calls");

  // Initiate the solver
  gCP_solver.init();

  const fs::path output_directory{parameters.output.output_directory};

  fs::path path_to_ouput_file =
    output_directory / "homogenization.txt";

  std::ofstream ofstream(path_to_ouput_file.string());

  postprocessor.init(gCP_solver.get_hooke_law());
}


template<int dim>
void SemicoupledProblem<dim>::update_dirichlet_boundary_conditions()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Update boundary conditions");

  // Instantiate the temporary AffineConstraint instance
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    std::map<dealii::types::boundary_id,
            const dealii::Function<dim> *> function_map;

    function_map[x_upper_boundary_id] = linear_displacement.get();

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
  affine_constraints.close();

  fe_field->set_affine_constraints(affine_constraints);
}



template<int dim>
void SemicoupledProblem<dim>::postprocessing()
{

}



template<int dim>
void SemicoupledProblem<dim>::checkpoint()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Checkpoint");

}



template<int dim>
void SemicoupledProblem<dim>::triangulation_output()
{
  dealii::Vector<float> locally_owned_subdomain(triangulation.n_active_cells());

  dealii::Vector<float> material_id(triangulation.n_active_cells());

  dealii::Vector<float> active_fe_index(triangulation.n_active_cells());

  dealii::Vector<float> boundary_id(triangulation.n_active_cells());

  locally_owned_subdomain = -1.0;
  material_id             = -1.0;
  active_fe_index         = -1.0;
  boundary_id             = -1.0;

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


      if (cell->at_boundary())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
      {
        boundary_id(cell->active_cell_index()) =
          face->boundary_id();
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

  data_out.add_data_vector(boundary_id,
                           "boundary_id");

  data_out.add_data_vector(gCP_solver.get_cell_is_at_grain_boundary_vector(),
                           "cell_is_at_grain_boundary");

  data_out.build_patches(2);

  data_out.write_vtu_in_parallel(
    parameters.output.output_directory + "triangulation.vtu",
    MPI_COMM_WORLD);
}



template<int dim>
void SemicoupledProblem<dim>::data_output()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Data output");

  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(fe_field->get_dof_handler());

  data_out.add_data_vector(fe_field->solution, postprocessor);

  data_out.build_patches(*mapping,
                         fe_field->get_displacement_fe_degree(),
                         dealii::DataOut<dim>::curved_inner_cells);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    parameters.output.output_directory + "paraview/",
    "solution",
    out_index,
    MPI_COMM_WORLD,
    5);

  if (parameters.output.flag_output_damage_variable ||
      discrete_time.get_current_time() ==
          discrete_time.get_end_time())
  {
    dealii::DataOutFaces<dim> data_out_face(false);

    std::vector<std::string>  face_name(1, "damage");

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
void SemicoupledProblem<dim>::run()
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

  //return;

  discrete_time.set_desired_next_step_size(
    parameters.temporal_discretization_parameters.time_step_size_in_preloading_phase);

  // Time loop. The current time at the beggining of each loop
  // corresponds to t^{n-1}
  while(discrete_time.get_current_time() < discrete_time.get_end_time())
  {
    if (std::abs(discrete_time.get_current_time() -
        parameters.temporal_discretization_parameters.start_of_loading_phase)
        < std::numeric_limits<double>::epsilon() * 1000)
    {
      discrete_time.set_desired_next_step_size(
        parameters.temporal_discretization_parameters.time_step_size_in_loading_and_unloading_phase);
    }

    if (std::abs(discrete_time.get_current_time() -
        parameters.temporal_discretization_parameters.start_of_cyclic_phase)
        < std::numeric_limits<double>::epsilon() * 1000)
    {
      discrete_time.set_desired_next_step_size(
        parameters.temporal_discretization_parameters.time_step_size_in_cyclic_phase);
    }

    if (std::abs(discrete_time.get_current_time() -
        parameters.temporal_discretization_parameters.start_of_unloading_phase)
        < std::numeric_limits<double>::epsilon() * 1000)
    {
      discrete_time.set_desired_next_step_size(
        parameters.temporal_discretization_parameters.time_step_size_in_loading_and_unloading_phase);
    }

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
    linear_displacement->set_time(discrete_time.get_next_time());

    // Update the Dirichlet boundary conditions values to t^{n}
    //update_dirichlet_boundary_conditions();


    // Solve the nonlinear system. After the call fe_field->solution
    // corresponds to the solution at t^n
    std::tuple<bool, unsigned int> results =
      gCP_solver.solve_nonlinear_system();

    (void)results;


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
      parameters_filepath = "input/parameter_files/GACM2023.prm";
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

    gCP::RunTimeParameters::SemicoupledParameters parameters(parameters_filepath);

    gCP::SemicoupledProblem<2> problem(parameters);

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