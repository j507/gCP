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

  const double                    componenet_11_mean_value;

  const double                    componenet_22_mean_value;

  const double                    componenet_33_mean_value;

  const double                    componenet_23_mean_value;

  const double                    componenet_23_amplitude;

  const double                    n_cycles;

  const double                    start_of_loading_phase;

  const double                    start_of_cyclic_phase;

  const double                    start_of_unloading_phase;

  const double                    preloading_phase_duration;

  const double                    unloading_and_unloading_phase_duration;

  dealii::SymmetricTensor<2,dim>  preloading_macroscopic_strain;

  dealii::SymmetricTensor<2,dim>  cyclic_macroscopic_strain_mean_value;

  dealii::SymmetricTensor<2,dim>  cyclic_macroscopic_strain_amplitude;

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
componenet_11_mean_value(-.02),
componenet_22_mean_value(-.02),
componenet_33_mean_value(.04),
componenet_23_mean_value(.002),
componenet_23_amplitude(.0002),
n_cycles(parameters.temporal_discretization_parameters.n_cycles),
start_of_loading_phase(parameters.temporal_discretization_parameters.start_of_loading_phase),
start_of_cyclic_phase(parameters.temporal_discretization_parameters.start_of_cyclic_phase),
start_of_unloading_phase(parameters.temporal_discretization_parameters.start_of_unloading_phase),
preloading_phase_duration(parameters.temporal_discretization_parameters.preloading_phase_duration),
unloading_and_unloading_phase_duration(parameters.temporal_discretization_parameters.unloading_and_unloading_phase_duration),
flag_is_decohesion_allowed(flag_is_decohesion_allowed)
{
  preloading_macroscopic_strain = 0.;

  cyclic_macroscopic_strain_mean_value  = 0.;

  cyclic_macroscopic_strain_amplitude   = 0.;

  preloading_macroscopic_strain[0][0] = componenet_11_mean_value;

  preloading_macroscopic_strain[1][1] = componenet_22_mean_value;

  preloading_macroscopic_strain[2][2] = componenet_33_mean_value;

  cyclic_macroscopic_strain_mean_value[1][2]  = componenet_23_mean_value;

  cyclic_macroscopic_strain_amplitude[1][2]   = componenet_23_amplitude;
}



template<int dim>
void LinearDisplacement<dim>::vector_value(
  const dealii::Point<dim>  &point,
  dealii::Vector<double>    &return_vector) const
{
  const double time = this->get_time();

  return_vector = 0.0;

  dealii::Tensor<1,dim> displacement;

  dealii::SymmetricTensor<2,dim> tensor;

  displacement = 0.;

  if (time < start_of_loading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI * time / (preloading_phase_duration * 2));

    tensor = factor * preloading_macroscopic_strain;

    displacement = factor * preloading_macroscopic_strain * point;
  }
  else if (time < start_of_cyclic_phase)
  {
      const double factor =
        (time - start_of_loading_phase) /
        unloading_and_unloading_phase_duration;

    tensor = factor * cyclic_macroscopic_strain_mean_value;

    displacement = factor * cyclic_macroscopic_strain_mean_value * point;
  }
  else if (time < start_of_unloading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI / period * (time - start_of_cyclic_phase));

    tensor = cyclic_macroscopic_strain_mean_value +
                    factor * cyclic_macroscopic_strain_amplitude;

    displacement = (cyclic_macroscopic_strain_mean_value +
                    factor * cyclic_macroscopic_strain_amplitude) *
                   point;
  }
  else
  {
    const double factor =
      1.0 - (time - start_of_unloading_phase) /
        unloading_and_unloading_phase_duration ;

    tensor = factor * cyclic_macroscopic_strain_mean_value;

    displacement = factor * cyclic_macroscopic_strain_mean_value * point;
  }

  //std::cout << tensor << std::endl;

  // Return
  return_vector[0] = displacement[0];
  return_vector[1] = displacement[1];
  return_vector[2] = displacement[2];

  if (flag_is_decohesion_allowed)
  {
    for (unsigned int i = 1; i < n_crystals; ++i)
    {
      return_vector[i*dim]      = displacement[0];
      return_vector[i*dim + 1]  = displacement[1];
      return_vector[i*dim + 2]  = displacement[2];
    }
  }
}



template <int dim>
class MacroscopicStrain
{
public:
  MacroscopicStrain(
    const gCP::RunTimeParameters::SemicoupledParameters parameters);

  dealii::SymmetricTensor<2,dim> get_value() const;

  void set_time(const double time);

  double get_time() const;

  void advance_time(const double time_step);

private:

  double                          time;

  const double                    period;

  double                          componenet_11_mean_value;

  double                          componenet_22_mean_value;

  double                          componenet_33_mean_value;

  double                          componenet_23_mean_value;

  double                          componenet_11_amplitude;

  double                          componenet_22_amplitude;

  double                          componenet_33_amplitude;

  double                          componenet_23_amplitude;

  const double                    reduction_factor;

  const double                    n_cycles;

  const double                    start_of_loading_phase;

  const double                    start_of_cyclic_phase;

  const double                    start_of_unloading_phase;

  const double                    preloading_phase_duration;

  const double                    unloading_and_unloading_phase_duration;
};



template <int dim>
MacroscopicStrain<dim>::MacroscopicStrain(
    const gCP::RunTimeParameters::SemicoupledParameters parameters)
:
time(parameters.temporal_discretization_parameters.start_time),
period(parameters.temporal_discretization_parameters.period),
reduction_factor(0.10),
n_cycles(parameters.temporal_discretization_parameters.n_cycles),
start_of_loading_phase(parameters.temporal_discretization_parameters.start_of_loading_phase),
start_of_cyclic_phase(parameters.temporal_discretization_parameters.start_of_cyclic_phase),
start_of_unloading_phase(parameters.temporal_discretization_parameters.start_of_unloading_phase),
preloading_phase_duration(parameters.temporal_discretization_parameters.preloading_phase_duration),
unloading_and_unloading_phase_duration(parameters.temporal_discretization_parameters.unloading_and_unloading_phase_duration)
{
  componenet_11_mean_value  = -0.02;
  componenet_22_mean_value  = -0.02;
  componenet_33_mean_value  = 0.04;
  componenet_23_mean_value  = 0.002;
  componenet_23_amplitude   = 0.0002;
}



template <int dim>
dealii::SymmetricTensor<2,dim> MacroscopicStrain<dim>::get_value() const
{
  dealii::SymmetricTensor<2,dim> macroscopic_strain;

  macroscopic_strain = 0.0;

  if (time < start_of_loading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI * time / (preloading_phase_duration * 2));

    macroscopic_strain[0][0] = componenet_11_mean_value * factor;

    macroscopic_strain[1][1] = componenet_22_mean_value * factor;

    macroscopic_strain[2][2] = componenet_33_mean_value * factor;
  }
  else if (time < start_of_cyclic_phase)
  {
      const double factor =
        (time - start_of_loading_phase) /
        unloading_and_unloading_phase_duration;

      macroscopic_strain[1][2] = componenet_23_mean_value * factor;
  }
  else if (time < start_of_unloading_phase)
  {
    const double factor =
      std::sin(2.0 * M_PI / period * (time - start_of_cyclic_phase));

    macroscopic_strain[1][2] =
      componenet_23_mean_value -
        componenet_23_amplitude * factor;
  }
  else
  {
    const double factor =
      1.0 - (time - start_of_unloading_phase) /
        unloading_and_unloading_phase_duration ;

    macroscopic_strain[1][2] =
      componenet_23_mean_value * factor;
  }

  return (macroscopic_strain);
}



template <int dim>
void MacroscopicStrain<dim>::set_time(
  const double time)
{
  this->time = time;
}



template <int dim>
double MacroscopicStrain<dim>::get_time() const
{
  return (time);
}



template <int dim>
void MacroscopicStrain<dim>::advance_time(
  const double time_step)
{
  this->time += time_step;
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

  gCP::MacroscopicStrain<dim>                       macroscopic_strain;

  std::unique_ptr<LinearDisplacement<dim>>          linear_displacement;

  Postprocessing::Homogenization<dim>               homogenization;

  Postprocessing::Postprocessor<dim>                postprocessor;

  Postprocessing::ResidualPostprocessor<dim>        residual_postprocessor;

  const double                                      string_width;

  const unsigned int                                x_lower_boundary_id = 0;

  const unsigned int                                x_upper_boundary_id = 1;

  const unsigned int                                y_lower_boundary_id = 2;

  const unsigned int                                y_upper_boundary_id = 3;

  const unsigned int                                z_lower_boundary_id = 4;

  const unsigned int                                z_upper_boundary_id = 5;

  double                                            extrude_distance;

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
           parameters.temporal_discretization_parameters,
           discrete_time,
           fe_field,
           crystals_data,
           mapping,
           pcout,
           timer_output),
macroscopic_strain(parameters),
homogenization(fe_field,
               mapping),
postprocessor(fe_field,
              crystals_data,
              parameters.flag_output_fluctuations),
residual_postprocessor(
  fe_field,
  crystals_data),
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

    if (fs::exists(parameters.graphical_output_directory + "checkpoints/"))
    {
      *pcout
        << "Deleting checkpoints files inside the checkpoints folder... "
        << std::flush;

      for (const auto& entry : fs::directory_iterator(parameters.graphical_output_directory + "checkpoints/"))
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

  dealii::parallel::distributed::Triangulation<dim-1>
    tmp_triangulation(
    MPI_COMM_WORLD,
    typename dealii::Triangulation<dim-1>::MeshSmoothing(
    dealii::Triangulation<dim-1>::smoothing_on_refinement |
    dealii::Triangulation<dim-1>::smoothing_on_coarsening));

  // Read mesh from file
  dealii::GridIn<dim-1> grid_in;

  grid_in.attach_triangulation(tmp_triangulation);

  std::ifstream input_file(parameters.msh_file_pathname);

  grid_in.read_msh(input_file);

  extrude_distance =
    1.0 / sqrt(tmp_triangulation.n_global_active_cells());

  dealii::GridGenerator::extrude_triangulation(
    tmp_triangulation,
    2,
    extrude_distance,
    triangulation);

  // Identify boundaries
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          if (face->center()[0] == 0.)
            face->set_boundary_id(x_lower_boundary_id);
          else if (face->center()[0] == 1.)
            face->set_boundary_id(x_upper_boundary_id);
          else if (face->center()[1] == 0.)
            face->set_boundary_id(y_lower_boundary_id);
          else if (face->center()[1] == 1.)
            face->set_boundary_id(y_upper_boundary_id);

          if constexpr(dim == 3)
          {
            if (face->center()[2] == 0.)
              face->set_boundary_id(z_lower_boundary_id);
            else if (face->center()[2] == extrude_distance)
              face->set_boundary_id(z_upper_boundary_id);
          }
        }

  this->triangulation.refine_global(parameters.n_global_refinements);

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
                      parameters.euler_angles_pathname,
                      parameters.slips_directions_pathname,
                      parameters.slips_normals_pathname);

  // Sets up the FEValuesExtractor instances
  fe_field->setup_extractors(crystals_data->get_n_crystals(),
                             crystals_data->get_n_slips());

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
  // Initiate the actual constraints – boundary conditions – of the problem
  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(fe_field->get_locally_relevant_dofs());
    affine_constraints.merge(fe_field->get_hanging_node_constraints());

    if (parameters.solver_parameters.boundary_conditions_at_grain_boundaries ==
          RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microhard)
    {
      std::vector<dealii::types::global_dof_index> local_face_dof_indices(
        fe_field->get_fe_collection().max_dofs_per_face());

      for (const auto &cell :
           fe_field->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &face_index : cell->face_indices())
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
                if (fe_field->get_global_component(crystal_id, i) >= dim)
                  affine_constraints.add_line(local_face_dof_indices[i]);
            }
    }

    std::map<dealii::types::boundary_id,
            const dealii::Function<dim> *> function_map;

    function_map[x_lower_boundary_id] = linear_displacement.get();
    function_map[x_upper_boundary_id] = linear_displacement.get();
    function_map[y_lower_boundary_id] = linear_displacement.get();
    function_map[y_upper_boundary_id] = linear_displacement.get();
    function_map[z_lower_boundary_id] = linear_displacement.get();
    function_map[z_upper_boundary_id] = linear_displacement.get();

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

  // Inhomogeneous constraints are zero-ed out for the Newton Rhapson method
  dealii::AffineConstraints<double> newton_method_constraints;

  newton_method_constraints.clear();
  {
    newton_method_constraints.reinit(fe_field->get_locally_relevant_dofs());
    newton_method_constraints.merge(fe_field->get_hanging_node_constraints());

    if (parameters.solver_parameters.boundary_conditions_at_grain_boundaries ==
          RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microhard)
    {
      std::vector<dealii::types::global_dof_index> local_face_dof_indices(
        fe_field->get_fe_collection().max_dofs_per_face());

      for (const auto &cell :
           fe_field->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &face_index : cell->face_indices())
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
                if (fe_field->get_global_component(crystal_id, i) >= dim)
                  newton_method_constraints.add_line(local_face_dof_indices[i]);
            }
    }

    dealii::Functions::ZeroFunction<dim> zero_function(
                                          fe_field->get_n_components());

    std::map<dealii::types::boundary_id,
             const dealii::Function<dim> *> function_map;

    function_map[x_lower_boundary_id] = &zero_function;
    function_map[x_upper_boundary_id] = &zero_function;
    function_map[y_lower_boundary_id] = &zero_function;
    function_map[y_upper_boundary_id] = &zero_function;
    function_map[z_lower_boundary_id] = &zero_function;
    function_map[z_upper_boundary_id] = &zero_function;

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
  }
  newton_method_constraints.close();

  // The constraints are now passed to the FEField instance
  fe_field->set_affine_constraints(affine_constraints);
  fe_field->set_newton_method_constraints(newton_method_constraints);
}

template<int dim>
void SemicoupledProblem<dim>::initialize_calls()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Initialize calls");

  // Initiate the solver
  gCP_solver.init();

  const fs::path output_directory{parameters.graphical_output_directory};

  fs::path path_to_ouput_file =
    output_directory / "homogenization.txt";

  std::ofstream ofstream(path_to_ouput_file.string());

  // Initiate the benchmark data
  homogenization.init(gCP_solver.get_elastic_strain_law(),
                      gCP_solver.get_hooke_law(),
                      ofstream);

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

    dealii::Functions::ZeroFunction<dim> zero_function(
                                          fe_field->get_n_components());


    std::map<dealii::types::boundary_id,
            const dealii::Function<dim> *> function_map;

    function_map[x_lower_boundary_id] = linear_displacement.get();
    function_map[x_upper_boundary_id] = linear_displacement.get();
    function_map[y_lower_boundary_id] = linear_displacement.get();
    function_map[y_upper_boundary_id] = linear_displacement.get();
    function_map[z_lower_boundary_id] = linear_displacement.get();
    function_map[z_upper_boundary_id] = linear_displacement.get();

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
  /*
  const RunTimeParameters::TemporalDiscretizationParameters &prm =
    parameters.temporal_discretization_parameters;

  bool flag_store_checkpoint = false;

  if (prm.loading_type == RunTimeParameters::LoadingType::Cyclic ||
      prm.loading_type == RunTimeParameters::LoadingType::CyclicWithUnloading)
  {
    const bool last_step_of_loading_phase =
        discrete_time.get_step_number() == prm.n_steps_in_loading_phase;

    const unsigned int n_steps_until_unloading =
      prm.n_steps_in_loading_phase +
      prm.n_steps_per_half_cycle * 2.0 * prm.n_cycles;

    const bool last_step_of_cyclic_phase =
        discrete_time.get_step_number() == n_steps_until_unloading;

    const bool extrema_step_of_cyclic_phase =
        discrete_time.get_step_number() > prm.n_steps_in_loading_phase
          &&
        discrete_time.get_step_number() < n_steps_until_unloading
          &&
        (discrete_time.get_step_number() - prm.n_steps_in_loading_phase) %
        (2 * prm.n_steps_per_half_cycle) == 0;

      if ((last_step_of_loading_phase || extrema_step_of_cyclic_phase ||
           last_step_of_cyclic_phase) &&
          parameters.flag_store_checkpoint)
      {
        flag_store_checkpoint = true;
      }
  }

  if (flag_store_checkpoint)
  {
    dealii::TimerOutput::Scope  t(*timer_output, "Problem: Checkpoint storage");

    std::string checkpoint_file_name =
      parameters.graphical_output_directory +
      "checkpoints/SolutionTransfer_" +
      std::to_string(discrete_time.get_step_number());

    dealii::parallel::distributed::SolutionTransfer<
      dim, dealii::LinearAlgebraTrilinos::MPI::Vector>
        solution_transfer(fe_field->get_dof_handler());

    fe_field->prepare_for_serialization_of_active_fe_indices();

    std::vector<const dealii::LinearAlgebraTrilinos::MPI::Vector*>
      solution_vectors(3);

    solution_vectors[0] = &fe_field->solution;
    solution_vectors[1] = &fe_field->old_solution;
    solution_vectors[2] = &fe_field->old_old_solution;

    solution_transfer.prepare_for_serialization(solution_vectors);

    triangulation.save(checkpoint_file_name);

    std::ofstream output_file_stream(checkpoint_file_name + ".txt");

    dealii::SymmetricTensor<2,dim> macroscopic_strain_value =
      macroscopic_strain.get_value();

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (unsigned int i = 0;
          i < macroscopic_strain_value.n_independent_components; ++i)
      {
        output_file_stream << macroscopic_strain_value.access_raw_entry(i);

        if (i < (macroscopic_strain_value.n_independent_components - 1))
        {
          output_file_stream << ",";
        }
      }
    }
  }
  */
  if (parameters.flag_compute_macroscopic_quantities &&
      (discrete_time.get_step_number() %
         parameters.homogenization_frequency == 0 ||
        discrete_time.get_current_time() ==
          discrete_time.get_end_time()))
  {
    dealii::TimerOutput::Scope  t(*timer_output, "Problem: Homogenization");

    //homogenization.set_macroscopic_strain(macroscopic_strain.get_value());

    homogenization.compute_macroscopic_quantities(
      discrete_time.get_current_time());
  }
}



template<int dim>
void SemicoupledProblem<dim>::checkpoint()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Checkpoint");
  /*
  dealii::parallel::distributed::SolutionTransfer<dim, dealii::Vector<double>>
    solution_transfer(dg_dof_handler);

  solution_transfer.prepare_for_serialization(solution);

  triangulation.save("file"); */
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
    parameters.graphical_output_directory + "triangulation.vtu",
    MPI_COMM_WORLD);
}



template<int dim>
void SemicoupledProblem<dim>::data_output()
{
  dealii::TimerOutput::Scope  t(*timer_output, "Problem: Data output");

  dealii::DataOut<dim> data_out;

  //postprocessor.set_macroscopic_strain(macroscopic_strain.get_value());

  data_out.attach_dof_handler(fe_field->get_dof_handler());

  data_out.add_data_vector(fe_field->solution, postprocessor);

  if (parameters.flag_output_residual)
  {
    data_out.add_data_vector(gCP_solver.get_residual(),
                             residual_postprocessor);
  }

  data_out.build_patches(*mapping,
                         fe_field->get_displacement_fe_degree(),
                         dealii::DataOut<dim>::curved_inner_cells);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    parameters.graphical_output_directory + "paraview/",
    "solution",
    out_index,
    MPI_COMM_WORLD,
    5);

  if (parameters.flag_output_damage_variable ||
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
      parameters.graphical_output_directory + "paraview/",
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


  const RunTimeParameters::ConvergenceControlParameters
    &convergence_control_parameters =
      parameters.solver_parameters.convergence_control_parameters;

  (void)convergence_control_parameters;

  std::ofstream macroscopic_damage_file;

  { // Clear file's contents
    macroscopic_damage_file.open(
      parameters.graphical_output_directory + "macroscopic_damage.txt",
      std::ofstream::out | std::ofstream::trunc);

    macroscopic_damage_file.close();
  }

  macroscopic_damage_file.open(
    parameters.graphical_output_directory + "macroscopic_damage.txt",
    std::ios::out | std::ios::app);

  if (macroscopic_damage_file.fail())
  {
    throw std::ios_base::failure(std::strerror(errno));
  }

  //make sure write fails with exception if something is wrong
  macroscopic_damage_file.exceptions(
    macroscopic_damage_file.exceptions() |
    std::ios::failbit |
    std::ifstream::badbit);

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
    //macroscopic_strain.set_time(discrete_time.get_next_time());

    linear_displacement->set_time(discrete_time.get_next_time());

    // Update the Dirichlet boundary conditions values to t^{n}
    update_dirichlet_boundary_conditions();

    //gCP_solver.set_macroscopic_strain(macroscopic_strain.get_value());

    //std::cout << macroscopic_strain.get_value() << std::endl;

    // Solve the nonlinear system. After the call fe_field->solution
    // corresponds to the solution at t^n
    std::tuple<bool, unsigned int> results =
      gCP_solver.solve_nonlinear_system();

    (void)results;
    /*
    if (std::get<0>(results) == false)
    {
      const double desired_next_step_size =
        discrete_time.get_next_step_size() /
        convergence_control_parameters.downscaling_factor;

      AssertThrow(
        desired_next_step_size >
          convergence_control_parameters.lower_threshold,
        dealii::ExcMessage("Way to small"));

      discrete_time.set_desired_next_step_size(desired_next_step_size);

      continue;
    }
    */

    // Update the solution vectors, i.e.,
    // fe_field->old_solution = fe_field->solution
    fe_field->update_solution_vectors();

    // Advance the DiscreteTime instance to t^{n}
    discrete_time.advance_time();

    /*
    if (std::get<0>(results) == true &&
        std::get<1>(results) < convergence_control_parameters.n_max_iterations)
    {
      double desired_next_step_size =
        discrete_time.get_next_step_size() *
        convergence_control_parameters.upscaling_factor;

      if (desired_next_step_size >
          convergence_control_parameters.upper_threshold)
      {
        desired_next_step_size =
          convergence_control_parameters.upper_threshold;
      }

      discrete_time.set_desired_next_step_size(desired_next_step_size);
    }
    */

    // Call to the postprocessing method
    postprocessing();

    // Call to the data output method
    if (discrete_time.get_step_number() %
         parameters.graphical_output_frequency == 0 ||
        discrete_time.get_current_time() ==
          discrete_time.get_end_time())
      data_output();

    // Print macroscopic damage to file
    if (parameters.flag_output_damage_variable ||
        discrete_time.get_current_time() ==
          discrete_time.get_end_time())
    {
      macroscopic_damage_file
        << discrete_time.get_current_time()
        << ", "
        << gCP_solver.get_macroscopic_damage()
        << std::endl;
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

    // The switch statement verifies that the filepath includes the
    // .prm extension. That if, a filepath was even passed to the
    // executable. The existance of the filepath is checked by the
    // gCP::RunTimeParameters::ProblemParameters class
    switch (argc)
    {
    case 1:
      parameters_filepath = "input/parameter_files/SPP2013.prm";
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

    gCP::SemicoupledProblem<3> problem(parameters);

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