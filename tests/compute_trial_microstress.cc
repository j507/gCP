#include <gCP/crystal_data.h>
#include <gCP/fe_field.h>
#include <gCP/utilities.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <random>

namespace Tests
{



template <int dim>
class FEFieldFunction : public dealii::Function<dim>
{
public:

  FEFieldFunction(
    const unsigned int  n_components,
    const unsigned int  n_crystals,
    const bool          flag_decohesion_allowed)
  :
  dealii::Function<dim>(n_components, 0),
  n_crystals(n_crystals),
  flag_decohesion_allowed(flag_decohesion_allowed)
  {}

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override
  {
    return_vector = 0.;

    unsigned int j = 0;

    const unsigned int n_displacement_degrees_of_freedom =
      dim * (flag_decohesion_allowed ? n_crystals : 1.0);

    for (unsigned int i = n_displacement_degrees_of_freedom;
         i < this->n_components;
         i++)
    {
      return_vector[i] =
        1.0 +
        j +
        point.norm()
        //point[0]
        ;

      j++;
    }
  }

private:

  const unsigned int  n_crystals;

  const bool          flag_decohesion_allowed;
};



template <int dim>
class TrialMicrostressFunction : public dealii::Function<dim>
{
public:

  TrialMicrostressFunction(
    const unsigned int  n_components,
    const unsigned int  n_crystals,
    const bool          flag_decohesion_allowed)
  :
  dealii::Function<dim>(n_components, 0),
  n_crystals(n_crystals),
  flag_decohesion_allowed(flag_decohesion_allowed)
  {}

  virtual void vector_value(
    const dealii::Point<dim>  &point,
    dealii::Vector<double>    &return_vector) const override
  {
    return_vector = 0.;

    unsigned int j = 0;

    for (unsigned int i = 0; i < this->n_components; i++)
    {
      return_vector[i] =
        1.0 +
        j +
        point.norm()
        //point[0]
        ;

      j++;
    }
  }

private:

  const unsigned int  n_crystals;

  const bool          flag_decohesion_allowed;
};



template <int dim>
class TrialMicrostress
{
public:

  TrialMicrostress(
    const dealii::Triangulation<dim>  &triangulation,
    const unsigned int                fe_degree)
  :
  fe_degree(fe_degree),
  dof_handler(triangulation)
  {}

  void setup_extractors(
    const unsigned n_crystals,
    const unsigned n_slips)
  {
    this->n_crystals  = n_crystals;
    this->n_slips     = n_slips;

    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      std::vector<dealii::FEValuesExtractors::Scalar>
        extractors_per_crystal;

      for (unsigned int j = 0; j < n_slips; ++j)
      {
        extractors_per_crystal.push_back(
          dealii::FEValuesExtractors::Scalar(i * n_slips + j));
      }

      extractors.push_back(extractors_per_crystal);
    }
  }

  void update_ghost_material_ids()
  {
    gCP::Utilities::update_ghost_material_ids(dof_handler);
  }


  void setup_dofs()
  {
    // FECollection
    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      std::vector<const dealii::FiniteElement<dim>*>  finite_elements;

      for (dealii::types::material_id j = 0; j < n_crystals; ++j)
      {
        for (unsigned int k = 0; k < n_slips; ++k)
        {
          if (i == j)
          {
            finite_elements.push_back(new dealii::FE_Q<dim>(fe_degree));
          }
          else
          {
            finite_elements.push_back(new dealii::FE_Nothing<dim>());
          }
        }
      }

      fe_collection.push_back(
        dealii::FESystem<dim>(
          finite_elements,
          std::vector<unsigned int>(finite_elements.size(), 1)));

      for (auto finite_element: finite_elements)
      {
        delete finite_element;
      }

      finite_elements.clear();
    }

    // Distribute and renumber
    dof_handler.distribute_dofs(fe_collection);

    dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

    // Extract sets relevant to parallel structure
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    dealii::DoFTools::extract_locally_relevant_dofs(
      dof_handler,
      locally_relevant_dofs);

    // Hanging node and empty affine constraints
    hanging_node_constraints.clear();
    {
      hanging_node_constraints.reinit(locally_relevant_dofs);

      dealii::DoFTools::make_hanging_node_constraints(
        dof_handler,
        hanging_node_constraints);
    }
    hanging_node_constraints.close();

    affine_constraints.clear();
    {
      affine_constraints.reinit(locally_relevant_dofs);
      affine_constraints.merge(hanging_node_constraints);
    }
    affine_constraints.close();

    // Mapping from the n-th component of the m-th crystal to the
    // global component
    global_component_mapping.resize(n_crystals * n_slips);

    for (dealii::types::material_id i = 0; i < n_crystals; ++i)
    {
      for (unsigned int j = 0; j < n_slips; ++j)
      {
        global_component_mapping[i * n_slips + j] = j;
      }
    }
  }

  void setup_vectors()
  {
    solution.reinit(
      locally_relevant_dofs,
      MPI_COMM_WORLD);

    old_solution.reinit(solution);

    distributed_vector.reinit(
      locally_owned_dofs,
      locally_relevant_dofs,
      MPI_COMM_WORLD,
      true);

    solution = 0.;

    old_solution = 0.;

    distributed_vector = 0.;
  }

  double get_fe_degree() const;

  const dealii::DoFHandler<dim>& get_dof_handler() const;

  const dealii::FEValuesExtractors::Scalar&
    get_extractor(const unsigned int crystal_id,
                       const unsigned int slip_id) const;

  const dealii::hp::FECollection<dim>& get_fe_collection() const;

  unsigned int get_n_components() const;

  const dealii::AffineConstraints<double>&
    get_hanging_node_constraints() const;

  unsigned int get_global_component(
    const unsigned int crystal_id,
    const unsigned int local_component) const;

  const dealii::IndexSet& get_locally_owned_dofs() const;

  const dealii::IndexSet& get_locally_relevant_dofs() const;

  const dealii::AffineConstraints<double>&
    get_affine_constraints() const;

  void set_affine_constraints(
    const dealii::AffineConstraints<double> &affine_constraints)
  {
    this->affine_constraints.merge(
      affine_constraints,
      dealii::AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  }

  dealii::LinearAlgebraTrilinos::MPI::Vector  solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector  old_solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector  distributed_vector;

private:

  const unsigned int                fe_degree;

  unsigned int                      n_crystals;

  unsigned int                      n_slips;

  dealii::DoFHandler<dim>           dof_handler;

  dealii::hp::FECollection<dim>     fe_collection;

  dealii::AffineConstraints<double> hanging_node_constraints;

  dealii::AffineConstraints<double> affine_constraints;

  dealii::IndexSet                  locally_owned_dofs;

  dealii::IndexSet                  locally_relevant_dofs;

  std::vector<unsigned int>         global_component_mapping;

  std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>
                                    extractors;
};



template <int dim>
inline double
TrialMicrostress<dim>::get_fe_degree() const
{
  return (fe_degree);
}



template <int dim>
inline const dealii::DoFHandler<dim> &
TrialMicrostress<dim>::get_dof_handler() const
{
  return (dof_handler);
}



template <int dim>
inline const dealii::FEValuesExtractors::Scalar&
TrialMicrostress<dim>::get_extractor(const unsigned int crystal_id,
                                 const unsigned int slip_id) const
{
  return (extractors[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::hp::FECollection<dim> &
TrialMicrostress<dim>::get_fe_collection() const
{
  return (fe_collection);
}



template <int dim>
inline unsigned int
TrialMicrostress<dim>::get_n_components() const
{
  return (fe_collection.n_components());
}




template <int dim>
inline const dealii::AffineConstraints<double> &
TrialMicrostress<dim>::get_hanging_node_constraints() const
{
  return (hanging_node_constraints);
}



template <int dim>
inline const dealii::IndexSet &
TrialMicrostress<dim>::get_locally_owned_dofs() const
{
  return (locally_owned_dofs);
}



template <int dim>
inline const dealii::IndexSet &
TrialMicrostress<dim>::get_locally_relevant_dofs() const
{
  return (locally_relevant_dofs);
}



template <int dim>
inline const dealii::AffineConstraints<double> &
TrialMicrostress<dim>::get_affine_constraints() const
{
  return (affine_constraints);
}



template <int dim>
inline unsigned int
TrialMicrostress<dim>::get_global_component(
  const unsigned int crystal_id,
  const unsigned int local_component) const
{
  return (global_component_mapping[fe_collection[crystal_id].
            system_to_component_index(local_component).first]);
}



template<int dim>
class LinearAlgebra
{
public:

  LinearAlgebra(const bool flag_decohesion_is_allowed);

  void run();

private:

  void make_grid();

  void setup();

  void create_mapping();

  void compare_vectors(const double constant = 1.0);

  void block_projection();

  void check_extractors();

  void check_global_component();

  void check_projection();

  void determine_active_set();

  void eliminate_active_set();

  void print() const;

  void vtk_output() const;

  std::shared_ptr<dealii::ConditionalOStream>       pcout;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  gCP::FEField<dim>                                 fe_field;

  TrialMicrostress<dim>                             trial_microstress;

  gCP::CrystalsData<dim>                            crystals_data;

  std::map<dealii::types::global_dof_index,
           dealii::types::global_dof_index>         mapping;

  dealii::IndexSet                                  index_set;

  const dealii::hp::MappingCollection<dim>          mapping_q;
};



template<int dim>
LinearAlgebra<dim>::LinearAlgebra(const bool flag_decohesion_is_allowed)
:
pcout(std::make_shared<dealii::ConditionalOStream>(
  std::cout,
  dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
triangulation(
  MPI_COMM_WORLD,
  typename dealii::Triangulation<dim>::MeshSmoothing(
  dealii::Triangulation<dim>::smoothing_on_refinement |
  dealii::Triangulation<dim>::smoothing_on_coarsening)),
fe_field(
  triangulation,
  1,
  1,
  flag_decohesion_is_allowed),
trial_microstress(
  triangulation,
  1),
  mapping_q(dealii::MappingQ<dim>(1))
{}



template<int dim>
void LinearAlgebra<dim>::run()
{
  make_grid();

  setup();

  create_mapping();

  check_extractors();

  check_global_component();

  check_projection();

  determine_active_set();

  eliminate_active_set();
}



template<int dim>
void LinearAlgebra<dim>::make_grid()
{
  { // Generate rectangle
    std::vector<unsigned int> repetitions(2, 1);

    repetitions[1] = 2;

    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0),
      dealii::Point<dim>(1.0, 2.0),
      true);
  }

  triangulation.refine_global(0);

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[1]) < 1.0)
      {
        cell->set_material_id(0);
      }
      else
      {
        cell->set_material_id(1);
      }
    }
  }
}



template<int dim>
void LinearAlgebra<dim>::setup()
{
  // Initiates the crystals' data (Slip directions, normals, orthogonals,
  // Schmid-Tensor and symmetrized Schmid-Tensors)
  crystals_data.init(
    triangulation,
    std::string(SOURCE_DIR) + "/input/2d_euler_angles",
    std::string(SOURCE_DIR) + "/input/2d_slip_directions",
    std::string(SOURCE_DIR) + "/input/2d_slip_normals");

  {// Setup of the FEField entity
    // Sets up the FEValuesExtractor instances
    fe_field.setup_extractors(
      crystals_data.get_n_crystals(),
      crystals_data.get_n_slips());

    trial_microstress.setup_extractors(
      crystals_data.get_n_crystals(),
      crystals_data.get_n_slips());

    // Update the material ids of ghost cells
    fe_field.update_ghost_material_ids();

    // Update the material ids of ghost cells
    trial_microstress.update_ghost_material_ids();

    // Set the active finite elemente index of each cell
    for (const auto &cell :
        fe_field.get_dof_handler().active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        cell->set_active_fe_index(cell->material_id());
      }
    }

    for (const auto &cell :
        trial_microstress.get_dof_handler().active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        cell->set_active_fe_index(cell->material_id());
      }
    }

    // Sets up the degrees of freedom
    fe_field.setup_dofs();

    trial_microstress.setup_dofs();

    // Sets up the solution vectors
    fe_field.setup_vectors();

    trial_microstress.setup_vectors();
  }
}



template <int dim>
void LinearAlgebra<dim>::create_mapping()
{
  // Initialize std::maps instances
  std::map<dealii::types::global_dof_index, dealii::Point<dim>>
    fe_field_map;

  std::map<dealii::types::global_dof_index, dealii::Point<dim>>
    trial_microstress_map;


  // Loop over crystals and slip systems
  for (unsigned int crystal_id = 0;
       crystal_id < crystals_data.get_n_crystals();
       crystal_id++)
  {
    for (unsigned int slip_id = 0;
        slip_id < crystals_data.get_n_slips();
        slip_id++)
    {
      //Clear std::maps instances
      fe_field_map.clear();

      trial_microstress_map.clear();

      // Get DoF-to-Support-Point mappings for a given slip system
      dealii::DoFTools::map_dofs_to_support_points(
        mapping_q,
        fe_field.get_dof_handler(),
        fe_field_map,
        fe_field.get_fe_collection().component_mask(
          fe_field.get_slip_extractor(crystal_id, slip_id)));

      dealii::DoFTools::map_dofs_to_support_points(
        mapping_q,
        trial_microstress.get_dof_handler(),
        trial_microstress_map,
        trial_microstress.get_fe_collection().component_mask(
          trial_microstress.get_extractor(crystal_id, slip_id)));

      // Both std::map instances have to be equal in sizq
      Assert(
        fe_field_map.size() == trial_microstress_map.size(),
        dealii::ExcMessage("Size mismatch!"))

      // Loop over the std::map instance
      for (auto fe_field_map_pair = fe_field_map.begin();
           fe_field_map_pair != fe_field_map.end();
           fe_field_map_pair++)
      {
        // Loop over the other std::map instance
        for (auto trial_microstress_map_pair = trial_microstress_map.begin();
            trial_microstress_map_pair != trial_microstress_map.end();
            trial_microstress_map_pair++)
        {
          // If the dealii::Point<dim> instance coincides, insert the
          // DoF-pair in the std::map instance
          if (fe_field_map_pair->second ==
              trial_microstress_map_pair->second)
          {
            mapping[trial_microstress_map_pair->first] =
              fe_field_map_pair->first;
          }
        } // Loop over the other std::map instance
      } // Loop over the std::map instance
    } // Loop over slip systems
  }  // Loop over crystals

  // Output mapping to file
  std::string filename = "mapping_" +
    std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

  std::ofstream file(filename.c_str());

  for(auto it = mapping.cbegin(); it != mapping.cend(); ++it)
  {
    file << it->first << " " << it->second << "\n";
  }
}



template <int dim>
void LinearAlgebra<dim>::compare_vectors(const double constant)
{

  bool equal_entries = true;

  for(auto pair = mapping.cbegin(); pair != mapping.cend(); ++pair)
  {
    if (std::abs(
          constant * fe_field.solution(pair->second) -
            trial_microstress.solution(pair->first)) >= 1e-10)
    {
      equal_entries = false;
    }
  }

  int n_processors;

  MPI_Comm_size(MPI_COMM_WORLD, &n_processors);

  if (dealii::Utilities::MPI::sum(
        static_cast<int>(equal_entries),MPI_COMM_WORLD) ==
          n_processors)
  {
    *pcout << "Passed" << std::endl;
  }
  else
  {
    *pcout << "Failed" << std::endl;
  }
}



template <int dim>
void LinearAlgebra<dim>::check_extractors()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_stress;

  distributed_solution.reinit(fe_field.distributed_vector);

  distributed_trial_stress.reinit(trial_microstress.distributed_vector);

  distributed_solution      = fe_field.solution;

  distributed_trial_stress  = trial_microstress.solution;

  for (unsigned int crystal_id = 0;
        crystal_id < crystals_data.get_n_crystals();
        ++crystal_id)
  {
    for(unsigned int slip_id = 0;
        slip_id < crystals_data.get_n_slips();
        ++slip_id)
    {
      dealii::VectorTools::interpolate(
        fe_field.get_dof_handler(),
        FEFieldFunction<dim>(
          fe_field.get_n_components(),
          crystals_data.get_n_crystals(),
          fe_field.is_decohesion_allowed()),
        distributed_solution,
        fe_field.get_fe_collection().component_mask(
          fe_field.get_slip_extractor(crystal_id, slip_id)));

      dealii::VectorTools::interpolate(
        trial_microstress.get_dof_handler(),
        TrialMicrostressFunction<dim>(
          trial_microstress.get_n_components(),
          crystals_data.get_n_crystals(),
          fe_field.is_decohesion_allowed()),
        distributed_trial_stress,
        trial_microstress.get_fe_collection().component_mask(
          trial_microstress.get_extractor(crystal_id, slip_id)));
    }
  }

  fe_field.get_hanging_node_constraints().distribute(
    distributed_solution);

  trial_microstress.get_hanging_node_constraints().distribute(
    distributed_trial_stress);

  fe_field.solution = distributed_solution;

  trial_microstress.solution = distributed_trial_stress;

  compare_vectors();
}



template <int dim>
void LinearAlgebra<dim>::check_global_component()
{
  // Fill FEField::solution based on the slip identifier and the
  // support point
  {
    dealii::AffineConstraints<double> affine_constraints;

    dealii::hp::QCollection<dim> q_collection;

    q_collection.push_back(dealii::QGauss<dim>(1));

    dealii::hp::FEValues<dim> hp_fe_values(
      fe_field.get_fe_collection(),
      q_collection,
      dealii::update_quadrature_points);

    affine_constraints.clear();
    {
      affine_constraints.reinit(fe_field.get_locally_relevant_dofs());
      affine_constraints.merge(fe_field.get_hanging_node_constraints());

      std::vector<dealii::types::global_dof_index> local_dof_indices(
        fe_field.get_fe_collection().max_dofs_per_cell());

      for (const auto &cell :
            fe_field.get_dof_handler().active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          const unsigned int crystal_id = cell->material_id();

          cell->get_dof_indices(local_dof_indices);

          hp_fe_values.reinit(cell);

          const dealii::FEValues<dim> &fe_values =
              hp_fe_values.get_present_fe_values();

          const std::vector<dealii::Point<dim>> unit_support_points =
            fe_values.get_fe().get_unit_support_points();

          for (unsigned int i = 0;
                i < local_dof_indices.size(); ++i)
          {
            const unsigned int slip_id =
              fe_field.get_global_component(crystal_id, i) - dim;

            if (fe_field.get_global_component(crystal_id, i) >= dim)
            {
              const double ith_value =
                (fe_values.get_mapping().transform_unit_to_real_cell(
                  cell,
                  unit_support_points[i]).norm() + 1)*
                (crystal_id + 1) *
                (slip_id + 1);

              affine_constraints.add_line(local_dof_indices[i]);

              affine_constraints.set_inhomogeneity(
                local_dof_indices[i],
                ith_value);
            }
          }
        }
      }

    }
    affine_constraints.close();

    fe_field.set_affine_constraints(affine_constraints);

    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

    distributed_solution.reinit(fe_field.distributed_vector);

    distributed_solution      = fe_field.solution;

    fe_field.get_affine_constraints().distribute(
      distributed_solution);

    fe_field.solution = distributed_solution;
  }

  // Fill TrailMicrostress::solution based on the slip identifier and
  // the support point
  {
    dealii::AffineConstraints<double> affine_constraints;

    dealii::hp::QCollection<dim> q_collection;

    q_collection.push_back(dealii::QGauss<dim>(1));

    dealii::hp::FEValues<dim> hp_fe_values(
      trial_microstress.get_fe_collection(),
      q_collection,
      dealii::update_quadrature_points);

    affine_constraints.clear();
    {
      affine_constraints.reinit(trial_microstress.get_locally_relevant_dofs());
      affine_constraints.merge(trial_microstress.get_hanging_node_constraints());

      std::vector<dealii::types::global_dof_index> local_dof_indices(
        trial_microstress.get_fe_collection().max_dofs_per_cell());

      for (const auto &cell :
            trial_microstress.get_dof_handler().active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          hp_fe_values.reinit(cell);

          const dealii::FEValues<dim> &fe_values =
              hp_fe_values.get_present_fe_values();

          const std::vector<dealii::Point<dim>> unit_support_points =
            fe_values.get_fe().get_unit_support_points();

          const unsigned int crystal_id = cell->material_id();

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0;
                i < local_dof_indices.size(); ++i)
          {
            const unsigned int slip_id =
              trial_microstress.get_global_component(crystal_id, i);

            const double ith_value =
              (fe_values.get_mapping().transform_unit_to_real_cell(
                cell,
                unit_support_points[i]).norm() + 1)*
              (crystal_id + 1) *
              (slip_id + 1);

            affine_constraints.add_line(local_dof_indices[i]);

            affine_constraints.set_inhomogeneity(
              local_dof_indices[i],
              ith_value);
          }
        }
      }
    }
    affine_constraints.close();

    trial_microstress.set_affine_constraints(affine_constraints);

    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

    distributed_solution.reinit(trial_microstress.distributed_vector);

    distributed_solution      = trial_microstress.solution;

    trial_microstress.get_affine_constraints().distribute(
      distributed_solution);

    trial_microstress.solution = distributed_solution;
  }

  compare_vectors();
}


template <int dim>
void LinearAlgebra<dim>::check_projection()
{
  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        lumped_matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        right_hand_side;

  { // Initialize matrix
    matrix.clear();

    dealii::TrilinosWrappers::SparsityPattern
      sparsity_pattern(
        trial_microstress.get_locally_owned_dofs(),
        trial_microstress.get_locally_owned_dofs(),
        trial_microstress.get_locally_relevant_dofs(),
        MPI_COMM_WORLD);

    dealii::DoFTools::make_sparsity_pattern(
      trial_microstress.get_dof_handler(),
      sparsity_pattern,
      trial_microstress.get_hanging_node_constraints(),
      false,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    sparsity_pattern.compress();

    matrix.reinit(sparsity_pattern);

    matrix = 0.;
  }

  { // Initialize vectors
    lumped_matrix.reinit(trial_microstress.distributed_vector);

    right_hand_side.reinit(trial_microstress.distributed_vector);

    lumped_matrix = 0.;

    right_hand_side = 0.;
  }

  dealii::hp::QCollection<dim> quadrature_collection;

  quadrature_collection.push_back(dealii::QGauss<dim>(2));

  // Initialize FEValues
  dealii::hp::FEValues<dim> trial_microstress_hp_fe_values(
    trial_microstress.get_fe_collection(),
    quadrature_collection,
    dealii::update_quadrature_points |
    dealii::update_values |
    dealii::update_JxW_values);

  dealii::hp::FEValues<dim> fe_field_hp_fe_values(
    fe_field.get_fe_collection(),
    quadrature_collection,
    dealii::update_quadrature_points |
    dealii::update_values |
    dealii::update_JxW_values);

  const unsigned int n_dofs_per_cell =
    trial_microstress.get_fe_collection().max_dofs_per_cell();

  const unsigned int n_quadrature_points =
    trial_microstress_hp_fe_values.get_quadrature_collection().
      max_n_quadrature_points();

  // Initialize local members
  dealii::FullMatrix<double> local_matrix(
    n_dofs_per_cell,
    n_dofs_per_cell);

  dealii::Vector<double> local_lumped_matrix(n_dofs_per_cell);

  dealii::Vector<double> local_right_hand_side(n_dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(
    n_dofs_per_cell);

  std::vector<std::vector<double>> test_function_values(
    crystals_data.get_n_slips(),
    std::vector<double>(n_dofs_per_cell));

  std::vector<std::vector<double>> slip_values(
    crystals_data.get_n_slips(),
    std::vector<double>(n_quadrature_points));

  std::vector<double> JxW_values(n_quadrature_points);

  for (const auto &cell :
        trial_microstress.get_dof_handler().active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      local_matrix = 0.;

      local_lumped_matrix = 0.;

      local_right_hand_side = 0.;

      cell->get_dof_indices(local_dof_indices);

      const unsigned int crystal_id = cell->material_id();

      // Initialize FEValues w.r.t. current cell
      trial_microstress_hp_fe_values.reinit(cell);

      const dealii::FEValues<dim> &fe_values =
        trial_microstress_hp_fe_values.get_present_fe_values();

      typename dealii::DoFHandler<dim>::active_cell_iterator
        fe_field_cell(&fe_field.get_triangulation(),
                      cell->level(),
                      cell->index(),
                      &fe_field.get_dof_handler());

      fe_field_hp_fe_values.reinit(fe_field_cell);

      const dealii::FEValues<dim> &fe_field_fe_values =
        fe_field_hp_fe_values.get_present_fe_values();

      for (unsigned int slip_id = 0;
            slip_id < fe_field.get_n_slips(); ++slip_id)
      {
        fe_field_fe_values[
          fe_field.get_slip_extractor(crystal_id, slip_id)].
            get_function_values(
              fe_field.solution,
              slip_values[slip_id]);
      }

      JxW_values = fe_values.get_JxW_values();

      for (unsigned int quadrature_point_id = 0;
           quadrature_point_id < n_quadrature_points;
           quadrature_point_id++)
      {
        for (unsigned int slip_id = 0;
             slip_id < fe_field.get_n_slips(); ++slip_id)
        {
          for (unsigned int local_dof_id = 0;
               local_dof_id < n_dofs_per_cell;
               local_dof_id++)
          {
            test_function_values[slip_id][local_dof_id] =
              fe_values[trial_microstress.get_extractor(
                  crystal_id,
                  slip_id)].
                    value(local_dof_id, quadrature_point_id);
          }
        }

        for (unsigned int local_row_dof_id = 0;
            local_row_dof_id < n_dofs_per_cell;
            local_row_dof_id++)
        {
          const unsigned int row_slip_id =
            trial_microstress.get_global_component(
              crystal_id,
              local_row_dof_id);

          local_right_hand_side(local_row_dof_id) +=
            test_function_values[row_slip_id][local_row_dof_id] *
            10. *
            slip_values[row_slip_id][quadrature_point_id] *
            JxW_values[quadrature_point_id];

          for (unsigned int local_column_dof_id = 0;
              local_column_dof_id < n_dofs_per_cell;
              local_column_dof_id++)
          {
            const unsigned int column_slip_id =
              trial_microstress.get_global_component(
                crystal_id,
                local_column_dof_id);

            if (row_slip_id != column_slip_id)
            {
              continue;
            }

            local_matrix(local_row_dof_id, local_column_dof_id) +=
              test_function_values[row_slip_id][local_row_dof_id] *
              test_function_values[column_slip_id][local_column_dof_id] *
              JxW_values[quadrature_point_id];

            local_lumped_matrix(local_row_dof_id) +=
              test_function_values[row_slip_id][local_row_dof_id] *
              test_function_values[column_slip_id][local_column_dof_id] *
              JxW_values[quadrature_point_id];
          }
        }
      }

      trial_microstress.get_hanging_node_constraints().
        distribute_local_to_global(
          local_matrix,
          local_dof_indices,
          matrix);

      trial_microstress.get_hanging_node_constraints().
        distribute_local_to_global(
          local_lumped_matrix,
          local_dof_indices,
          lumped_matrix);

      trial_microstress.get_hanging_node_constraints().
        distribute_local_to_global(
          local_right_hand_side,
          local_dof_indices,
          right_hand_side);
    }
  }

  { // Solve
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

    distributed_solution.reinit(trial_microstress.distributed_vector);

    distributed_solution = 0.;

    {
      dealii::SolverControl solver_control(
        100,
        1e-8);

      dealii::TrilinosWrappers::SolverDirect solver(solver_control);

      try
      {
        solver.solve(matrix, distributed_solution, right_hand_side);
      }
      catch (std::exception &exc)
      {
        std::cerr << "Exception in the solve method: " << std::endl
                  << exc.what() << std::endl;
      }
      catch (...)
      {
        std::cerr << "Unknown exception in the solve method!" << std::endl
                  << "Aborting!" << std::endl;
      }
    }

    trial_microstress.get_hanging_node_constraints().distribute(
      distributed_solution);

    trial_microstress.solution = distributed_solution;

    compare_vectors(10.);

    distributed_solution = 0.;

    {
      for (unsigned int entry_id = 0;
            entry_id < lumped_matrix.size();
            entry_id++)
      {
        if (trial_microstress.get_locally_owned_dofs().
              is_element(entry_id))
        {
          if (lumped_matrix(entry_id) != 0.0)
          {
            distributed_solution(entry_id) =
              right_hand_side(entry_id) /
              lumped_matrix(entry_id);
          }
        }
      }
    }

    trial_microstress.get_hanging_node_constraints().distribute(
      distributed_solution);

    trial_microstress.solution = distributed_solution;
  }
}



template <int dim>
void LinearAlgebra<dim>::determine_active_set()
{
  index_set.clear();

  index_set.set_size(fe_field.get_dof_handler().n_dofs());

  //trial_microstress.solution.print(std::cout, 1, false);

  dealii::LinearAlgebraTrilinos::MPI::Vector active_set;

  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_active_set;

  active_set.reinit(trial_microstress.solution);

  distributed_active_set.reinit(trial_microstress.distributed_vector);

  active_set = 0.;

  distributed_active_set = 0.;

  for (const auto &locally_owned_dof :
        trial_microstress.get_locally_owned_dofs())
  {
    const bool plastic_flow =
      trial_microstress.solution[locally_owned_dof] > 30.;

    if (plastic_flow)
    {
      distributed_active_set[locally_owned_dof] = plastic_flow;

      index_set.add_index(mapping[locally_owned_dof]);
    }

  }

  trial_microstress.get_hanging_node_constraints().distribute(
    distributed_active_set);

  active_set = distributed_active_set;

  //active_set.print(std::cout, 1, false);

  //index_set.print(std::cout);
}



template <int dim>
void LinearAlgebra<dim>::eliminate_active_set()
{
  dealii::LinearAlgebraTrilinos::MPI::SparseMatrix  matrix;

  dealii::LinearAlgebraTrilinos::MPI::Vector        right_hand_side;

  { // Initialize matrix
    matrix.clear();

    dealii::TrilinosWrappers::SparsityPattern
      sparsity_pattern(
        fe_field.get_locally_owned_dofs(),
        fe_field.get_locally_owned_dofs(),
        fe_field.get_locally_relevant_dofs(),
        MPI_COMM_WORLD);

    dealii::DoFTools::make_sparsity_pattern(
      fe_field.get_dof_handler(),
      sparsity_pattern,
      fe_field.get_hanging_node_constraints(),
      false,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    sparsity_pattern.compress();

    matrix.reinit(sparsity_pattern);

    matrix = 0.;
  }

  { // Initialize vectors
    right_hand_side.reinit(fe_field.distributed_vector);

    right_hand_side = 0.;
  }

  dealii::hp::QCollection<dim> quadrature_collection;

  quadrature_collection.push_back(dealii::QGauss<dim>(2));

  // Initialize FEValues
  dealii::hp::FEValues<dim> fe_field_hp_fe_values(
    fe_field.get_fe_collection(),
    quadrature_collection,
    dealii::update_quadrature_points |
    dealii::update_values |
    dealii::update_JxW_values);

  const unsigned int n_dofs_per_cell =
    fe_field.get_fe_collection().max_dofs_per_cell();

  const unsigned int n_quadrature_points =
    fe_field_hp_fe_values.get_quadrature_collection().
      max_n_quadrature_points();

  // Initialize local members
  dealii::FullMatrix<double> local_matrix(
    n_dofs_per_cell,
    n_dofs_per_cell);

  dealii::Vector<double> local_right_hand_side(n_dofs_per_cell);

  std::vector<dealii::types::global_dof_index>
    fe_field_local_dof_indices(n_dofs_per_cell);

  std::vector<dealii::types::global_dof_index>
    trial_microstress_local_dof_indices(
      trial_microstress.get_fe_collection().max_dofs_per_cell());

  std::vector<std::vector<double>> test_function_values(
    crystals_data.get_n_slips(),
    std::vector<double>(n_dofs_per_cell));

  std::vector<std::vector<double>> slip_values(
    crystals_data.get_n_slips(),
    std::vector<double>(n_quadrature_points));

  std::vector<double> JxW_values(n_quadrature_points);

  for (const auto &fe_field_cell :
        fe_field.get_dof_handler().active_cell_iterators())
  {
    if (fe_field_cell->is_locally_owned())
    {
      local_matrix = 0.;

      local_right_hand_side = 0.;

      const unsigned int crystal_id = fe_field_cell->material_id();

      // Initialize FEValues w.r.t. current cell
      fe_field_hp_fe_values.reinit(fe_field_cell);

      const dealii::FEValues<dim> &fe_field_fe_values =
        fe_field_hp_fe_values.get_present_fe_values();

      typename dealii::DoFHandler<dim>::active_cell_iterator
        trial_microstress_cell(
          &fe_field.get_triangulation(),
          fe_field_cell->level(),
          fe_field_cell->index(),
          &trial_microstress.get_dof_handler());

      fe_field_cell->get_dof_indices(fe_field_local_dof_indices);

      trial_microstress_cell->get_dof_indices(
        trial_microstress_local_dof_indices);

      for (unsigned int slip_id = 0;
            slip_id < fe_field.get_n_slips(); ++slip_id)
      {
        fe_field_fe_values[
          fe_field.get_slip_extractor(crystal_id, slip_id)].
            get_function_values(
              fe_field.solution,
              slip_values[slip_id]);
      }

      JxW_values = fe_field_fe_values.get_JxW_values();

      for (unsigned int quadrature_point_id = 0;
           quadrature_point_id < n_quadrature_points;
           quadrature_point_id++)
      {
        for (unsigned int slip_id = 0;
             slip_id < fe_field.get_n_slips(); ++slip_id)
        {
          for (unsigned int local_dof_id = 0;
               local_dof_id < n_dofs_per_cell;
               local_dof_id++)
          {
            test_function_values[slip_id][local_dof_id] =
              fe_field_fe_values[fe_field.get_slip_extractor(
                crystal_id,
                slip_id)].
                  value(local_dof_id, quadrature_point_id);
          }
        }

        for (unsigned int local_row_dof_id = 0;
            local_row_dof_id < n_dofs_per_cell;
            local_row_dof_id++)
        {
          const unsigned int row_slip_id =
            fe_field.get_global_component(
              crystal_id, local_row_dof_id) - dim;

          if (fe_field.get_global_component(
                crystal_id,
                local_row_dof_id) >= dim)
          {
            local_right_hand_side(local_row_dof_id) +=
              test_function_values[row_slip_id][local_row_dof_id] *
              slip_values[row_slip_id][quadrature_point_id] *
              JxW_values[quadrature_point_id];
          }

          for (unsigned int local_column_dof_id = 0;
              local_column_dof_id < n_dofs_per_cell;
              local_column_dof_id++)
          {
            if (fe_field.get_global_component(
                  crystal_id, local_row_dof_id) >= dim)
            {
              if (fe_field.get_global_component(
                    crystal_id, local_column_dof_id) >= dim)
              {
                const unsigned int column_slip_id =
                  fe_field.get_global_component(
                    crystal_id,
                    local_column_dof_id) - dim;

                if (row_slip_id != column_slip_id)
                {
                  continue;
                }

                local_matrix(local_row_dof_id, local_column_dof_id) +=
                  test_function_values[row_slip_id][local_row_dof_id] *
                  test_function_values[column_slip_id][local_column_dof_id] *
                  JxW_values[quadrature_point_id];
              }
            }
          }
        }
      }

      //local_matrix.print(std::cout);

      //local_right_hand_side.print(std::cout);

      for (unsigned int local_row_dof_id = 0;
          local_row_dof_id < n_dofs_per_cell;
          local_row_dof_id++)
      {

        if (index_set.is_element(
            fe_field_local_dof_indices[local_row_dof_id]))
        {
          local_right_hand_side(local_row_dof_id) *= 1e-10;
        }

        for (unsigned int local_column_dof_id = 0;
            local_column_dof_id < n_dofs_per_cell;
            local_column_dof_id++)
        {
          if (index_set.is_element(
              fe_field_local_dof_indices[local_row_dof_id]) ||
              index_set.is_element(
              fe_field_local_dof_indices[local_column_dof_id]))
          {
            local_matrix(local_row_dof_id, local_column_dof_id) *=
              1e-10;
          }
        }
      }

      //local_matrix.print(std::cout);

      //local_right_hand_side.print(std::cout);

      fe_field.get_hanging_node_constraints().
        distribute_local_to_global(
          local_matrix,
          fe_field_local_dof_indices,
          matrix);

      fe_field.get_hanging_node_constraints().
        distribute_local_to_global(
          local_right_hand_side,
          fe_field_local_dof_indices,
          right_hand_side);
    }
  }
}



template <int dim>
void LinearAlgebra<dim>::print() const
{
  fe_field.solution.print(std::cout, 2, false, true);

  std::cout << std::endl << std::flush;

  trial_microstress.solution.print(std::cout, 2, false, true);

  std::cout << std::endl << std::flush;
}



template <int dim>
void LinearAlgebra<dim>::vtk_output() const
{
  dealii::DataOut<dim> data_out;

  data_out.add_data_vector(
    fe_field.get_dof_handler(),
    fe_field.solution,
    "fe_field");

  data_out.add_data_vector(
    trial_microstress.get_dof_handler(),
    trial_microstress.solution,
    "trial_microstress");

  data_out.build_patches(0);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    "./",
    "solution",
    out_index,
    MPI_COMM_WORLD,
    5);

  out_index++;
}



} // namespace Tests



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc,
      argv,
      dealii::numbers::invalid_unsigned_int);

    bool flag_decohesion_is_allowed = false;

    switch (argc)
    {
      case 1:
      {
        flag_decohesion_is_allowed = false;
      }
      break;

      case 2:
      {
        const std::string arg(argv[1]);

        if (arg == "true")
        {
          flag_decohesion_is_allowed = true;
        }
        else
        {
          flag_decohesion_is_allowed = false;
        }
      }
      break;
    }

    Tests::LinearAlgebra<2> test(flag_decohesion_is_allowed);

    test.run();
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