#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>

namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::init()
{
  if (parameters.verbose)
    *pcout << std::setw(38) << std::left
           << "  Solver: Initializing solver...";

  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Initialize");

  AssertThrow(fe_field->is_initialized(),
              dealii::ExcMessage("The underlying FEField<dim> instance"
                                 " has not been initialized."))
  AssertThrow(crystals_data->is_initialized(),
              dealii::ExcMessage("The underlying CrystalsData<dim>"
                                 " instance has not been "
                                 " initialized."));

  // Initiate vectors
  trial_solution.reinit(fe_field->solution);
  newton_update.reinit(fe_field->solution);
  residual.reinit(fe_field->distributed_vector);
  cell_is_at_grain_boundary.reinit(
    fe_field->get_triangulation().n_active_cells());

  trial_solution            = 0.0;
  newton_update             = 0.0;
  residual                  = 0.0;
  cell_is_at_grain_boundary = 0.0;

  // Identify which cells are located at a grain boundary
  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->active_fe_index() !=
              cell->neighbor(face_index)->active_fe_index())
        {
          cell_is_at_grain_boundary(cell->active_cell_index()) = 1.0;
          break;
        }

  // Initiate Jacobian matrix
  {
    jacobian.clear();

    dealii::TrilinosWrappers::SparsityPattern
      sparsity_pattern(fe_field->get_locally_owned_dofs(),
                       fe_field->get_locally_owned_dofs(),
                       fe_field->get_locally_relevant_dofs(),
                       MPI_COMM_WORLD);
    /*
    dealii::DoFTools::make_flux_sparsity_pattern(
      fe_field->get_dof_handler(),
      sparsity_pattern,
      fe_field->get_newton_method_constraints(),
      false,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    */

    make_sparsity_pattern(sparsity_pattern);

    sparsity_pattern.compress();

    if (parameters.print_sparsity_pattern)
    {
      *pcout
        << "Printing the sparsity pattern in the gnplot file "
        << (parameters.logger_output_directory + "sparsity_pattern.gpl")
        << ". This might take a while... " << std::flush;

      std::ofstream out(parameters.logger_output_directory +
                        "sparsity_pattern.gpl");

      sparsity_pattern.print_gnuplot(out);
      *pcout << "done! \n\n";
    }

    jacobian.reinit(sparsity_pattern);
  }

  // Initiate constitutive laws
  hooke_law->init();

  vector_microscopic_stress_law->init();

  init_quadrature_point_history();

  // Check boundary ids of the Neumann boundary conditions
  {
    const std::vector<dealii::types::boundary_id> boundary_ids =
      fe_field->get_triangulation().get_boundary_ids();

    for (const auto &neumann_boundary_condition : neumann_boundary_conditions)
    {
      (void)neumann_boundary_condition;
      Assert(
        std::find(boundary_ids.begin(), boundary_ids.end(), neumann_boundary_condition.first)
          != boundary_ids.end(),
        dealii::ExcMessage("The boundary id assigned does not exist in"
        " the dealii::parallel::Triangulation<dim> instance"));
    }
  }

  // Set-up memberes related to the L2 projection of the damage variable
  {
    // The FE collection consists of a single second order
    // Lagrange-Element
    projection_fe_collection.push_back(dealii::FE_Q<dim>(2));

    // Distribute degrees of freedom based on the defined finite elements
    projection_dof_handler.reinit(fe_field->get_triangulation());

    projection_dof_handler.distribute_dofs(projection_fe_collection);

    // Renumbering of the degrees of freedom
    dealii::DoFRenumbering::Cuthill_McKee(projection_dof_handler);

    // Get the locally owned and relevant degrees of freedom of
    // each processor
    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    locally_owned_dofs = projection_dof_handler.locally_owned_dofs();

    dealii::DoFTools::extract_locally_relevant_dofs(
      projection_dof_handler,
      locally_relevant_dofs);

    // Initiate the hanging node constraints
    projection_hanging_node_constraints.clear();
    {
      projection_hanging_node_constraints.reinit(locally_relevant_dofs);

      dealii::DoFTools::make_hanging_node_constraints(
        projection_dof_handler,
        projection_hanging_node_constraints);
    }
    projection_hanging_node_constraints.close();

    // Initiate vectors
    {
      damage_variable_values.reinit(locally_relevant_dofs,
                                    MPI_COMM_WORLD);
      projection_rhs.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            MPI_COMM_WORLD,
                            true);
      lumped_projection_matrix.reinit(locally_owned_dofs,
                                      locally_relevant_dofs,
                                      MPI_COMM_WORLD,
                                      true);
    }

    assemble_projection_matrix();
  } // End of set-up memberes related to the L2 projection of the
    // damage variable

  flag_init_was_called = true;

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::set_supply_term(
  std::shared_ptr<dealii::TensorFunction<1,dim>> supply_term)
{
  this->supply_term = supply_term;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::
set_neumann_boundary_condition(
  const dealii::types::boundary_id                      boundary_id,
  const std::shared_ptr<dealii::TensorFunction<1,dim>>  function)
{
  Assert(
    function.get() != nullptr,
    dealii::ExcMessage(
      "The Neumann boundary conditions's shared pointer contains a "
      "nullptr."));

  neumann_boundary_conditions[boundary_id] = function;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::make_sparsity_pattern(
  dealii::TrilinosWrappers::SparsityPattern &sparsity_pattern)
{
  const dealii::types::subdomain_id subdomain_id =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const dealii::types::global_dof_index n_dofs =
    fe_field->get_dof_handler().n_dofs();
  (void)n_dofs;

  Assert(sparsity_pattern.n_rows() == n_dofs,
         dealii::ExcDimensionMismatch(sparsity_pattern.n_rows(), n_dofs));
  Assert(sparsity_pattern.n_cols() == n_dofs,
         dealii::ExcDimensionMismatch(sparsity_pattern.n_cols(), n_dofs));

  if (const auto *triangulation = dynamic_cast<
        const dealii::parallel::DistributedTriangulationBase<dim, dim> *>(
        &(fe_field->get_dof_handler()).get_triangulation()))
    {
      Assert(
        (subdomain_id == dealii::numbers::invalid_subdomain_id) ||
          (subdomain_id == triangulation->locally_owned_subdomain()),
        dealii::ExcMessage(
          "For distributed Triangulation objects and associated "
          "DoFHandler objects, asking for any subdomain other than the "
          "locally owned one does not make sense."));
    }

  std::vector<dealii::types::global_dof_index> dof_indices_on_this_cell;
  std::vector<dealii::types::global_dof_index> dof_indices_on_neighbour_cell;

  dof_indices_on_this_cell.reserve(
    fe_field->get_dof_handler().get_fe_collection().max_dofs_per_cell());
  dof_indices_on_neighbour_cell.reserve(
    fe_field->get_dof_handler().get_fe_collection().max_dofs_per_cell());

  for (const auto &cell :
       fe_field->get_dof_handler().active_cell_iterators())
    if (((subdomain_id == dealii::numbers::invalid_subdomain_id) ||
          (subdomain_id == cell->subdomain_id())) &&
        cell->is_locally_owned())
      {
        AssertThrow(
          cell->active_fe_index() == cell->material_id(),
          dealii::ExcMessage(
            "The active finite element index and the material "
            " identifier of the cell have to coincide!"));

        const unsigned int n_dofs_on_this_cell = cell->get_fe().n_dofs_per_cell();
        dof_indices_on_this_cell.resize(n_dofs_on_this_cell);
        cell->get_dof_indices(dof_indices_on_this_cell);

        fe_field->get_newton_method_constraints().add_entries_local_to_global(
          dof_indices_on_this_cell,
          sparsity_pattern,
          false);

        if (cell_is_at_grain_boundary(cell->active_cell_index()) &&
            parameters.boundary_conditions_at_grain_boundaries ==
             RunTimeParameters::BoundaryConditionsAtGrainBoundaries::Microtraction)
          for (const auto &face_index : cell->face_indices())
            if (!cell->face(face_index)->at_boundary() &&
                cell->active_fe_index() !=
                  cell->neighbor(face_index)->active_fe_index())
            {
              AssertThrow(
                cell->neighbor(face_index)->active_fe_index() ==
                  cell->neighbor(face_index)->material_id(),
                dealii::ExcMessage(
                  "The active finite element index and the material "
                  " identifier of the cell have to coincide!"));

              const unsigned int n_dofs_on_neighbour_cell =
               cell->neighbor(face_index)->get_fe().n_dofs_per_cell();
              dof_indices_on_neighbour_cell.resize(
                n_dofs_on_neighbour_cell);
              cell->neighbor(face_index)->get_dof_indices(
                dof_indices_on_neighbour_cell);

              fe_field->get_newton_method_constraints().add_entries_local_to_global(
                dof_indices_on_this_cell,
                dof_indices_on_neighbour_cell,
                sparsity_pattern,
                false);
            }
      }
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::init_quadrature_point_history()
{
  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  const unsigned int n_q_points =
    quadrature_collection.max_n_quadrature_points();

  const unsigned int n_face_q_points =
    face_quadrature_collection.max_n_quadrature_points();

  quadrature_point_history.initialize(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    n_q_points);

  interface_quadrature_point_history.initialize(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    n_face_q_points);

  for (const auto &cell : fe_field->get_triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
    {
      const std::vector<std::shared_ptr<QuadraturePointHistory<dim>>>
        local_quadrature_point_history =
          quadrature_point_history.get_data(cell);

      Assert(local_quadrature_point_history.size() == n_q_points,
              dealii::ExcInternalError());

      for (unsigned int q_point = 0;
            q_point < n_q_points;
            ++q_point)
        local_quadrature_point_history[q_point]->init(
          parameters.scalar_microscopic_stress_law_parameters,
          crystals_data->get_n_slips());

      if (cell_is_at_grain_boundary(cell->active_cell_index()) &&
          fe_field->is_decohesion_allowed())
        for (const auto &face_index : cell->face_indices())
          if (!cell->face(face_index)->at_boundary() &&
              cell->material_id() !=
                cell->neighbor(face_index)->material_id())
          {
            const std::vector<std::shared_ptr<InterfaceQuadraturePointHistory<dim>>>
              local_interface_quadrature_point_history =
                interface_quadrature_point_history.get_data(
                  cell->id(),
                  cell->neighbor(face_index)->id());

            Assert(local_interface_quadrature_point_history.size() ==
                     n_face_q_points,
                   dealii::ExcInternalError());

            for (unsigned int face_q_point = 0;
                  face_q_point < n_face_q_points; ++face_q_point)
              local_interface_quadrature_point_history[face_q_point]->init(
                parameters.cohesive_law_parameters);
          }

    }
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::print_decohesion_data()
{
  for (const auto &cell : fe_field->get_triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      if (cell_is_at_grain_boundary(cell->active_cell_index()) &&
          fe_field->is_decohesion_allowed())
        for (const auto &face_index : cell->face_indices())
          if (!cell->face(face_index)->at_boundary() &&
              cell->material_id() !=
                cell->neighbor(face_index)->material_id())
          {
            const std::vector<std::shared_ptr<InterfaceQuadraturePointHistory<dim>>>
              local_interface_quadrature_point_history =
                interface_quadrature_point_history.get_data(
                  cell->id(),
                  cell->neighbor(face_index)->id());

            decohesion_logger.add_value(
              "time",
              discrete_time.get_next_time());
            decohesion_logger.add_value(
              "max_effective_opening_displacement",
              local_interface_quadrature_point_history[0]->
                get_max_effective_opening_displacement());
            decohesion_logger.add_value(
              "effective_opening_displacement",
              local_interface_quadrature_point_history[0]->
                get_effective_opening_displacement());
            decohesion_logger.add_value(
              "normal_opening_displacement",
              local_interface_quadrature_point_history[0]->
                get_normal_opening_displacement());
            decohesion_logger.add_value(
              "tangential_opening_displacement",
              local_interface_quadrature_point_history[0]->
                get_tangential_opening_displacement());
            decohesion_logger.add_value(
              "effective_cohesive_traction",
              local_interface_quadrature_point_history[0]->
                get_effective_cohesive_traction());
            decohesion_logger.add_value(
              "damage_variable",
              local_interface_quadrature_point_history[0]->
                get_damage_variable());

            std::ofstream fstream(
              parameters.logger_output_directory + "decohesion_log.txt");

            decohesion_logger.write_text(
              fstream,
              dealii::TableHandler::TextOutputFormat::org_mode_table);

            return;
          }

}



} // namespace gCP



// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::init();
template void gCP::GradientCrystalPlasticitySolver<3>::init();

template void gCP::GradientCrystalPlasticitySolver<2>::set_supply_term(
  std::shared_ptr<dealii::TensorFunction<1,2>>);
template void gCP::GradientCrystalPlasticitySolver<3>::set_supply_term(
  std::shared_ptr<dealii::TensorFunction<1,3>>);

template void gCP::GradientCrystalPlasticitySolver<2>::set_neumann_boundary_condition(
  const dealii::types::boundary_id,
  const std::shared_ptr<dealii::TensorFunction<1,2>>);
template void gCP::GradientCrystalPlasticitySolver<3>::set_neumann_boundary_condition(
  const dealii::types::boundary_id,
  const std::shared_ptr<dealii::TensorFunction<1,3>>);

template void
gCP::GradientCrystalPlasticitySolver<2>::make_sparsity_pattern(
   dealii::TrilinosWrappers::SparsityPattern &);
template void
gCP::GradientCrystalPlasticitySolver<3>::make_sparsity_pattern(
   dealii::TrilinosWrappers::SparsityPattern &);

template void gCP::GradientCrystalPlasticitySolver<2>::init_quadrature_point_history();
template void gCP::GradientCrystalPlasticitySolver<3>::init_quadrature_point_history();

template void gCP::GradientCrystalPlasticitySolver<2>::print_decohesion_data();
template void gCP::GradientCrystalPlasticitySolver<3>::print_decohesion_data();
