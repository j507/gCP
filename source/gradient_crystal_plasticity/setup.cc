#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/dofs/dof_tools.h>

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
       fe_field->get_triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face_id : cell->face_indices())
        if (!cell->face(face_id)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_id)->material_id())
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

    dealii::DoFTools::make_sparsity_pattern(
      fe_field->get_dof_handler(),
      sparsity_pattern,
      fe_field->get_newton_method_constraints(),
      false,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    sparsity_pattern.compress();

    jacobian.reinit(sparsity_pattern);
  }

  // Initiate constitutive laws
  hooke_law->init();

  vector_microscopic_stress_law->init();

  init_quadrature_point_history();

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
void GradientCrystalPlasticitySolver<dim>::init_quadrature_point_history()
{
  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  const unsigned int n_q_points =
    quadrature_collection.max_n_quadrature_points();

  quadrature_point_history.initialize(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               fe_field->get_dof_handler().end()),
    n_q_points);

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

template void gCP::GradientCrystalPlasticitySolver<2>::init_quadrature_point_history();
template void gCP::GradientCrystalPlasticitySolver<3>::init_quadrature_point_history();