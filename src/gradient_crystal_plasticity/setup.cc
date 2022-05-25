#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/dofs/dof_tools.h>

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

  // Initiate vectors
  solution.reinit(fe_field->solution);
  newton_update.reinit(fe_field->solution);
  residual.reinit(fe_field->distributed_vector);

  // Distribute the constraints to the pertinenet vectors
  // The distribute() method only works on distributed vectors.
  {
    // Initiate the distributed vector
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_vector;

    distributed_vector.reinit(fe_field->distributed_vector);

    // Distribute the affine constraints to the solution vector
    distributed_vector = solution;

    fe_field->get_affine_constraints().distribute(distributed_vector);

    fe_field->solution  = distributed_vector;

    // Distribute the affine constraints to the newton update vector
    distributed_vector  = newton_update;

    fe_field->get_newton_method_constraints().distribute(distributed_vector);

    newton_update       = distributed_vector;
  }

  // Initiate constitutive laws
  elastic_strain->init(fe_field->get_extractors());

  hooke_law->init();

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
  const unsigned int n_q_points =
    quadrature_collection.max_n_quadrature_points();

  quadrature_point_history.initialize(
    fe_field->get_triangulation().begin_active(),
    fe_field->get_triangulation().end(),
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
