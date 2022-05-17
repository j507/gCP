#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/dofs/dof_tools.h>

namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::init()
{
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

  // Initiate

  flag_init_was_called = true;
}


template <int dim>
void GradientCrystalPlasticitySolver<dim>::set_supply_term(
  std::shared_ptr<dealii::Function<dim>> &supply_term)
{
  this->supply_term = supply_term;
}

} // namespace gCP