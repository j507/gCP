#include <gCP/gradient_crystal_plasticity.h>



namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::solve()
{
  assemble_jacobian();

  assemble_residual();

  dealii::TimerOutput::Scope  t(*timer_output, "Solver: Solve ");
  {
    // In this method we create temporal non ghosted copies
    // of the pertinent vectors to be able to perform the solve()
    // operation.
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;
    //dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

    distributed_solution.reinit(fe_field->distributed_vector);
    //distributed_newton_update.reinit(distributed_solution);

    distributed_solution      = fe_field->solution;
    //distributed_newton_update = newton_update;

    // The solver's tolerances are passed to the SolverControl instance
    // used to initialize the solver
    dealii::SolverControl solver_control(
      parameters.n_maximum_iterations,
      std::max(residual_norm * parameters.relative_tolerance,
               parameters.absolute_tolerance));

    dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

    // The preconditioner is instanciated and initialized
    dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;

    preconditioner.initialize(jacobian);

    // try-catch scope for the solve() call
    try
    {
      solver.solve(jacobian,
                   distributed_solution, // distributed_newton_update
                   residual,
                   preconditioner);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in the solve method: " << std::endl
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
      std::cerr << "Unknown exception in the solve method!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
    /*
    // Zero out the Dirichlet boundary conditions
    fe_field->get_newton_method_constraints().distribute(
      distributed_newton_update);

    // Compute the updated solution
    distributed_solution.add(relaxation_parameter,
                            distributed_newton_update);*/

    fe_field->get_affine_constraints().distribute(
      distributed_solution);

    // Pass the distributed vectors to their ghosted counterpart
    //newton_update       = distributed_newton_update;
    fe_field->solution  = distributed_solution;
  }
}



} // namespace gCP


template void gCP::GradientCrystalPlasticitySolver<2>::solve();
template void gCP::GradientCrystalPlasticitySolver<3>::solve();
