#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/numerics/data_out.h>


namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::solve_nonlinear_system()
{
  solution = fe_field->solution;

  unsigned int nonlinear_iteration = 0;

  assemble_residual();
  /*!
   * @todo This loop needs some work. No line search is performed.
   */
  for (;nonlinear_iteration < parameters.n_max_nonlinear_iterations;
       ++nonlinear_iteration)
  {
    if (residual_norm < parameters.nonlinear_tolerance)
      break;

    assemble_jacobian();

    solve_linearized_system();

    assemble_residual();

    if (nonlinear_solver_log)
      nonlinear_solver_log << std::setw(19) << std::right
                           << nonlinear_iteration << std::string(5, ' ')
                           << std::setw(39) << std::right
                           << std::fixed << std::scientific
                           << std::showpos << std::setprecision(6)
                           << residual_norm << std::endl;

  }

  AssertThrow(nonlinear_iteration <
                parameters.n_max_nonlinear_iterations,
              dealii::ExcMessage(
                "The nonlinear solver has reach the given maximum "
                "number of iterations ("
                + std::to_string(parameters.n_max_nonlinear_iterations)
                + ")."));

  fe_field->solution = solution;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::solve_linearized_system()
{
  if (parameters.verbose)
    *pcout << std::setw(38) << std::left
           << "  Solver: Solving linearized system...";

  dealii::TimerOutput::Scope  t(*timer_output, "Solver: Solve ");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_solution.reinit(fe_field->distributed_vector);
  distributed_newton_update.reinit(fe_field->distributed_vector);

  distributed_solution      = solution;
  distributed_newton_update = solution;

  // The solver's tolerances are passed to the SolverControl instance
  // used to initialize the solver
  dealii::SolverControl solver_control(
    parameters.n_max_krylov_iterations,
    std::max(residual_norm * parameters.krylov_relative_tolerance,
             parameters.krylov_absolute_tolerance));

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  // The preconditioner is instanciated and initialized
  dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;

  preconditioner.initialize(jacobian);

  // try-catch scope for the solve() call
  try
  {
    solver.solve(jacobian,
                 distributed_newton_update,
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
  // Zero out the Dirichlet boundary conditions
  fe_field->get_newton_method_constraints().distribute(
    distributed_newton_update);

  // Compute the updated solution
  distributed_solution.add(1.0,
                           distributed_newton_update);

  fe_field->get_affine_constraints().distribute(
    distributed_solution);

  // Pass the distributed vectors to their ghosted counterpart
  solution      = distributed_solution;
  newton_update = distributed_newton_update;

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



} // namespace gCP


template void gCP::GradientCrystalPlasticitySolver<2>::solve_nonlinear_system();
template void gCP::GradientCrystalPlasticitySolver<3>::solve_nonlinear_system();

template void gCP::GradientCrystalPlasticitySolver<2>::solve_linearized_system();
template void gCP::GradientCrystalPlasticitySolver<3>::solve_linearized_system();
