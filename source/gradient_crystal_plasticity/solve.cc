#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/numerics/data_out.h>


namespace gCP
{



template <int dim>
void GradientCrystalPlasticitySolver<dim>::distribute_constraints_to_trial_solution()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

  distributed_trial_solution.reinit(fe_field->distributed_vector);

  distributed_trial_solution = trial_solution;

  fe_field->get_affine_constraints().distribute(distributed_trial_solution);

  trial_solution = distributed_trial_solution;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::solve_nonlinear_system()
{
  nonlinear_solver_logger.add_break(
    "Solving for t = " +
    std::to_string(discrete_time.get_next_time())+
    " with dt = " +
    std::to_string(discrete_time.get_next_step_size()));

  nonlinear_solver_logger.log_headers_to_terminal();

  double relaxation_parameter = 1.0;

  distribute_constraints_to_trial_solution();

  prepare_quadrature_point_history();

  unsigned int nonlinear_iteration = 0;

  for (;nonlinear_iteration < parameters.n_max_nonlinear_iterations;
       ++nonlinear_iteration)
  {
    reset_and_update_quadrature_point_history();

    assemble_residual();

    assemble_jacobian();

    solve_linearized_system();

    update_trial_solution(relaxation_parameter);

    reset_and_update_quadrature_point_history();

    assemble_residual();

    nonlinear_solver_logger.update_value("Iteration",
                                         nonlinear_iteration);
    nonlinear_solver_logger.update_value("L2-Norm(Newton update)",
                                         newton_update_norm);
    nonlinear_solver_logger.update_value("L2-Norm(Residual)",
                                         residual_norm);

    nonlinear_solver_logger.log_to_file();
    nonlinear_solver_logger.log_values_to_terminal();

    if (residual_norm < parameters.residual_tolerance ||
        newton_update_norm < parameters.newton_update_tolerance)
      break;
  }

  AssertThrow(nonlinear_iteration <
                parameters.n_max_nonlinear_iterations,
              dealii::ExcMessage(
                "The nonlinear solver has reach the given maximum "
                "number of iterations ("
                + std::to_string(parameters.n_max_nonlinear_iterations)
                + ")."));

  fe_field->solution = trial_solution;

  extrapolate_initial_trial_solution();

  print_decohesion_data();

  *pcout << std::endl;
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
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_newton_update.reinit(fe_field->distributed_vector);

  distributed_newton_update = newton_update;

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

  // Pass the distributed vectors to their ghosted counterpart
  newton_update = distributed_newton_update;

  // Compute the L2-Norm of the Newton update
  newton_update_norm = distributed_newton_update.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::update_trial_solution(
  const double relaxation_parameter)
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_trial_solution.reinit(fe_field->distributed_vector);
  distributed_newton_update.reinit(fe_field->distributed_vector);

  distributed_trial_solution  = trial_solution;
  distributed_newton_update   = newton_update;

  distributed_trial_solution.add(relaxation_parameter, distributed_newton_update);

  fe_field->get_affine_constraints().distribute(distributed_trial_solution);

  trial_solution  = distributed_trial_solution;
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::extrapolate_initial_trial_solution()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_old_solution;
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

  distributed_trial_solution.reinit(fe_field->distributed_vector);
  distributed_old_solution.reinit(fe_field->distributed_vector);
  distributed_newton_update.reinit(fe_field->distributed_vector);

  distributed_trial_solution  = fe_field->solution;
  distributed_old_solution    = fe_field->old_solution;
  distributed_newton_update   = fe_field->solution;

  distributed_trial_solution.sadd(2.0, -1.0, distributed_old_solution);
  distributed_newton_update.sadd(2.0, -2.0, distributed_old_solution);

  fe_field->get_affine_constraints().distribute(distributed_trial_solution);
  fe_field->get_newton_method_constraints().distribute(distributed_newton_update);

  trial_solution  = distributed_trial_solution;
  newton_update   = distributed_newton_update;
}


} // namespace gCP


template void gCP::GradientCrystalPlasticitySolver<2>::distribute_constraints_to_trial_solution();
template void gCP::GradientCrystalPlasticitySolver<3>::distribute_constraints_to_trial_solution();

template void gCP::GradientCrystalPlasticitySolver<2>::solve_nonlinear_system();
template void gCP::GradientCrystalPlasticitySolver<3>::solve_nonlinear_system();

template void gCP::GradientCrystalPlasticitySolver<2>::solve_linearized_system();
template void gCP::GradientCrystalPlasticitySolver<3>::solve_linearized_system();

template void gCP::GradientCrystalPlasticitySolver<2>::update_trial_solution(const double);
template void gCP::GradientCrystalPlasticitySolver<3>::update_trial_solution(const double);

template void gCP::GradientCrystalPlasticitySolver<2>::extrapolate_initial_trial_solution();
template void gCP::GradientCrystalPlasticitySolver<3>::extrapolate_initial_trial_solution();

