#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/numerics/data_out.h>

namespace gCP
{



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::extrapolate_initial_trial_solution()
  {
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_old_solution;

    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

    distributed_trial_solution.reinit(fe_field->distributed_vector);

    distributed_old_solution.reinit(fe_field->distributed_vector);

    distributed_newton_update.reinit(fe_field->distributed_vector);

    distributed_trial_solution  = fe_field->old_solution;

    distributed_old_solution    = fe_field->old_old_solution;

    distributed_newton_update   = fe_field->old_solution;

    bool flag_extrapolate_old_solutions = true;

    if (cyclic_step_data.loading_type ==
        RunTimeParameters::LoadingType::Cyclic)
    {
      const bool last_step_of_loading_phase =
          (discrete_time.get_step_number() + 1) ==
            cyclic_step_data.n_steps_in_loading_phase;

      const bool extrema_step_of_cyclic_phase =
          (discrete_time.get_step_number() + 1) >
            cyclic_step_data.n_steps_in_loading_phase
           &&
          ((discrete_time.get_step_number() + 1) -
              cyclic_step_data.n_steps_in_loading_phase) %
                cyclic_step_data.n_steps_per_half_cycle == 0;

      if ((last_step_of_loading_phase || extrema_step_of_cyclic_phase) &&
          parameters.flag_skip_extrapolation_at_extrema)
        flag_extrapolate_old_solutions = false;
    }

    double step_size_ratio = 1.0;

    if (discrete_time.get_step_number() > 0)
    {
      step_size_ratio =
        discrete_time.get_next_step_size() /
        discrete_time.get_previous_step_size();
    }

    //flag_extrapolate_old_solutions = false;

    if (flag_extrapolate_old_solutions)
    {
      distributed_trial_solution.sadd(
        1.0 + step_size_ratio,
        -step_size_ratio,
        distributed_old_solution);

      distributed_newton_update.sadd(
        -1.0,
        1.0,
        distributed_trial_solution);
    }

    fe_field->get_affine_constraints().distribute(
      distributed_trial_solution);

    fe_field->get_newton_method_constraints().distribute(
      distributed_newton_update);

    trial_solution  = distributed_trial_solution;

    newton_update   = distributed_newton_update;
  }



  template <int dim>
  std::tuple<bool, unsigned int> GradientCrystalPlasticitySolver<dim>::solve_nonlinear_system()
  {
    nonlinear_solver_logger.add_break(
      "Step " + std::to_string(discrete_time.get_step_number() + 1) +
      ": Solving for t = " + std::to_string(discrete_time.get_next_time()) +
      " with dt = " + std::to_string(discrete_time.get_next_step_size()));

    nonlinear_solver_logger.log_headers_to_terminal();

    extrapolate_initial_trial_solution();

    store_trial_solution(true);

    prepare_quadrature_point_history();

    bool flag_successful_convergence      = false;

    unsigned int nonlinear_iteration      = 0;

    //unsigned int regularization_iteration = 0;

    double previous_residual_norm         = 0.0;

    //double regularization_multiplier      = 1.0;

    const RunTimeParameters::NewtonRaphsonParameters
      &newton_parameters = parameters.newton_parameters;

    /*const RunTimeParameters::ConvergenceControlParameters
      &convergence_control_parameters =
        parameters.convergence_control_parameters;*/

    // Newton-Raphson loop
    do
    {
      nonlinear_iteration++;

      AssertThrow(
        nonlinear_iteration <= newton_parameters.n_max_iterations,
        dealii::ExcMessage(
          "The nonlinear solver has reach the given maximum number of "
          "iterations (" +
          std::to_string(newton_parameters.n_max_iterations) + ")."));

      if (nonlinear_iteration > newton_parameters.n_max_iterations)
      {
        reset_quadrature_point_history();

        return (std::make_tuple(flag_successful_convergence,
                                nonlinear_iteration));
      }

      // */

      /*
      // Check if the start value of the Newton-Raphson was inside the
      // desired convergence rate locus. If not, compute a new start
      // value using a lower regularization parameter.
      // An exception is made during the first load step, as the
      // displacement jumps seem to be the main cause of slow convergence
      // at this stage
      if (nonlinear_iteration > parameters.n_max_nonlinear_iterations)
      {
        bool flag_initial_guess_was_computed = false;

        std::stringstream message;


        message << "\n  Order of convergence is beneath the threshold"
                   ", i.e. ("
                << order_of_convergence << " < "
                << parameters.convergence_rate_threshold << "). "
                << "Computing a new initial guess with";
        //

        message << "\n  Maximum amount of nonlinear iterations reached. "
                << "Computing a new initial solution with";

        nonlinear_solver_logger.add_break(message.str().c_str());

        *pcout << message.rdbuf();

        do
        {
          regularization_iteration++;

          AssertThrow(
            regularization_iteration <=
                parameters.n_max_regularization_iterations,
            dealii::ExcMessage(
              "The maximum number of regularization loops (" +
              std::to_string(parameters.n_max_regularization_iterations) +
              ") have been reached"));

          regularization_multiplier *= parameters.regularization_factor;

          std::stringstream().swap(message);

          message << "\n  a regularization parameter of "
                  << (regularization_multiplier*
                      parameters.scalar_microscopic_stress_law_parameters.regularization_parameter)
                  << "...\n";

          *pcout << message.rdbuf();

          scalar_microscopic_stress_law->set_regularization_multiplier(
              regularization_multiplier);

          reset_trial_solution(true);

          flag_initial_guess_was_computed = compute_initial_guess();

        } while (!flag_initial_guess_was_computed);

        std::stringstream().swap(message);

        message << "\n  Initial guess computed. Restarting iteration...\n";

        nonlinear_solver_logger.add_break(message.str().c_str());

        *pcout << message.rdbuf();

        nonlinear_iteration = 0;

        regularization_multiplier = 1.0;

        scalar_microscopic_stress_law->set_regularization_multiplier(
          regularization_multiplier);

        continue;
      }
      */
      // The current trial solution has to be stored in case
      store_trial_solution();

      reset_and_update_quadrature_point_history();

      const double initial_value_scalar_function = assemble_residual();

      assemble_jacobian();

      const unsigned int n_krylov_iterations = solve_linearized_system();

      double relaxation_parameter = 1.0;

      update_trial_solution(relaxation_parameter);

      reset_and_update_quadrature_point_history();

      // Line search algorithm
      {
        double trial_value_scalar_function = assemble_residual();

        line_search.reinit(initial_value_scalar_function);

        while (!line_search.suficient_descent_condition(
            trial_value_scalar_function, relaxation_parameter))
        {
          relaxation_parameter =
              line_search.get_lambda(trial_value_scalar_function, relaxation_parameter);

          reset_trial_solution();

          update_trial_solution(relaxation_parameter);

          reset_and_update_quadrature_point_history();

          trial_value_scalar_function = assemble_residual();
        }
      }

      const auto residual_l2_norms =
          fe_field->get_l2_norms(residual);

      const auto newton_update_l2_norms =
          fe_field->get_l2_norms(newton_update);

      const double order_of_convergence =
          (nonlinear_iteration != 1) ? std::log(residual_norm) / std::log(previous_residual_norm) : 0.0;

      previous_residual_norm = residual_norm;

      nonlinear_solver_logger.update_value("N-Itr",
                                           nonlinear_iteration);
      nonlinear_solver_logger.update_value("K-Itr",
                                           n_krylov_iterations);
      nonlinear_solver_logger.update_value("L-Itr",
                                           line_search.get_n_iterations());
      nonlinear_solver_logger.update_value("(NS)_L2",
                                           std::get<0>(newton_update_l2_norms));
      nonlinear_solver_logger.update_value("(NS_U)_L2",
                                           std::get<1>(newton_update_l2_norms));
      nonlinear_solver_logger.update_value("(NS_G)_L2",
                                           std::get<2>(newton_update_l2_norms));
      nonlinear_solver_logger.update_value("(R)_L2",
                                           std::get<0>(residual_l2_norms));
      nonlinear_solver_logger.update_value("(R_U)_L2",
                                           std::get<1>(residual_l2_norms));
      nonlinear_solver_logger.update_value("(R_G)_L2",
                                           std::get<2>(residual_l2_norms));
      nonlinear_solver_logger.update_value("C-Rate",
                                           order_of_convergence);

      nonlinear_solver_logger.log_to_file();
      nonlinear_solver_logger.log_values_to_terminal();

      flag_successful_convergence =
          residual_norm < newton_parameters.absolute_tolerance ||
          newton_update_norm < newton_parameters.step_tolerance;

    } while (!flag_successful_convergence);

    store_effective_opening_displacement_in_quadrature_history();

    fe_field->solution = trial_solution;

    extrapolate_initial_trial_solution();

    print_decohesion_data();

    *pcout << std::endl;

    return (std::make_tuple(flag_successful_convergence, nonlinear_iteration));
  }



  template <int dim>
  unsigned int GradientCrystalPlasticitySolver<dim>::solve_linearized_system()
  {
    if (parameters.verbose)
      *pcout << std::setw(38) << std::left
             << "  Solver: Solving linearized system...";

    dealii::TimerOutput::Scope t(*timer_output, "Solver: Solve ");

    // In this method we create temporal non ghosted copies
    // of the pertinent vectors to be able to perform the solve()
    // operation.
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

    distributed_newton_update.reinit(fe_field->distributed_vector);

    distributed_newton_update = newton_update;

    const RunTimeParameters::KrylovParameters &krylov_parameters =
      parameters.krylov_parameters;

    // The solver's tolerances are passed to the SolverControl instance
    // used to initialize the solver
    dealii::SolverControl solver_control(
        krylov_parameters.n_max_iterations,
        std::max(residual_norm * krylov_parameters.relative_tolerance,
                 krylov_parameters.absolute_tolerance));

    switch (krylov_parameters.solver_type)
    {
    case RunTimeParameters::SolverType::DirectSolver:
    {
      dealii::TrilinosWrappers::SolverDirect solver(solver_control);

      try
      {
        solver.solve(jacobian, distributed_newton_update, residual);
      }
      catch (std::exception &exc)
      {
        std::cerr << std::endl
                  << std::endl
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
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception in the solve method!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
    }
    break;
    case RunTimeParameters::SolverType::CG:
    {
      dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

      dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;

      preconditioner.initialize(jacobian);

      try
      {
        solver.solve(jacobian,
                     distributed_newton_update,
                     residual,
                     preconditioner);
      }
      catch (std::exception &exc)
      {
        std::cerr << std::endl
                  << std::endl
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
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception in the solve method!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
    }
    break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
      break;
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

    return (solver_control.last_step());
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

    trial_solution = distributed_trial_solution;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::store_trial_solution(
    const bool flag_store_initial_trial_solution)
  {
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

    distributed_trial_solution.reinit(fe_field->distributed_vector);

    distributed_trial_solution = trial_solution;

    fe_field->get_affine_constraints().distribute(distributed_trial_solution);

    if (flag_store_initial_trial_solution)
    {
      initial_trial_solution  = distributed_trial_solution;
    }
    else
    {
      tmp_trial_solution      = distributed_trial_solution;
    }
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::reset_trial_solution(
    const bool flag_reset_to_initial_trial_solution)
  {
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

    distributed_trial_solution.reinit(fe_field->distributed_vector);

    if (flag_reset_to_initial_trial_solution)
    {
      distributed_trial_solution = initial_trial_solution;
    }
    else
    {
      distributed_trial_solution = tmp_trial_solution;
    }

    fe_field->get_affine_constraints().distribute(distributed_trial_solution);

    trial_solution = distributed_trial_solution;
  }


  /*
  template <int dim>
  bool GradientCrystalPlasticitySolver<dim>::compute_initial_guess()
  {
    bool flag_successful_convergence = false;

    unsigned int nonlinear_iteration = 0;

    double previous_residual_norm = 0.0;

    // Newton-Raphson loop
    do
    {
      nonlinear_iteration++;

      AssertThrow(
          nonlinear_iteration <= parameters.n_max_nonlinear_iterations,
          dealii::ExcMessage("The nonlinear solver has reach the given "
                             "maximum number of iterations (" +
                             std::to_string(parameters.n_max_nonlinear_iterations) + ")."));


      // Check if the start value of the Newton-Raphson was inside the
      // desired convergence rate locus. If not, compute a new start
      // value using a lower regularization parameter.
      if (nonlinear_iteration>parameters.n_max_nonlinear_iterations)
      {
        std::stringstream message;

        message << "\n  Order of convergence is beneath the threshold"
                   ", i.e. ("
                << order_of_convergence << " < "
                << parameters.convergence_rate_threshold << "). "
                << "Computing a new initial guess with";

        message << "\n  Maximum amount of nonlinear iterations reached. "
                << "Computing a new initial solution with";

        nonlinear_solver_logger.add_break(message.str().c_str());

        *pcout << message.rdbuf();

        return (false);
      }

      // The current trial solution has to be stored in case
      store_trial_solution();

      reset_and_update_quadrature_point_history();

      const double initial_value_scalar_function = assemble_residual();

      assemble_jacobian();

      const unsigned int n_krylov_iterations = solve_linearized_system();

      double relaxation_parameter = 1.0;

      update_trial_solution(relaxation_parameter);

      reset_and_update_quadrature_point_history();

      // Line search algorithm
      {
        double trial_value_scalar_function = assemble_residual();

        line_search.reinit(initial_value_scalar_function);

        while (!line_search.suficient_descent_condition(
            trial_value_scalar_function, relaxation_parameter))
        {
          relaxation_parameter =
              line_search.get_lambda(trial_value_scalar_function, relaxation_parameter);

          reset_trial_solution();

          update_trial_solution(relaxation_parameter);

          reset_and_update_quadrature_point_history();

          trial_value_scalar_function = assemble_residual();
        }
      }

      const auto residual_l2_norms =
          fe_field->get_l2_norms(residual);

      const auto newton_update_l2_norms =
          fe_field->get_l2_norms(newton_update);

      const double order_of_convergence =
          (nonlinear_iteration != 1) ? std::log(residual_norm) / std::log(previous_residual_norm) : 0.0;

      previous_residual_norm = residual_norm;

      nonlinear_solver_logger.update_value("N-Itr",
                                           nonlinear_iteration);
      nonlinear_solver_logger.update_value("K-Itr",
                                           n_krylov_iterations);
      nonlinear_solver_logger.update_value("L-Itr",
                                           line_search.get_n_iterations());
      nonlinear_solver_logger.update_value("(NS)_L2",
                                           std::get<0>(newton_update_l2_norms));
      nonlinear_solver_logger.update_value("(NS_U)_L2",
                                           std::get<1>(newton_update_l2_norms));
      nonlinear_solver_logger.update_value("(NS_G)_L2",
                                           std::get<2>(newton_update_l2_norms));
      nonlinear_solver_logger.update_value("(R)_L2",
                                           std::get<0>(residual_l2_norms));
      nonlinear_solver_logger.update_value("(R_U)_L2",
                                           std::get<1>(residual_l2_norms));
      nonlinear_solver_logger.update_value("(R_G)_L2",
                                           std::get<2>(residual_l2_norms));
      nonlinear_solver_logger.update_value("C-Rate",
                                           order_of_convergence);

      nonlinear_solver_logger.log_to_file();
      nonlinear_solver_logger.log_values_to_terminal();

      flag_successful_convergence =
          residual_norm < parameters.residual_tolerance ||
          newton_update_norm < parameters.newton_update_tolerance;

    } while (!flag_successful_convergence);

    return (true);
  }
  */


} // namespace gCP



template void gCP::GradientCrystalPlasticitySolver<2>::extrapolate_initial_trial_solution();
template void gCP::GradientCrystalPlasticitySolver<3>::extrapolate_initial_trial_solution();

template std::tuple<bool,unsigned int> gCP::GradientCrystalPlasticitySolver<2>::solve_nonlinear_system();
template std::tuple<bool,unsigned int> gCP::GradientCrystalPlasticitySolver<3>::solve_nonlinear_system();

template unsigned int gCP::GradientCrystalPlasticitySolver<2>::solve_linearized_system();
template unsigned int gCP::GradientCrystalPlasticitySolver<3>::solve_linearized_system();

template void gCP::GradientCrystalPlasticitySolver<2>::update_trial_solution(const double);
template void gCP::GradientCrystalPlasticitySolver<3>::update_trial_solution(const double);
