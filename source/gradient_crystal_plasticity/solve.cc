#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/numerics/data_out.h>

namespace gCP
{



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  extrapolate_initial_trial_solution(const bool flag_skip_extrapolation)
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

    double step_size_ratio = 1.0;

    if (discrete_time.get_step_number() > 0)
    {
      step_size_ratio =
        discrete_time.get_next_step_size() /
        discrete_time.get_previous_step_size();
    }

    if (!flag_skip_extrapolation)
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

    //fe_field->get_affine_constraints().distribute(
    fe_field->get_hanging_node_constraints().distribute(
      distributed_trial_solution);

    fe_field->get_newton_method_constraints().distribute(
      distributed_newton_update);

    trial_solution  = distributed_trial_solution;

    newton_update   = distributed_newton_update;
  }



  template <int dim>
  std::tuple<bool, unsigned int> GradientCrystalPlasticitySolver<dim>::
  solve_nonlinear_system(const bool flag_skip_extrapolation)
  {
    // Terminal and log output
    nonlinear_solver_logger.add_break(
      "Step " + std::to_string(discrete_time.get_step_number() + 1) +
      ": Solving for t = " + std::to_string(discrete_time.get_next_time()) +
      " with dt = " + std::to_string(discrete_time.get_next_step_size()));

    nonlinear_solver_logger.log_headers_to_terminal();

    // Initialize local variables
    bool flag_successful_convergence  = false;

    bool flag_compute_active_set      = true;

    unsigned int nonlinear_iteration  = 0;

    double old_residual_norm          = 0.0;

    std::vector<double> residual_l2_norms;

    residual_l2_norms.reserve(3);

    const RunTimeParameters::NewtonRaphsonParameters
      &newton_parameters = parameters.newton_parameters;

    // Internal variables' values at the previous step are stored
    prepare_quadrature_point_history();

    // Compute and store starting point
    extrapolate_initial_trial_solution(flag_skip_extrapolation);

    store_trial_solution(true);

    // Active set
    reset_internal_newton_method_constraints();

    // Newton-Raphson loop
    do
    {
      // Increment iteration count
      nonlinear_iteration++;

      AssertThrow(
        nonlinear_iteration <= newton_parameters.n_max_iterations,
        dealii::ExcMessage(
          "The nonlinear solver has reach the given maximum number of "
          "iterations (" +
          std::to_string(newton_parameters.n_max_iterations) + ")."));

      if (parameters.constitutive_laws_parameters.
            scalar_microstress_law_parameters.flag_rate_independent)
      {
        if (flag_compute_active_set)
        {
          determine_active_set();

          determine_inactive_set();

          reset_inactive_set_values();

          flag_compute_active_set = false;
        }
      }
      else
      {
        active_set = plastic_slip_dofs_set;
      }

      debug_output();

      // Constraints distribution (Done later to obtain a continous
      // start value for the active set determination) Temporary code
      {
        dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

        distributed_trial_solution.reinit(fe_field->distributed_vector);

        distributed_trial_solution = trial_solution;

        fe_field->get_affine_constraints().distribute(
          distributed_trial_solution);

        trial_solution = distributed_trial_solution;
      }

      // Preparations for the Newton-Update and Line-Search
      {
        store_trial_solution();

        reset_and_update_quadrature_point_history();
      }

      // Assemble linear system
      assemble_residual();

      assemble_jacobian();

      // Residuals
      residual_l2_norms = compute_residual_l2_norms();

      const double initial_objective_function_value =
        residual_l2_norms[0];

      const std::vector<double> initial_objective_function_values =
        LineSearch::get_objective_function_values(
          std::vector<double>(
            residual_l2_norms.begin() + 1,
            residual_l2_norms.end()));

      old_residual_norm = residual_norm;

      // Terminal and log output
      if (nonlinear_iteration == 1)
      {
        const auto residual_l2_norms =
            fe_field->get_l2_norms(residual);

        update_and_output_nonlinear_solver_logger(
          residual_l2_norms);
      }

      // Newton-Update
      const unsigned int n_krylov_iterations =
        solve_linearized_system();

      double relaxation_parameter =
        newton_parameters.relaxation_parameter;


      std::vector<double> relaxation_parameters(
        2,
        newton_parameters.relaxation_parameter);

      update_trial_solution(relaxation_parameter);

      reset_and_update_quadrature_point_history();

      // Compute residuals for convergence-check
      // (and possibly Line-Search)
      assemble_residual();

      residual_l2_norms = compute_residual_l2_norms();

      double objective_function_value =
        LineSearch::get_objective_function_value(
          residual_l2_norms[0]);

      std::vector<double> objective_function_values =
        LineSearch::get_objective_function_values(
          std::vector<double>(
            residual_l2_norms.begin() + 1,
            residual_l2_norms.end()));

      // Line search algorithm
      if (newton_parameters.flag_line_search)
      {
        line_search.reinit(
          initial_objective_function_value,
          discrete_time.get_step_number() + 1,
          nonlinear_iteration);

        while (!line_search.suficient_descent_condition(
            objective_function_value, relaxation_parameter))
        {
          relaxation_parameter =
              line_search.get_lambda(objective_function_value,
                                     relaxation_parameter);

          reset_trial_solution();

          update_trial_solution(relaxation_parameter);

          reset_and_update_quadrature_point_history();

          objective_function_value = assemble_residual();
        }

        /*
        line_search.reinit(
          initial_objective_function_values,
          discrete_time.get_step_number() + 1,
          nonlinear_iteration);

        while (!line_search.suficient_descent_condition(
                  objective_function_values,
                  relaxation_parameters))
        {
          std::cout << "Before: "
                    << relaxation_parameters[0] << ", "
                    << relaxation_parameters[1] << std::endl;

          relaxation_parameters =
              line_search.get_lambdas(
                objective_function_values,
                relaxation_parameters);

          std::cout << "After: "
                    << relaxation_parameters[0] << ", "
                    << relaxation_parameters[1] << std::endl;

          reset_trial_solution();

          update_trial_solution(std::min(relaxation_parameters[0],
                                         relaxation_parameters[1]));

          reset_and_update_quadrature_point_history();

          assemble_residual();

          residual_l2_norms =
            compute_residual_l2_norms();

          objective_function_values =
            LineSearch::get_objective_function_values(
              std::vector<double>(
                residual_l2_norms.begin() + 1,
                residual_l2_norms.end()));
        } */
      }

      // Terminal and log output
      {
        const auto newton_update_l2_norms =
          fe_field->get_l2_norms(newton_update);

        const auto residual_l2_norms =
          fe_field->get_l2_norms(residual);

        const double order_of_convergence =
          std::log(residual_norm) /std::log(old_residual_norm);

        update_and_output_nonlinear_solver_logger(
          nonlinear_iteration,
          n_krylov_iterations,
          line_search.get_n_iterations(),
          newton_update_l2_norms,
          residual_l2_norms,
          order_of_convergence,
          relaxation_parameter);
      }

      residual_l2_norms = compute_residual_l2_norms();

      //slip_rate_output(false);

      flag_successful_convergence =
          //residual_norm < newton_parameters.absolute_tolerance;
          residual_l2_norms[0] < newton_parameters.absolute_tolerance;

    if (flag_successful_convergence &&
        parameters.constitutive_laws_parameters.
            scalar_microstress_law_parameters.flag_rate_independent)
    {
      const dealii::IndexSet original_active_set = active_set;

      // Active set
      reset_internal_newton_method_constraints();

      if (parameters.constitutive_laws_parameters.
            scalar_microstress_law_parameters.flag_rate_independent)
      {
        if (nonlinear_iteration == 1)
        {
          determine_active_set();

          determine_inactive_set();

          reset_inactive_set_values();
        }
      }
      else
      {
        active_set = plastic_slip_dofs_set;
      }

      if (original_active_set != active_set)
      {
        flag_successful_convergence = false;

        nonlinear_iteration = 0;

        reset_trial_solution(true);

        *pcout
          << std::endl
          << "Active set mismatch. Restarting Newton-loop with the "
          << "new active set..." << std::endl << std::endl;
      }
    }

    } while (!flag_successful_convergence);

    store_effective_opening_displacement_in_quadrature_history();

    fe_field->solution = trial_solution;

    *pcout << std::endl;

    return (std::make_tuple(
              flag_successful_convergence, nonlinear_iteration));
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

        dealii::LinearAlgebraTrilinos::MPI::PreconditionILU::AdditionalData
          additional_data;

        dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;

        preconditioner.initialize(jacobian, additional_data);

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

    case RunTimeParameters::SolverType::GMRES:
      {
        dealii::LinearAlgebraTrilinos::SolverGMRES solver(solver_control);

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
    //fe_field->get_newton_method_constraints()
    internal_newton_method_constraints.distribute(
      distributed_newton_update);

    // Pass the distributed vectors to their ghosted counterpart
    newton_update = distributed_newton_update;

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
  void GradientCrystalPlasticitySolver<dim>::update_trial_solution(
      const std::vector<double> relaxation_parameters)
  {
    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

    dealii::LinearAlgebraTrilinos::MPI::Vector distributed_newton_update;

    distributed_trial_solution.reinit(fe_field->distributed_vector);

    distributed_newton_update.reinit(fe_field->distributed_vector);

    distributed_trial_solution  = trial_solution;

    distributed_newton_update   = newton_update;

    for (const auto &locally_owned_dof :
          fe_field->get_locally_owned_dofs())
    {
      if (displacement_dofs_set.is_element(locally_owned_dof))
      {
        distributed_trial_solution(locally_owned_dof) +=
          relaxation_parameters[0] *
          distributed_newton_update(locally_owned_dof);
      }
      else if (plastic_slip_dofs_set.is_element(locally_owned_dof))
      {
        distributed_trial_solution(locally_owned_dof) +=
          relaxation_parameters[1] *
          distributed_newton_update(locally_owned_dof);
      }
      else
      {
        dealii::ExcInternalError();
      }
    }

    fe_field->get_affine_constraints().distribute(
      distributed_trial_solution);

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
      distributed_trial_solution = fe_field->old_solution; // initial_trial_solution
    }
    else
    {
      distributed_trial_solution = tmp_trial_solution;
    }

    fe_field->get_affine_constraints().distribute(distributed_trial_solution);

    trial_solution = distributed_trial_solution;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  update_and_output_nonlinear_solver_logger(
    const std::tuple<double, double, double>  residual_l2_norms)
  {
    nonlinear_solver_logger.update_value("N-Itr",
                                        0);
    nonlinear_solver_logger.update_value("K-Itr",
                                        0);
    nonlinear_solver_logger.update_value("L-Itr",
                                        0);
    nonlinear_solver_logger.update_value("(NS)_L2",
                                        0);
    nonlinear_solver_logger.update_value("(NS_U)_L2",
                                        0);
    nonlinear_solver_logger.update_value("(NS_G)_L2",
                                        0);
    nonlinear_solver_logger.update_value("(R)_L2",
                                        std::get<0>(residual_l2_norms));
    nonlinear_solver_logger.update_value("(R_U)_L2",
                                        std::get<1>(residual_l2_norms));
    nonlinear_solver_logger.update_value("(R_G)_L2",
                                        std::get<2>(residual_l2_norms));
    nonlinear_solver_logger.update_value("C-Rate",
                                        0);

    nonlinear_solver_logger.log_to_file();

    nonlinear_solver_logger.log_values_to_terminal();
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  update_and_output_nonlinear_solver_logger(
    const unsigned int                        nonlinear_iteration,
    const unsigned int                        n_krylov_iterations,
    const unsigned int                        n_line_search_iterations,
    const std::tuple<double, double, double>  newton_update_l2_norms,
    const std::tuple<double, double, double>  residual_l2_norms,
    const double                              order_of_convergence,
    const double                              relaxation_parameter)
  {
    nonlinear_solver_logger.update_value("N-Itr",
                                        nonlinear_iteration);
    nonlinear_solver_logger.update_value("K-Itr",
                                        n_krylov_iterations);
    nonlinear_solver_logger.update_value("L-Itr",
                                        n_line_search_iterations);
    nonlinear_solver_logger.update_value("(NS)_L2",
                                        relaxation_parameter *
                                        std::get<0>(newton_update_l2_norms));
    nonlinear_solver_logger.update_value("(NS_U)_L2",
                                        relaxation_parameter *
                                        std::get<1>(newton_update_l2_norms));
    nonlinear_solver_logger.update_value("(NS_G)_L2",
                                        relaxation_parameter *
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
  }



  template <int dim>
  std::vector<double> GradientCrystalPlasticitySolver<dim>::
  compute_residual_l2_norms()
  {
    dealii::LinearAlgebraTrilinos::MPI::Vector  distributed_residual;

    distributed_residual.reinit(fe_field->distributed_vector);

    distributed_residual = residual;

    double vector_squared_entries = 0.0;

    double scalar_squared_entries = 0.0;

    for (auto const &locally_owned_dof: displacement_dofs_set)
    {
      vector_squared_entries +=
        distributed_residual[locally_owned_dof] *
        distributed_residual[locally_owned_dof];
    }

    for (auto const &locally_owned_dof: active_set)
    {
      scalar_squared_entries +=
        distributed_residual[locally_owned_dof] *
        distributed_residual[locally_owned_dof];
    }

    vector_squared_entries =
      dealii::Utilities::MPI::sum(vector_squared_entries,
                                  MPI_COMM_WORLD);

    scalar_squared_entries =
      dealii::Utilities::MPI::sum(scalar_squared_entries,
                                  MPI_COMM_WORLD);

    const double residual_l2_norm =
      std::sqrt(vector_squared_entries + scalar_squared_entries);

    const double linear_momentum_balance_residual_l2_norm =
      std::sqrt(vector_squared_entries);

    const double pseudo_balance_residual_l2_norm =
      std::sqrt(scalar_squared_entries);

    std::vector<double> residual_l2_norms(3);

    residual_l2_norms[0] = residual_l2_norm;

    residual_l2_norms[1] = linear_momentum_balance_residual_l2_norm;

    residual_l2_norms[2] = pseudo_balance_residual_l2_norm;

    return (residual_l2_norms);
  }



} // namespace gCP



template void gCP::GradientCrystalPlasticitySolver<2>::extrapolate_initial_trial_solution(const bool);
template void gCP::GradientCrystalPlasticitySolver<3>::extrapolate_initial_trial_solution(const bool);

template std::tuple<bool,unsigned int> gCP::GradientCrystalPlasticitySolver<2>::solve_nonlinear_system(const bool);
template std::tuple<bool,unsigned int> gCP::GradientCrystalPlasticitySolver<3>::solve_nonlinear_system(const bool);

template unsigned int gCP::GradientCrystalPlasticitySolver<2>::solve_linearized_system();
template unsigned int gCP::GradientCrystalPlasticitySolver<3>::solve_linearized_system();

template void gCP::GradientCrystalPlasticitySolver<2>::update_trial_solution(const double);
template void gCP::GradientCrystalPlasticitySolver<3>::update_trial_solution(const double);

template void gCP::GradientCrystalPlasticitySolver<2>::
  update_trial_solution(const std::vector<double>);
template void gCP::GradientCrystalPlasticitySolver<3>::
  update_trial_solution(const std::vector<double>);
