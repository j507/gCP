#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/numerics/data_out.h>

namespace gCP
{



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  extrapolate_initial_trial_solution(const bool flag_skip_extrapolation)
  {
    double step_size_ratio = 1.0;

    if (discrete_time.get_step_number() > 0)
    {
      step_size_ratio =
        discrete_time.get_next_step_size() /
        discrete_time.get_previous_step_size();
    }

    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_trial_solution =
        fe_field->get_distributed_vector_instance(
          fe_field->old_solution);

    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_old_solution =
        fe_field->get_distributed_vector_instance(
          fe_field->old_old_solution);;

    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_newton_update =
        fe_field->get_distributed_vector_instance(
          fe_field->old_solution);;

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

    trial_solution = distributed_trial_solution;

    newton_update = distributed_newton_update;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  distribute_affine_constraints_to_trial_solution()
  {
    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_trial_solution =
        fe_field->get_distributed_vector_instance(trial_solution);

    fe_field->get_affine_constraints().distribute(
      distributed_trial_solution);

    trial_solution = distributed_trial_solution;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  solve_nonlinear_system(const bool flag_skip_extrapolation)
  {
    // Terminal and log output
    nonlinear_solver_logger.add_break(
      "Step " + std::to_string(discrete_time.get_step_number() + 1) +
      ": Solving for t = " +
      std::to_string(discrete_time.get_next_time()) +
      " with dt = " +
      std::to_string(discrete_time.get_next_step_size()));

    nonlinear_solver_logger.log_headers_to_terminal();

    // Internal variables' values at the previous step are stored
    prepare_quadrature_point_history();

    // Compute and store starting point
    extrapolate_initial_trial_solution(flag_skip_extrapolation);

    store_trial_solution(true);

    // Active set
    reset_internal_newton_method_constraints();

    // Solution algorithm
    switch (parameters.solution_algorithm)
    {
      case RunTimeParameters::SolutionAlgorithm::Monolithic:
      {
        monolithic_algorithm();
      }
      break;

      case RunTimeParameters::SolutionAlgorithm::Bouncing:
      {
        bouncing_algorithm();
      }
      break;

      case RunTimeParameters::SolutionAlgorithm::Embracing:
      {
        embracing_algorihtm();
      }
      break;

      default:
      {
        Assert(false, dealii::ExcMessage(
          "Unexpected identifier for the solution algorithm."));
      }
      break;
    }

    // Compute and store the opening displacement at the quadrature
    // points based on the converged solution
    store_effective_opening_displacement_in_quadrature_history();

    // Update the actual solution vector
    fe_field->solution = trial_solution;

    *pcout << std::endl;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::monolithic_algorithm()
  {
    // Instance local variables and references
    unsigned int nonlinear_iteration = 0;

    bool flag_successful_convergence = false,
         flag_compute_active_set = true;

    dealii::Vector<double> residual_l2_norms, old_residual_l2_norms;

    line_search =
      std::make_unique<gCP::LineSearch>(
        parameters.monolithic_algorithm_parameters.
          monolithic_system_solver_parameters.line_search_parameters);

    const RunTimeParameters::NewtonRaphsonParameters
      &newton_parameters = parameters.monolithic_algorithm_parameters.
        monolithic_system_solver_parameters.newton_parameters;

    const RunTimeParameters::KrylovParameters
      &krylov_parameters = parameters.monolithic_algorithm_parameters.
        monolithic_system_solver_parameters.krylov_parameters;

    // Newton-Raphson loop
    do
    {
      // Increase iteration counter
      nonlinear_iteration++;

      AssertThrow(
        nonlinear_iteration <=
          newton_parameters.n_max_iterations,
        dealii::ExcMessage(
          "The nonlinear solver has reach the given maximum number of "
          "iterations (" +
          std::to_string(
            newton_parameters.n_max_iterations) + ")."));

      // Determine the active (and also inactive) set
      active_set_algorithm(flag_compute_active_set);

      // Output for debugging purposes
      debug_output();

      // Constraints distribution (Done later to obtain a continous
      // start value for the active set determination) Temporary code
      distribute_affine_constraints_to_trial_solution();

      // Preparations for the Newton-Update and Line-Search
      store_trial_solution();

      reset_and_update_quadrature_point_history();

      // Assemble linear system
      assemble_jacobian();

      assemble_residual();

      // Store current l2-norm values and initial objective
      // function
      residual_l2_norms = fe_field->get_sub_l2_norms(residual);

      old_residual_l2_norms = residual_l2_norms;

      line_search->reinit(
        LineSearch::get_objective_function_value(
          residual_l2_norms.l2_norm()),
        discrete_time.get_step_number() + 1,
        nonlinear_iteration);

      // Terminal and log output
      if (nonlinear_iteration == 1)
      {
        update_and_output_nonlinear_solver_logger(
          residual_l2_norms);
      }

      // Newton-Raphson update
      const unsigned int n_krylov_iterations =
        solve_linearized_system(
          krylov_parameters,
          0,
          residual_l2_norms.l2_norm());

      double relaxation_parameter = 1.0;

      update_trial_solution(relaxation_parameter);

      reset_and_update_quadrature_point_history();

      // Compute residuals for convergence-check
      // (and possibly Line-Search)
      assemble_residual();

      residual_l2_norms = fe_field->get_sub_l2_norms(residual);

      // Line search algorithm
      if (newton_parameters.flag_line_search)
      {
        relaxation_parameter = line_search_algorithm();

        residual_l2_norms = fe_field->get_sub_l2_norms(residual);
      }

      // Terminal and log output
      {
        const dealii::Vector<double> newton_update_l2_norms =
          fe_field->get_sub_l2_norms(newton_update);

        const double order_of_convergence =
          std::log(residual_l2_norms.l2_norm()) /
            std::log(old_residual_l2_norms.l2_norm());

        update_and_output_nonlinear_solver_logger(
          nonlinear_iteration,
          n_krylov_iterations,
          line_search->get_n_iterations(),
          newton_update_l2_norms,
          residual_l2_norms,
          order_of_convergence,
          relaxation_parameter);
      }

      //slip_rate_output(false);

      // Convergence check
      flag_successful_convergence =
        residual_l2_norms.l2_norm() <
          newton_parameters.absolute_tolerance;

      // Check if the active set changed
      if (flag_successful_convergence &&
          parameters.constitutive_laws_parameters.
              scalar_microstress_law_parameters.flag_rate_independent)
      {
        const dealii::IndexSet original_active_set =
          locally_owned_active_set;

        flag_compute_active_set = true;

        active_set_algorithm(flag_compute_active_set);

        if (original_active_set != locally_owned_active_set)
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
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::bouncing_algorithm()
  {
    // Declare and initilize local variables and references
    unsigned int macro_nonlinear_iteration = 0,
                 micro_nonlinear_iteration = 0;

    bool flag_successful_convergence = false,
         flag_successful_macro_convergence = false,
         flag_successful_micro_convergence = false,
         flag_compute_active_set = true;

    dealii::Vector<double> residual_l2_norms, old_residual_l2_norms;

    line_search =
      std::make_unique<gCP::LineSearch>(
        parameters.staggered_algorithm_parameters.
          linear_momentum_solver_parameters.line_search_parameters);

    const RunTimeParameters::NewtonRaphsonParameters
      &macro_newton_parameters = parameters.
        staggered_algorithm_parameters.
          linear_momentum_solver_parameters.newton_parameters,
      &micro_newton_parameters = parameters.
        staggered_algorithm_parameters.pseudo_balance_solver_parameters.
          newton_parameters;

    const RunTimeParameters::KrylovParameters
      &macro_krylov_parameters = parameters.
        staggered_algorithm_parameters.
          linear_momentum_solver_parameters.krylov_parameters,
      &micro_krylov_parameters = parameters.
        staggered_algorithm_parameters.pseudo_balance_solver_parameters.
          krylov_parameters;

    // The distribution corresponds to another method for the sake of
    // the monolithic algorithm
    distribute_affine_constraints_to_trial_solution();

    // Macro-Newton-Raphson loop
    do
    {
      // Convergence check
      residual_l2_norms = fe_field->get_sub_l2_norms(residual);

      flag_successful_macro_convergence =
        residual_l2_norms[0] <
          macro_newton_parameters.absolute_tolerance;

      if (flag_successful_macro_convergence)
      {
        // If the linear momentum balance converges with the plastic
        // slips values of the lattest converged micro Newton-Raphson
        // loop, the solution is considered as found.
        if (flag_successful_micro_convergence)
        {
          flag_successful_convergence = true;

          continue;
        }
        // Otherwise compute a new trial solution for the plastic slips
        else
        {
          // Determine the active (and also inactive) set
          active_set_algorithm(flag_compute_active_set);

          // Micro-Newton-Raphson loop
          do
          {
            // Increase iteration counter
            micro_nonlinear_iteration++;

            // Preparations for the Newton-Update and Line-Search
            store_trial_solution();

            reset_and_update_quadrature_point_history();

            // Assemble linear system;
            assemble_jacobian();

            assemble_residual();

            // Store current l2-norm values and initial objective
            // function
            residual_l2_norms = fe_field->get_sub_l2_norms(residual);

            old_residual_l2_norms = residual_l2_norms;

            line_search->reinit(
              LineSearch::get_objective_function_value(
                residual_l2_norms[1]),
              discrete_time.get_step_number() + 1,
              micro_nonlinear_iteration);

            // Terminal and log output
            {

            }

            // Newton-Raphson update
            const unsigned int n_krylov_iterations =
              solve_linearized_system(
                micro_krylov_parameters,
                1,
                residual_l2_norms[1]);

            double relaxation_parameter = 1.0;

            update_trial_solution(relaxation_parameter, 1);

            reset_and_update_quadrature_point_history();

            // Evaluate new trial solution
            assemble_residual();

            residual_l2_norms =
              fe_field->get_sub_l2_norms(residual);

            // Line search
            if (micro_newton_parameters.flag_line_search)
            {

            }

            // Terminal and log output
            {
              (void)n_krylov_iterations;
            }

            // Convergence check
            flag_successful_micro_convergence =
              residual_l2_norms[1] <
                micro_newton_parameters.absolute_tolerance;

          } while (!flag_successful_micro_convergence);
        }
      }
      else
      {
        // Increase iteration counter
        macro_nonlinear_iteration++;

        // Preparations for the Newton-Update and Line-Search
        store_trial_solution();

        reset_and_update_quadrature_point_history();

        // Assemble linear system
        if (macro_nonlinear_iteration == 1)
        {
          // The submatrix at (0,0) is a constant, therefore it only
          // needs to be assembled once
          assemble_jacobian();
        }

        assemble_residual();

        // Store current l2-norm values and initial objective function
        residual_l2_norms = fe_field->get_sub_l2_norms(residual);

        old_residual_l2_norms = residual_l2_norms;

        line_search->reinit(
          LineSearch::get_objective_function_value(
            residual_l2_norms[0]),
          discrete_time.get_step_number() + 1,
          macro_nonlinear_iteration);

        // Terminal and log output ()
        {

        }

        // Newton-Raphson update
        const unsigned int n_krylov_iterations =
          solve_linearized_system(
            macro_krylov_parameters,
            0,
            residual_l2_norms[0]);

        double relaxation_parameter = 1.0;

        update_trial_solution(relaxation_parameter, 0);

        reset_and_update_quadrature_point_history();

        // Evaluate new trial solution
        assemble_residual();

        residual_l2_norms =
          fe_field->get_sub_l2_norms(residual);

        // Line search algorithm
        if (macro_newton_parameters.flag_line_search)
        {

        }

        // Terminal and log output
        {
          (void)n_krylov_iterations;
        }

        // Convergence check

        flag_successful_macro_convergence =
          residual_l2_norms[0] <
            macro_newton_parameters.absolute_tolerance;

        flag_successful_micro_convergence = false;
      }
    } while (!flag_successful_convergence);
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::embracing_algorihtm()
  {

  }



  template <int dim>
  unsigned int GradientCrystalPlasticitySolver<dim>::
  solve_linearized_system(
    const RunTimeParameters::KrylovParameters &krylov_parameters,
    const unsigned int block_id,
    const double right_hand_side_l2_norm)
  {
    if (parameters.verbose)
      *pcout << std::setw(38) << std::left
             << "  Solver: Solving linearized system...";

    dealii::TimerOutput::Scope t(*timer_output, "Solver: Solve ");

    // In this method we create temporal non ghosted copies
    // of the pertinent vectors to be able to perform the solve()
    // operation.
    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_newton_update =
        fe_field->get_distributed_vector_instance(newton_update);

    // The solver's tolerances are passed to the SolverControl instance
    // used to initialize the solver
    dealii::SolverControl solver_control(
        krylov_parameters.n_max_iterations,
        std::max(right_hand_side_l2_norm *
                  krylov_parameters.relative_tolerance,
                 krylov_parameters.absolute_tolerance));

    const auto &system_matrix = jacobian.block(block_id,block_id);

    const auto &right_hand_side = residual.block(block_id);

    auto &solution = distributed_newton_update.block(block_id);

    switch (krylov_parameters.solver_type)
    {
      case RunTimeParameters::SolverType::DirectSolver:
      {
        dealii::TrilinosWrappers::SolverDirect solver(solver_control);

        try
        {
          solver.solve(system_matrix, solution, right_hand_side);
        }
        catch (std::exception &exc)
        {
          internal::handle_std_excepction(exc, "solve method");
        }
        catch (...)
        {
          internal::handle_unknown_exception("solve method");
        }
      }
      break;

    case RunTimeParameters::SolverType::CG:
      {
        dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

        dealii::LinearAlgebraTrilinos::MPI::PreconditionILU
          preconditioner;

        dealii::LinearAlgebraTrilinos::MPI::PreconditionILU::
          AdditionalData additional_data;

        preconditioner.initialize(system_matrix, additional_data);

        try
        {
          solver.solve(system_matrix,
                       solution,
                       right_hand_side,
                       preconditioner);
        }
        catch (std::exception &exc)
        {
          internal::handle_std_excepction(exc, "solve method");
        }
        catch (...)
        {
          internal::handle_unknown_exception("solve method");
        }
      }
      break;

    case RunTimeParameters::SolverType::GMRES:
      {
        dealii::LinearAlgebraTrilinos::SolverGMRES solver(
          solver_control);

        dealii::LinearAlgebraTrilinos::MPI::PreconditionILU
          preconditioner;

        dealii::LinearAlgebraTrilinos::MPI::PreconditionILU::
          AdditionalData additional_data;

        preconditioner.initialize(system_matrix, additional_data);

        try
        {
          solver.solve(system_matrix,
                       solution,
                       right_hand_side,
                       preconditioner);
        }
        catch (std::exception &exc)
        {
          internal::handle_std_excepction(exc, "solve method");
        }
        catch (...)
        {
          internal::handle_unknown_exception("solve method");
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
    newton_update.block(block_id) = solution;

    if (parameters.verbose)
      *pcout << " done!" << std::endl;

    return (solver_control.last_step());
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::update_trial_solution(
      const double relaxation_parameter,
      const unsigned int block_id)
  {
    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_trial_solution =
        fe_field->get_distributed_vector_instance(trial_solution);

    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_newton_update =
        fe_field->get_distributed_vector_instance(newton_update);

    distributed_trial_solution.block(block_id).add(
      relaxation_parameter, distributed_newton_update.block(block_id));

    fe_field->get_affine_constraints().distribute(
      distributed_trial_solution);

    trial_solution.block(block_id) =
      distributed_trial_solution.block(block_id);
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::update_trial_solution(
      const std::vector<double> relaxation_parameters)
  {
    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_trial_solution =
        fe_field->get_distributed_vector_instance(trial_solution);

    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_newton_update =
        fe_field->get_distributed_vector_instance(newton_update);

    for (const auto &locally_owned_dof :
          fe_field->get_locally_owned_dofs())
    {
      if (fe_field->get_locally_owned_displacement_dofs().
            is_element(locally_owned_dof))
      {
        distributed_trial_solution(locally_owned_dof) +=
          relaxation_parameters[0] *
          distributed_newton_update(locally_owned_dof);
      }
      else if (fe_field->get_locally_owned_plastic_slip_dofs().
                is_element(locally_owned_dof))
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
  double GradientCrystalPlasticitySolver<dim>::
  line_search_algorithm()
  {
    double relaxation_parameter = 1.0,
           objective_function_value =
              LineSearch::get_objective_function_value(
                fe_field->get_l2_norm(residual));

    while (!line_search->suficient_descent_condition(
              objective_function_value, relaxation_parameter))
    {
      relaxation_parameter =
        line_search->get_lambda(objective_function_value,
                                relaxation_parameter);

      reset_trial_solution();

      update_trial_solution(relaxation_parameter, 0);

      reset_and_update_quadrature_point_history();

      assemble_residual();

      objective_function_value =
        LineSearch::get_objective_function_value(
          fe_field->get_l2_norm(residual));
    }

    return relaxation_parameter;
  }



  template <int dim>
  double GradientCrystalPlasticitySolver<dim>::
  line_search_algorithm(
    dealii::Vector<double> residual_l2_norms,
    const unsigned int block_id)
  {
    double relaxation_parameter = 1.0,
           objective_function_value =
              LineSearch::get_objective_function_value(
                residual_l2_norms[block_id]);

    while (!line_search->suficient_descent_condition(
              objective_function_value, relaxation_parameter))
    {
      relaxation_parameter =
        line_search->get_lambda(objective_function_value,
                                relaxation_parameter);

      reset_trial_solution();

      update_trial_solution(relaxation_parameter, block_id);

      reset_and_update_quadrature_point_history();

      assemble_residual();

      residual_l2_norms = fe_field->get_sub_l2_norms(residual);

      objective_function_value =
        LineSearch::get_objective_function_value(
          residual_l2_norms[block_id]);
    }

    return relaxation_parameter;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::store_trial_solution(
    const bool flag_store_initial_trial_solution)
  {
    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_trial_solution =
        fe_field->get_distributed_vector_instance(trial_solution);

    if (flag_store_initial_trial_solution)
    {
      initial_trial_solution = distributed_trial_solution;
    }
    else
    {
      tmp_trial_solution = distributed_trial_solution;
    }
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::reset_trial_solution(
    const bool flag_reset_to_initial_trial_solution)
  {
    dealii::LinearAlgebraTrilinos::MPI::BlockVector
      distributed_trial_solution;

    distributed_trial_solution.reinit(
      fe_field->distributed_vector);

    if (flag_reset_to_initial_trial_solution)
    {
      distributed_trial_solution = fe_field->old_solution;
    }
    else
    {
      distributed_trial_solution = tmp_trial_solution;
    }

    fe_field->get_affine_constraints().distribute(
      distributed_trial_solution);

    trial_solution = distributed_trial_solution;
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  update_and_output_nonlinear_solver_logger(
    const dealii::Vector<double>  residual_l2_norms)
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
                                        residual_l2_norms.l2_norm());
    nonlinear_solver_logger.update_value("(R_U)_L2",
                                        residual_l2_norms[0]);
    nonlinear_solver_logger.update_value("(R_G)_L2",
                                        residual_l2_norms[1]);
    nonlinear_solver_logger.update_value("C-Rate",
                                        0);

    nonlinear_solver_logger.log_to_file();

    nonlinear_solver_logger.log_values_to_terminal();
  }



  template <int dim>
  void GradientCrystalPlasticitySolver<dim>::
  update_and_output_nonlinear_solver_logger(
    const unsigned int           nonlinear_iteration,
    const unsigned int           n_krylov_iterations,
    const unsigned int           n_line_search_iterations,
    const dealii::Vector<double> newton_update_l2_norms,
    const dealii::Vector<double> residual_l2_norms,
    const double                 order_of_convergence,
    const double                 relaxation_parameter)
  {
    nonlinear_solver_logger.update_value("N-Itr",
                                        nonlinear_iteration);
    nonlinear_solver_logger.update_value("K-Itr",
                                        n_krylov_iterations);
    nonlinear_solver_logger.update_value("L-Itr",
                                        n_line_search_iterations);
    nonlinear_solver_logger.update_value("(NS)_L2",
                                        relaxation_parameter *
                                        newton_update_l2_norms.l2_norm());
    nonlinear_solver_logger.update_value("(NS_U)_L2",
                                        relaxation_parameter *
                                        newton_update_l2_norms[0]);
    nonlinear_solver_logger.update_value("(NS_G)_L2",
                                        relaxation_parameter *
                                        newton_update_l2_norms[1]);
    nonlinear_solver_logger.update_value("(R)_L2",
                                        residual_l2_norms.l2_norm());
    nonlinear_solver_logger.update_value("(R_U)_L2",
                                        residual_l2_norms[0]);
    nonlinear_solver_logger.update_value("(R_G)_L2",
                                        residual_l2_norms[1]);
    nonlinear_solver_logger.update_value("C-Rate",
                                        order_of_convergence);

    nonlinear_solver_logger.log_to_file();

    nonlinear_solver_logger.log_values_to_terminal();
  }



} // namespace gCP



template void gCP::GradientCrystalPlasticitySolver<2>::
extrapolate_initial_trial_solution(const bool);
template void gCP::GradientCrystalPlasticitySolver<3>::
extrapolate_initial_trial_solution(const bool);

template void gCP::GradientCrystalPlasticitySolver<2>::
solve_nonlinear_system(const bool);
template void gCP::GradientCrystalPlasticitySolver<3>::
solve_nonlinear_system(const bool);

template unsigned int gCP::GradientCrystalPlasticitySolver<2>::
solve_linearized_system(
    const RunTimeParameters::KrylovParameters &,
    const unsigned int,
    const double);
template unsigned int gCP::GradientCrystalPlasticitySolver<3>::
solve_linearized_system(
    const RunTimeParameters::KrylovParameters &,
    const unsigned int,
    const double);

template void gCP::GradientCrystalPlasticitySolver<2>::
update_trial_solution(
  const double,
  const unsigned int);
template void gCP::GradientCrystalPlasticitySolver<3>::
update_trial_solution(
  const double,
  const unsigned int);

template void gCP::GradientCrystalPlasticitySolver<2>::
  update_trial_solution(const std::vector<double>);
template void gCP::GradientCrystalPlasticitySolver<3>::
  update_trial_solution(const std::vector<double>);
