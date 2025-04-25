#include <gCP/gradient_crystal_plasticity.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

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
      parameters.extrapolation_factor *
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
        fe_field->old_old_solution);

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_newton_update =
      fe_field->get_distributed_vector_instance(
        fe_field->old_solution);

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

  // fe_field->get_affine_constraints().distribute(
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

  // Internal variables' values at the previous step are stored
  prepare_quadrature_point_history();

  store_slip_resistances();

  // Compute and store starting point
  extrapolate_initial_trial_solution(flag_skip_extrapolation ||
    parameters.flag_skip_extrapolation);

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

  std::unique_ptr<gCP::LineSearch> line_search =
    std::make_unique<gCP::LineSearch>(
      parameters.monolithic_algorithm_parameters.
        monolithic_system_solver_parameters.line_search_parameters);

  const RunTimeParameters::NewtonRaphsonParameters
    &newton_parameters = parameters.monolithic_algorithm_parameters.
      monolithic_system_solver_parameters.newton_parameters;

  nonlinear_solver_logger.log_headers_to_terminal();

  // Debug
  double average_order_of_convergence = 0.;

  // Newton-Raphson loop
  do
  {
    // Increase iteration counter
    nonlinear_iteration++;

    AssertThrow(
      nonlinear_iteration <=
        newton_parameters.n_max_iterations,
      ExcMaxIterations(newton_parameters.n_max_iterations));

    // Determine the active (and also inactive) set
    //active_set_algorithm(flag_compute_active_set);

    // Output for debugging purposes (Done at this stage to visualize
    // the computed active set)
    //debug_output();

    // Constraints distribution (Done later to obtain a continous
    // start value for the active set determination). Temporary code
    //distribute_affine_constraints_to_trial_solution();

    // Preparations for the Newton-Update and Line-Search
    store_trial_solution();

    reset_and_update_internal_variables();

    active_set_algorithm(flag_compute_active_set);

    (nonlinear_iteration == 1) ? debug_output() : void(0);

    distribute_affine_constraints_to_trial_solution();

    // Assemble linear system
    assemble_linear_system();

    // Store current l2-norm values and initial objective
    // function
    residual_l2_norms = fe_field->get_sub_l2_norms(residual);

    old_residual_l2_norms = residual_l2_norms;

    line_search->reinit(LineSearch::get_objective_function_value(
      residual_l2_norms.l2_norm()));

    // Terminal and log output
    if (nonlinear_iteration == 1)
    {
      update_and_output_nonlinear_solver_logger(
        residual_l2_norms);
    }

    // Newton-Raphson update
    const unsigned int n_krylov_iterations =
      solve_linearized_system();

    double relaxation_parameter = 1.0;

    update_trial_solution(relaxation_parameter);

    reset_and_update_internal_variables();

    // Compute residuals for convergence-check
    // (and possibly Line-Search)
    assemble_residual();

    residual_l2_norms = fe_field->get_sub_l2_norms(residual);

    // Line search algorithm
    if (newton_parameters.flag_line_search)
    {
      relaxation_parameter = line_search_algorithm(line_search);

      residual_l2_norms = fe_field->get_sub_l2_norms(residual);
    }

    // Terminal and log output
    {
      const dealii::Vector<double> newton_update_l2_norms =
        fe_field->get_sub_l2_norms(newton_update);

      const double order_of_convergence =
        std::log(residual_l2_norms.l2_norm()) /
          std::log(old_residual_l2_norms.l2_norm());

      average_order_of_convergence += order_of_convergence;

      update_and_output_nonlinear_solver_logger(
        nonlinear_iteration,
        n_krylov_iterations,
        line_search->get_n_iterations(),
        newton_update_l2_norms,
        residual_l2_norms,
        order_of_convergence,
        relaxation_parameter);
    }

    debug_output();

    // Convergence check
    flag_successful_convergence =
      residual_l2_norms.l2_norm() <
        newton_parameters.absolute_tolerance; //||
          //residual_l2_norms.l2_norm() <
          //  newton_parameters.relative_tolerance *
          //    initial_residual_l2_norms.l2_norm();

    // Check if the active set changed
    if (flag_successful_convergence)
    {
      if (parameters.constitutive_laws_parameters.
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
            << "new active set..." << std::endl
            << std::endl;
        }
        else
        {
          table_handler.add_value("Iterations", nonlinear_iteration);
          table_handler.add_value(
            "MonoAverageConvergence",
            average_order_of_convergence / nonlinear_iteration);
          table_handler.add_value("MacroAverageConvergence", 0.0);
          table_handler.add_value("ReducedMacroAverageConvergence", 0.0);
          table_handler.add_value("MicroAverageConvergence", 0.0);
        }
      }
      else
      {
        table_handler.add_value("Iterations", nonlinear_iteration);
        table_handler.add_value(
          "MonoAverageConvergence",
          average_order_of_convergence / nonlinear_iteration);
        table_handler.add_value("MacroAverageConvergence", 0.0);
        table_handler.add_value("ReducedMacroAverageConvergence", 0.0);
        table_handler.add_value("MicroAverageConvergence", 0.0);
      }
    }

  } while (!flag_successful_convergence);
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::bouncing_algorithm()
{
  // Declare and initilize local variables and references
  unsigned int macro_nonlinear_iteration = 0,
               micro_nonlinear_iteration = 0,
               macro_loop_counter = 1,
               micro_loop_counter = 0;

  bool
    flag_successful_convergence = false,
    flag_successful_macro_convergence = false,
    flag_successful_micro_convergence = false,
    flag_compute_active_set = true;

  dealii::Vector<double>
    residual_l2_norms,
    old_residual_l2_norms;

  std::unique_ptr<gCP::LineSearch> line_search;

  const RunTimeParameters::NewtonRaphsonParameters
    &macro_newton_parameters = parameters.
      staggered_algorithm_parameters.linear_momentum_solver_parameters.
        newton_parameters,
    &micro_newton_parameters = parameters.
      staggered_algorithm_parameters.pseudo_balance_solver_parameters.
        newton_parameters;

  nonlinear_solver_logger.log_to_all(
    " Linear momentum balance: Starting solution loop #" +
      std::to_string(macro_loop_counter));

  nonlinear_solver_logger.log_headers_to_terminal();

  // Debug
  double average_macro_order_of_convergence = 0.;
  double average_micro_order_of_convergence = 0.;
  double local_average_macro_order_of_convergence = 0.;

  line_search =
    std::make_unique<gCP::LineSearch>(
      parameters.staggered_algorithm_parameters.
        linear_momentum_solver_parameters.
          line_search_parameters);

  // Macro-Newton-Raphson loop
  do
  {
    // Convergence check
    residual_l2_norms = fe_field->get_sub_l2_norms(residual);

    flag_successful_macro_convergence =
      residual_l2_norms[0] <
        macro_newton_parameters.absolute_tolerance; //||
          //residual_l2_norms[0] <
          //  macro_newton_parameters.relative_tolerance *
          //    initial_residual_l2_norms[0];

    if (macro_loop_counter == 1 && macro_nonlinear_iteration == 0)
    {
      flag_successful_macro_convergence = false;
    }

    if (flag_successful_macro_convergence)
    {
      // If the linear momentum balance converges with the plastic
      // slips values of the lattest converged micro Newton-Raphson
      // loop, the solution is considered as found.
      if (flag_successful_micro_convergence)
      {
        flag_successful_convergence = true;

        nonlinear_solver_logger.log_to_all("  Solution converges!");

        if (flag_successful_convergence)
        {
          table_handler.add_value("Iterations", micro_loop_counter);
          table_handler.add_value("MonoAverageConvergence", 0.0);
          table_handler.add_value(
            "MacroAverageConvergence",
            average_macro_order_of_convergence / micro_loop_counter);
          table_handler.add_value("ReducedMacroAverageConvergence", 0.0);
          table_handler.add_value(
            "MicroAverageConvergence",
            average_micro_order_of_convergence / micro_loop_counter);
        }


        continue;
      }
      // Otherwise compute a new trial solution for the plastic slips
      else
      {
        nonlinear_solver_logger.log_to_all("  Solution converges!\n");

        micro_loop_counter++;

        nonlinear_solver_logger.log_to_all(
          " Pseudo-balance: Starting solution loop #" +
            std::to_string(micro_loop_counter));

        nonlinear_solver_logger.log_headers_to_terminal();

        micro_nonlinear_iteration = 0;

        flag_compute_active_set = true;

        // Determine the active (and also inactive) set
        active_set_algorithm(flag_compute_active_set);

        // The distribution corresponds to another method for the
        // sake of the monolithic algorithm
        distribute_affine_constraints_to_trial_solution();

        /*!
          * @todo Exit the micro loop if all locally owned active sets
          * are empty
          */

        debug_output();

        // Revert the plastic slips to the initial trial solution
        // values
        if (parameters.staggered_algorithm_parameters.
              flag_reset_trial_solution_at_micro_loop)
        {
          reset_trial_solution(true, BlockIndex::Micro);
        }

        double local_average_micro_order_of_convergence = 0.;

        line_search =
          std::make_unique<gCP::LineSearch>(
            parameters.staggered_algorithm_parameters.
              pseudo_balance_solver_parameters.
                line_search_parameters);

        // Micro-Newton-Raphson loop
        do
        {
          // Increase iteration counter
          micro_nonlinear_iteration++;

          AssertThrow(
            micro_nonlinear_iteration <=
              micro_newton_parameters.n_max_iterations,
            ExcMaxIterations(
              micro_newton_parameters.n_max_iterations));

          // Preparations for the Newton-Update and Line-Search
          store_trial_solution();

          reset_and_update_internal_variables();

          // Assemble linear system;
          assemble_linear_system();

          // Store current l2-norm values and initial objective
          // function
          residual_l2_norms = fe_field->get_sub_l2_norms(residual);

          old_residual_l2_norms = residual_l2_norms;

          line_search->reinit(
            LineSearch::get_objective_function_value(
              residual_l2_norms[1]));

          // Terminal and log output
          if (micro_nonlinear_iteration == 1)
          {
            update_and_output_nonlinear_solver_logger(
              residual_l2_norms);
          }

          // Newton-Raphson update
          const unsigned int n_krylov_iterations =
            solve_decoupled_linearized_subsystem(BlockIndex::Micro);

          double relaxation_parameter = 1.0;

          update_trial_solution(relaxation_parameter, BlockIndex::Micro);

          reset_and_update_internal_variables();

          // Evaluate new trial solution
          assemble_residual();

          residual_l2_norms =
            fe_field->get_sub_l2_norms(residual);

          // Line search
          if (micro_newton_parameters.flag_line_search)
          {
            relaxation_parameter =
              line_search_algorithm(line_search, BlockIndex::Micro);

            residual_l2_norms = fe_field->get_sub_l2_norms(residual);
          }

          // Terminal and log output
          {
            const dealii::Vector<double> newton_update_l2_norms =
              fe_field->get_sub_l2_norms(newton_update);

            const double order_of_convergence =
              old_residual_l2_norms[1] != 0. ?
                std::log(residual_l2_norms[1]) /
                  std::log(old_residual_l2_norms[1])
              : 0.0;

            local_average_micro_order_of_convergence +=
              order_of_convergence;

            update_and_output_nonlinear_solver_logger(
              micro_nonlinear_iteration,
              n_krylov_iterations,
              line_search->get_n_iterations(),
              newton_update_l2_norms,
              residual_l2_norms,
              order_of_convergence,
              relaxation_parameter);
          }

          // Convergence check
          flag_successful_micro_convergence =
            residual_l2_norms[1] <
              micro_newton_parameters.absolute_tolerance;

        } while (!flag_successful_micro_convergence);

        average_micro_order_of_convergence +=
          local_average_micro_order_of_convergence /
          micro_nonlinear_iteration;

        // Initialize the line search instances
        line_search =
          std::make_unique<gCP::LineSearch>(
            parameters.staggered_algorithm_parameters.
              linear_momentum_solver_parameters.
                line_search_parameters);

        macro_nonlinear_iteration = 0;

        macro_loop_counter++;

        local_average_macro_order_of_convergence = 0.;

        AssertThrow(
          macro_loop_counter <=
            parameters.staggered_algorithm_parameters.
              max_n_solution_loops,
          ExcMaxIterations(
            parameters.staggered_algorithm_parameters.
              max_n_solution_loops));

        nonlinear_solver_logger.log_to_all("  Solution converges!\n");

        nonlinear_solver_logger.log_to_all(
          " Linear momentum balance: Starting solution loop #" +
          std::to_string(macro_loop_counter));

        nonlinear_solver_logger.log_headers_to_terminal();
      }
    }
    else
    {
      // Increase iteration counter
      macro_nonlinear_iteration++;

      AssertThrow(
        macro_nonlinear_iteration <=
          macro_newton_parameters.n_max_iterations,
        ExcMaxIterations(macro_newton_parameters.n_max_iterations));

      if (macro_nonlinear_iteration == 0)
      {
        reset_trial_solution(true, BlockIndex::Macro);
      }

      // Preparations for the Newton-Update and Line-Search
      store_trial_solution();

      reset_and_update_internal_variables();

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
          residual_l2_norms[0]));

      if (macro_nonlinear_iteration == 1)
      {
        update_and_output_nonlinear_solver_logger(
          residual_l2_norms);
      }

      // Newton-Raphson update
      const unsigned int n_krylov_iterations =
        solve_decoupled_linearized_subsystem(BlockIndex::Macro);

      double relaxation_parameter = 1.0;

      update_trial_solution(relaxation_parameter, BlockIndex::Macro);

      reset_and_update_internal_variables();

      // Evaluate new trial solution
      assemble_residual();

      residual_l2_norms =
        fe_field->get_sub_l2_norms(residual);

      // Line search algorithm
      if (macro_newton_parameters.flag_line_search)
      {
        relaxation_parameter =
          line_search_algorithm(line_search, BlockIndex::Macro);

        residual_l2_norms = fe_field->get_sub_l2_norms(residual);
      }

      // Terminal and log output
      {
        const dealii::Vector<double> newton_update_l2_norms =
          fe_field->get_sub_l2_norms(newton_update);

        const double order_of_convergence =
          std::log(residual_l2_norms[0]) /
          std::log(old_residual_l2_norms[0]);

        local_average_macro_order_of_convergence +=
          order_of_convergence;

        update_and_output_nonlinear_solver_logger(
          macro_nonlinear_iteration,
          n_krylov_iterations,
          line_search->get_n_iterations(),
          newton_update_l2_norms,
          residual_l2_norms,
          order_of_convergence,
          relaxation_parameter);
      }

      // Convergence check
      flag_successful_macro_convergence =
        residual_l2_norms[0] <
          macro_newton_parameters.absolute_tolerance;

      if (flag_successful_macro_convergence)
      {
        average_macro_order_of_convergence +=
          local_average_macro_order_of_convergence /
          macro_nonlinear_iteration;
      }

      flag_successful_micro_convergence = false;
    }
  } while (!flag_successful_convergence);
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::embracing_algorihtm()
{
  // Declare and initilize local variables and references
  unsigned int
    macro_nonlinear_iteration = 0,
    micro_nonlinear_iteration = 0;

  bool
    flag_successful_convergence = false,
    flag_successful_macro_convergence = false,
    flag_successful_micro_convergence = false,
    flag_compute_active_set = true;

  dealii::Vector<double>
    residual_l2_norms,
    old_residual_l2_norms,
    tmp_residual_l2_norms,
    tmp_old_residual_l2_norms;

  const RunTimeParameters::NewtonRaphsonParameters
    &macro_newton_parameters = parameters.
      staggered_algorithm_parameters.linear_momentum_solver_parameters.
        newton_parameters,
    &micro_newton_parameters = parameters.
      staggered_algorithm_parameters.pseudo_balance_solver_parameters.
        newton_parameters;

  std::unique_ptr<gCP::LineSearch> line_search;

  // Print terminal headers
  nonlinear_solver_logger.log_headers_to_terminal();

  // Debug
  double average_macro_order_of_convergence = 0.;
  double reduced_average_macro_order_of_convergence = 0.;
  double average_micro_order_of_convergence = 0.;
  double local_average_micro_order_of_convergence = 0.;

  // Distribute constraints to the (extrapolated) solution
  distribute_affine_constraints_to_trial_solution();

  do
  {
    // Initialize the line search instances
    line_search =
      std::make_unique<gCP::LineSearch>(
        parameters.staggered_algorithm_parameters.
          linear_momentum_solver_parameters.line_search_parameters);

    // Store the trial solution for the line-search algorithm
    store_trial_solution();

    // Reset and update all internal variables (Must preceed assembly)
    reset_and_update_internal_variables();

    // Assemble
    assemble_residual();

    // Convergence check
    residual_l2_norms = fe_field->get_sub_l2_norms(residual);

    if (macro_nonlinear_iteration == 0)
    {
      old_residual_l2_norms = residual_l2_norms;
    }

    flag_successful_macro_convergence =
      residual_l2_norms[0] <
      macro_newton_parameters.absolute_tolerance;

    flag_successful_convergence =
      flag_successful_macro_convergence &&
      flag_successful_micro_convergence;

    if (!flag_successful_convergence)
    {
      // Increase iteration counter
      macro_nonlinear_iteration++;

      AssertThrow(
        macro_nonlinear_iteration <=
          macro_newton_parameters.n_max_iterations,
        ExcMaxIterations(macro_newton_parameters.n_max_iterations));

      // Assemble
      assemble_jacobian();

      // Initialize line search instance
      line_search->reinit(
        LineSearch::get_objective_function_value(
          residual_l2_norms[0]));

      // Terminal and log output
      if (macro_nonlinear_iteration == 1)
      {
        update_and_output_nonlinear_solver_logger(
            residual_l2_norms);
      }

      // Compute the Newton-Raphson update
      // For the derivation of the algorithmic jacobian the condition of
      // a neglectable sub-residuum. This condition can not be met in
      // the first nonlinear iteration so we ignore the coupling in this
      // iteration
      unsigned int n_krylov_iterations = 0.;

      if (macro_nonlinear_iteration == 1)
      {
        n_krylov_iterations =
          solve_decoupled_linearized_subsystem(BlockIndex::Macro);
      }
      else
      {
        n_krylov_iterations = solve_reduced_linearized_system();
      }

      // Update trial solution and its dependencies
      double relaxation_parameter = 1.0;

      debug_output();

      update_trial_solution(relaxation_parameter, BlockIndex::Macro);

      reset_and_update_internal_variables();

      assemble_residual();

      residual_l2_norms = fe_field->get_sub_l2_norms(residual);

      // Line search algorithm
      if (macro_newton_parameters.flag_line_search)
      {
        relaxation_parameter =
          line_search_algorithm(line_search, BlockIndex::Macro);

        residual_l2_norms = fe_field->get_sub_l2_norms(residual);
      }

      // Terminal and log output
      {
        const double order_of_convergence =
          std::log(residual_l2_norms[0]) /
          std::log(old_residual_l2_norms[0]);

        average_macro_order_of_convergence += order_of_convergence;

        if (macro_nonlinear_iteration > 1)
        {
          reduced_average_macro_order_of_convergence +=
            order_of_convergence;
        }

        update_and_output_nonlinear_solver_logger(
          macro_nonlinear_iteration,
          n_krylov_iterations,
          line_search->get_n_iterations(),
          fe_field->get_sub_l2_norms(newton_update),
          residual_l2_norms,
          order_of_convergence,
          relaxation_parameter);
      }

      old_residual_l2_norms = residual_l2_norms;

      //// Initialization and start of the micro-loop
      nonlinear_solver_logger.log_to_all("  Microloop...");

      // Increase iteration counter
      micro_nonlinear_iteration = 0;

      // Determine the active (and also inactive) set
      active_set_algorithm(flag_compute_active_set);

      if (parameters.staggered_algorithm_parameters.
            flag_reset_trial_solution_at_micro_loop)
      {
        reset_trial_solution(true, BlockIndex::Micro);
      }

      line_search =
        std::make_unique<gCP::LineSearch>(
          parameters.staggered_algorithm_parameters.
            pseudo_balance_solver_parameters.line_search_parameters);

      local_average_micro_order_of_convergence = 0.;

      do
      {
        // Increase iteration counter
        micro_nonlinear_iteration++;

        AssertThrow(
          micro_nonlinear_iteration <=
              micro_newton_parameters.n_max_iterations,
          ExcMaxIterations(micro_newton_parameters.n_max_iterations));

        // Store the trial solution for the line-search algorithm
        store_trial_solution();

        // Reset and update all internal variables (Must preceed
        // assembly)
        reset_and_update_internal_variables();

        // Assemble linear system
        assemble_linear_system();

        // Store current l2-norm values and initial objective
        // function
        tmp_residual_l2_norms = tmp_old_residual_l2_norms =
          fe_field->get_sub_l2_norms(residual);

        // Initialize line search instance
        line_search->reinit(
          LineSearch::get_objective_function_value(
            tmp_residual_l2_norms[1]));

        // Terminal and log output
        if (micro_nonlinear_iteration == 1)
        {
          update_and_output_nonlinear_solver_logger(
            tmp_residual_l2_norms);
        }

        // Compute Newton-Raphson update
        const unsigned int n_krylov_iterations =
          solve_decoupled_linearized_subsystem(BlockIndex::Micro);

        // Update trial solution and its dependencies
        double relaxation_parameter = 1.0;

        debug_output();

        update_trial_solution(relaxation_parameter, BlockIndex::Micro);

        reset_and_update_internal_variables();

        assemble_residual();

        tmp_residual_l2_norms =
          fe_field->get_sub_l2_norms(residual);

        // Line search
        if (micro_newton_parameters.flag_line_search)
        {
          relaxation_parameter =
            line_search_algorithm(line_search, BlockIndex::Micro);

          tmp_residual_l2_norms = fe_field->get_sub_l2_norms(residual);
        }

        // Terminal and log output
        {
          const double order_of_convergence =
            tmp_old_residual_l2_norms[1] != 0. ?
              std::log(tmp_residual_l2_norms[1]) /
                std::log(tmp_old_residual_l2_norms[1]) : 0.0;

          local_average_micro_order_of_convergence +=
            order_of_convergence;

          update_and_output_nonlinear_solver_logger(
            micro_nonlinear_iteration,
            n_krylov_iterations,
            line_search->get_n_iterations(),
            fe_field->get_sub_l2_norms(newton_update),
            tmp_residual_l2_norms,
            order_of_convergence,
            relaxation_parameter);
        }

        // Convergence check
        flag_successful_micro_convergence =
          tmp_residual_l2_norms[1] <
          micro_newton_parameters.absolute_tolerance;

        if (flag_successful_micro_convergence)
        {
          average_micro_order_of_convergence +=
            local_average_micro_order_of_convergence /
            micro_nonlinear_iteration;
        }

      } while (!flag_successful_micro_convergence);

      nonlinear_solver_logger.log_to_all("  converges!");

    }
    else
    {
      // Terminal and log output
      const double order_of_convergence =
        std::log(tmp_residual_l2_norms[0]) /
        std::log(old_residual_l2_norms[0]);

      update_and_output_nonlinear_solver_logger(
        0,
        0,
        0,
        dealii::Vector<double>(2),
        residual_l2_norms,
        order_of_convergence,
        0.0);

      {
        table_handler.add_value(
          "Iterations",
          macro_nonlinear_iteration);
        table_handler.add_value("MonoAverageConvergence", 0.0);
        table_handler.add_value(
          "MacroAverageConvergence",
          average_macro_order_of_convergence / macro_nonlinear_iteration);
        table_handler.add_value(
          "ReducedMacroAverageConvergence",
          reduced_average_macro_order_of_convergence /
            (macro_nonlinear_iteration - 1));
        table_handler.add_value(
          "MicroAverageConvergence",
          average_micro_order_of_convergence / macro_nonlinear_iteration);
      }
    }

  } while (!flag_successful_convergence);

}



template <int dim>
unsigned int GradientCrystalPlasticitySolver<dim>::
solve_linearized_system()
{
  if (parameters.verbose)
  {
    *pcout << std::setw(38) << std::left
            << "  Solver: Solving linearized system...";
  }

  dealii::TimerOutput::Scope t(
    *timer_output, "Solver: Solve linearized system");

  const RunTimeParameters::KrylovParameters
    &krylov_parameters = parameters.monolithic_algorithm_parameters.
      monolithic_system_solver_parameters.krylov_parameters;

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
    std::max(residual.block(0).l2_norm() *
               krylov_parameters.relative_tolerance,
             krylov_parameters.absolute_tolerance));

  const auto &system_matrix = jacobian.block(0, 0);

  const auto &right_hand_side = residual.block(0);

  auto &solution = distributed_newton_update.block(0);

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
  // fe_field->get_newton_method_constraints()
  internal_newton_method_constraints.distribute(
    distributed_newton_update);

  // Pass the distributed vectors to their ghosted counterpart
  newton_update.block(0) = solution;

  if (parameters.verbose)
  {
      *pcout << " done!" << std::endl;
  }

  return (solver_control.last_step());
}



template <int dim>
unsigned int GradientCrystalPlasticitySolver<dim>::
solve_decoupled_linearized_subsystem(
  const BlockIndex block_index)
{
  if (parameters.verbose)
  {
    *pcout << std::setw(38) << std::left
            << "  Solver: Solving linearized system...";
  }

  dealii::TimerOutput::Scope t(
    *timer_output, "Solver: Solve linearized system");

  // Declare and initialize local variables and references
  const unsigned int block_id = static_cast<unsigned int>(block_index);

  const RunTimeParameters::KrylovParameters
    krylov_parameters =
      (block_index == BlockIndex::Macro) ?
        parameters.staggered_algorithm_parameters.
          linear_momentum_solver_parameters.krylov_parameters :
        parameters.staggered_algorithm_parameters.
          pseudo_balance_solver_parameters.krylov_parameters;

  // Set-up solution vector
  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_newton_update;

  distributed_newton_update.reinit(fe_field->distributed_vector);

  distributed_newton_update = 0.;

  auto &solution = distributed_newton_update.block(block_id);

  // Set-up right-hand-side
  auto &right_hand_side = residual.block(block_id);

  // Set-up system matrix
  auto &system_matrix = jacobian.block(block_id, block_id);

  // The solver's tolerances are passed to the SolverControl instance
  // used to initialize the solver
  dealii::SolverControl solver_control(
    krylov_parameters.n_max_iterations,
    std::max(
      right_hand_side.l2_norm() * krylov_parameters.relative_tolerance,
      krylov_parameters.absolute_tolerance));

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
  // fe_field->get_newton_method_constraints()
  internal_newton_method_constraints.distribute(
    distributed_newton_update);

  // Pass the distributed vectors to their ghosted counterpart
  newton_update = 0.;

  newton_update.block(block_id) = solution;

  if (parameters.verbose)
  {
    *pcout << " done!" << std::endl;
  }

  return (solver_control.last_step());
}



template <int dim>
unsigned int GradientCrystalPlasticitySolver<dim>::
solve_reduced_linearized_system()
{
  if (parameters.verbose)
  {
    *pcout << std::setw(38) << std::left
           << "  Solver: Solving linearized system...";
  }

  dealii::TimerOutput::Scope t(
    *timer_output, "Solver: Solve linearized system");

  // Declare and initialize local variables and references
  const unsigned int macro_block_id = 0,
                     micro_block_id = 1;

  const RunTimeParameters::KrylovParameters
    &macro_krylov_parameters = parameters.
      staggered_algorithm_parameters.linear_momentum_solver_parameters.
        krylov_parameters,
    &micro_krylov_parameters = parameters.
      staggered_algorithm_parameters.pseudo_balance_solver_parameters.
        krylov_parameters;

  // Set-up solution vector
  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_newton_update;

  distributed_newton_update.reinit(fe_field->distributed_vector);

  distributed_newton_update = 0.;

  auto &solution = distributed_newton_update.block(macro_block_id);

  // Set-up right-hand-side
  auto &right_hand_side = residual.block(macro_block_id);

  // Set-up Schur-Complement
  // | D E |
  // | F G |
  const auto D = dealii::TrilinosWrappers::linear_operator<
    dealii::LinearAlgebraTrilinos::MPI::Vector>(
      jacobian.block(macro_block_id, macro_block_id));

  const auto E = dealii::TrilinosWrappers::linear_operator<
    dealii::LinearAlgebraTrilinos::MPI::Vector>(
      jacobian.block(macro_block_id, micro_block_id));

  const auto F = dealii::TrilinosWrappers::linear_operator<
    dealii::LinearAlgebraTrilinos::MPI::Vector>(
      jacobian.block(micro_block_id, macro_block_id));

  const auto G = dealii::TrilinosWrappers::linear_operator<
    dealii::LinearAlgebraTrilinos::MPI::Vector>(
      jacobian.block(micro_block_id, micro_block_id));

  dealii::SolverControl inverse_solver_control(
    micro_krylov_parameters.n_max_iterations,
    std::max(
      residual.block(micro_block_id).l2_norm() *
        micro_krylov_parameters.relative_tolerance,
      micro_krylov_parameters.absolute_tolerance));

  dealii::LinearAlgebraTrilinos::SolverCG
    inverse_solver(inverse_solver_control);

  dealii::LinearAlgebraTrilinos::MPI::PreconditionILU
    inverse_preconditioner;

  dealii::LinearAlgebraTrilinos::MPI::PreconditionILU::AdditionalData
    inverse_additional_data;

  inverse_preconditioner.initialize(
    jacobian.block(
      micro_block_id, micro_block_id),
      inverse_additional_data);

  const auto inv_G =
    dealii::inverse_operator(G, inverse_solver, inverse_preconditioner);

  const auto system_matrix = D - E * inv_G * F;
  // Solve operation
  dealii::SolverControl solver_control(
    macro_krylov_parameters.n_max_iterations,
    std::max(
      right_hand_side.l2_norm() *
        macro_krylov_parameters.relative_tolerance,
      macro_krylov_parameters.absolute_tolerance));

  dealii::LinearAlgebraTrilinos::SolverCG solver(solver_control);

  dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;

  dealii::LinearAlgebraTrilinos::MPI::PreconditionILU::AdditionalData
    additional_data;

  preconditioner.initialize(
    jacobian.block(macro_block_id, macro_block_id),
    additional_data);

  const auto inversed_system_matrix =
    dealii::inverse_operator(
    system_matrix, solver, preconditioner);

  solution = inversed_system_matrix * right_hand_side;

  // Zero out the Dirichlet boundary conditions
  // fe_field->get_newton_method_constraints()
  internal_newton_method_constraints.distribute(
    distributed_newton_update);

  // Pass the distributed vectors to their ghosted counterpart
  newton_update = 0.;

  newton_update.block(macro_block_id) = solution;

  if (parameters.verbose)
  {
    *pcout << " done!" << std::endl;
  }

  return (solver_control.last_step());
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::update_trial_solution(
  const double relaxation_parameter,
  const BlockIndex block_index)
{
  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_trial_solution =
      fe_field->get_distributed_vector_instance(trial_solution);

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_newton_update =
      fe_field->get_distributed_vector_instance(newton_update);

  const unsigned int block_id = static_cast<unsigned int>(block_index);

  distributed_trial_solution.block(block_id).add(
    relaxation_parameter, distributed_newton_update.block(block_id));

  fe_field->get_affine_constraints().distribute(
    distributed_trial_solution);

  trial_solution.block(block_id) =
    distributed_trial_solution.block(block_id);
}




template <int dim>
void GradientCrystalPlasticitySolver<dim>::update_trial_solution(
  const double relaxation_parameter)
{
  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_trial_solution =
      fe_field->get_distributed_vector_instance(trial_solution);

  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_newton_update =
      fe_field->get_distributed_vector_instance(newton_update);

  const unsigned int block_id = 0;

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
    if (fe_field->get_locally_owned_displacement_dofs().is_element(locally_owned_dof))
    {
      distributed_trial_solution(locally_owned_dof) +=
        relaxation_parameters[0] *
        distributed_newton_update(locally_owned_dof);
    }
    else if (fe_field->get_locally_owned_plastic_slip_dofs().is_element(locally_owned_dof))
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
line_search_algorithm(
  const std::unique_ptr<gCP::LineSearch> &line_search)
{
  dealii::TimerOutput::Scope t(
    *timer_output, "Solver: Line-search algorithm");

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

    update_trial_solution(relaxation_parameter, BlockIndex::Macro);

    reset_and_update_internal_variables();

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
  const std::unique_ptr<gCP::LineSearch> &line_search,
  const BlockIndex block_index)
{
  dealii::TimerOutput::Scope t(
    *timer_output, "Solver: Line-search algorithm");

  const unsigned int block_id =
    static_cast<unsigned int>(block_index);

  double relaxation_parameter = 1.0,
         objective_function_value =
            LineSearch::get_objective_function_value(
              fe_field->get_sub_l2_norms(residual)[block_id]);

  while (!line_search->suficient_descent_condition(
      objective_function_value, relaxation_parameter))
  {
    relaxation_parameter =
      line_search->get_lambda(objective_function_value,
                              relaxation_parameter);

    reset_trial_solution();

    update_trial_solution(relaxation_parameter, block_index);

    reset_and_update_internal_variables();

    assemble_residual();

    objective_function_value =
      LineSearch::get_objective_function_value(
        fe_field->get_sub_l2_norms(residual)[block_id]);
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
  const bool flag_reset_to_initial_trial_solution,
  const BlockIndex block_index)
{
  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_trial_solution;

  distributed_trial_solution.reinit(
    fe_field->distributed_vector);

  if (flag_reset_to_initial_trial_solution)
  {
    distributed_trial_solution = initial_trial_solution;
  }
  else
  {
    distributed_trial_solution = tmp_trial_solution;
  }

  fe_field->get_affine_constraints().distribute(
    distributed_trial_solution);

  const unsigned int block_id =
    static_cast<unsigned int>(block_index);

  trial_solution.block(block_id) =
    distributed_trial_solution.block(block_id);
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::
update_and_output_nonlinear_solver_logger(
  const dealii::Vector<double> residual_l2_norms)
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
    const unsigned int nonlinear_iteration,
    const unsigned int n_krylov_iterations,
    const unsigned int n_line_search_iterations,
    const dealii::Vector<double> newton_update_l2_norms,
    const dealii::Vector<double> residual_l2_norms,
    const double order_of_convergence,
    const double relaxation_parameter)
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
solve_linearized_system();
template unsigned int gCP::GradientCrystalPlasticitySolver<3>::
solve_linearized_system();

template unsigned int gCP::GradientCrystalPlasticitySolver<2>::
solve_reduced_linearized_system();
template unsigned int gCP::GradientCrystalPlasticitySolver<3>::
solve_reduced_linearized_system();

template unsigned int gCP::GradientCrystalPlasticitySolver<2>::
solve_decoupled_linearized_subsystem(const BlockIndex);
template unsigned int gCP::GradientCrystalPlasticitySolver<3>::
solve_decoupled_linearized_subsystem(const BlockIndex);

template void gCP::GradientCrystalPlasticitySolver<2>::
update_trial_solution(
  const double);
template void gCP::GradientCrystalPlasticitySolver<3>::
update_trial_solution(
  const double);

template void gCP::GradientCrystalPlasticitySolver<2>::
update_trial_solution(
  const double,
  const BlockIndex);
template void gCP::GradientCrystalPlasticitySolver<3>::
update_trial_solution(
  const double,
  const BlockIndex);

template void gCP::GradientCrystalPlasticitySolver<2>::
update_trial_solution(const std::vector<double>);
template void gCP::GradientCrystalPlasticitySolver<3>::
update_trial_solution(const std::vector<double>);
