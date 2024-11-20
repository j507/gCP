#include <gCP/gradient_crystal_plasticity.h>



namespace gCP
{



template<int dim>
void GradientCrystalPlasticitySolver<dim>::active_set_algorithm(
  bool &flag_compute_active_set)
{
  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Active set algorithm");

  if (parameters.constitutive_laws_parameters.
        scalar_microstress_law_parameters.flag_rate_independent)
  {
    if (flag_compute_active_set)
    {
      // Reset inactive set (on the constraints level)
      reset_internal_newton_method_constraints();

      determine_active_set();

      determine_inactive_set();

      reset_inactive_set_values();

      flag_compute_active_set = false;
    }
  }
  else
  {
    locally_owned_active_set =
      fe_field->get_locally_owned_plastic_slip_dofs();
  }
}


template<int dim>
void GradientCrystalPlasticitySolver<dim>::determine_active_set()
{
  if (crystals_data->get_n_slips() == 0)
  {
    return;
  }

  const RunTimeParameters::HardeningLaw &prm =
    parameters.constitutive_laws_parameters.hardening_law_parameters;

  assemble_trial_microstress_right_hand_side();

  compute_trial_microstress();

  locally_owned_active_set.clear();

  dealii::AffineConstraints<double> inactive_set_affine_constraints;

  inactive_set_affine_constraints.clear();
  {
    inactive_set_affine_constraints.reinit(
      fe_field->get_locally_owned_dofs());

    inactive_set_affine_constraints.merge(
      fe_field->get_hanging_node_constraints());

    for (const auto &locally_owned_dof :
          fe_field->get_locally_owned_plastic_slip_dofs())
    {
      double local_yield_stress = 0.;

      if (prm.flag_perfect_plasticity)
      {
        local_yield_stress =
          prm.initial_slip_resistance /
            parameters.dimensionless_form_parameters.
              characteristic_quantities.slip_resistance;
      }
      else
      {
        local_yield_stress = slip_resistance(locally_owned_dof);
      }

      if (std::abs(trial_microstress->
            solution[locally_owned_dof]) > local_yield_stress)
      {
        // The material can not plastically flow at the Dirichlet
        // boundary
        if (!fe_field->get_affine_constraints().is_constrained(
              locally_owned_dof))
        {
          locally_owned_active_set.add_index(locally_owned_dof);
        }

        // If a degree of freedom leads to plastic flow at a periodic
        // boundary, the corresponding degree of freedom needs to be
        // also added to the active set
        if (fe_field->get_affine_constraints().is_identity_constrained(
              locally_owned_dof))
        {
          const std::vector<
            std::pair<dealii::types::global_dof_index, double>>
              *constraint_entries =
                fe_field->get_affine_constraints().
                  get_constraint_entries(locally_owned_dof);

          locally_owned_active_set.add_index(locally_owned_dof);

          locally_owned_active_set.add_index((*constraint_entries)[0].first);
        }
      }

      if (!locally_owned_active_set.is_element(locally_owned_dof))
      {
        inactive_set_affine_constraints.add_line(locally_owned_dof);
      }
    }

    internal_newton_method_constraints.merge(
        inactive_set_affine_constraints,
        dealii::AffineConstraints<double>::
          MergeConflictBehavior::right_object_wins);
  }
  inactive_set_affine_constraints.close();

  locally_owned_active_set.compress();

  if (parameters.flag_output_debug_fields)
  {
    active_set =
      trial_microstress->get_distributed_vector_instance(
        trial_microstress->solution);

    for (unsigned int entry_id = 0; entry_id < active_set.size();
          entry_id++)
    {
      if (trial_microstress->get_locally_owned_plastic_slip_dofs().
            is_element(entry_id))
      {
        active_set(entry_id) =
          std::abs(active_set(entry_id)) >
                slip_resistance(entry_id);
      }
    }
  }
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::determine_inactive_set()
{
  locally_owned_inactive_set.clear();

  for (const auto &locally_owned_dof :
        fe_field->get_locally_owned_dofs())
  {
    const bool flag_plastic_slip_dof =
      fe_field->get_locally_owned_plastic_slip_dofs().is_element(
        locally_owned_dof);

    const bool flag_no_dirichlet_dof =
      fe_field->get_affine_constraints().is_identity_constrained(
        locally_owned_dof) ||
      !fe_field->get_affine_constraints().is_constrained(
        locally_owned_dof);

    const bool flag_inactive_dof =
      !locally_owned_active_set.is_element(locally_owned_dof);

    if (flag_plastic_slip_dof && flag_no_dirichlet_dof &&
          flag_inactive_dof)
    {
      locally_owned_inactive_set.add_index(locally_owned_dof);
    }
  }

  locally_owned_inactive_set.compress();
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::reset_inactive_set_values()
{
  dealii::LinearAlgebraTrilinos::MPI::BlockVector
    distributed_trial_solution;

  distributed_trial_solution.reinit(fe_field->distributed_vector);

  distributed_trial_solution = trial_solution;

  for (const auto &locally_owned_dof : locally_owned_inactive_set)
  {
    distributed_trial_solution(locally_owned_dof) =
      fe_field->old_solution(locally_owned_dof);
  }

  fe_field->get_affine_constraints().distribute(
    distributed_trial_solution);

  trial_solution = distributed_trial_solution;
}


template<int dim>
void GradientCrystalPlasticitySolver<dim>::compute_trial_microstress()
{
  dealii::LinearAlgebraTrilinos::MPI::BlockVector distributed_solution;

  distributed_solution.reinit(
    trial_microstress->distributed_vector);

  distributed_solution = 0.;

  if (true)
  {
    for (unsigned int entry_id = 0; entry_id < distributed_solution.size();
          entry_id++)
    {
      if (trial_microstress->get_locally_owned_plastic_slip_dofs().
            is_element(entry_id))
      {
        AssertThrow(
          trial_microstress_lumped_matrix(entry_id) != 0.0,
          dealii::ExcMessage(""));

        distributed_solution(entry_id) =
          trial_microstress_right_hand_side(entry_id) /
            trial_microstress_lumped_matrix(entry_id);
      }
    }
  }
  else
  {/*
    const RunTimeParameters::KrylovParameters &krylov_parameters =
      parameters.krylov_parameters;

    // The solver's tolerances are passed to the SolverControl instance
    // used to initialize the solver
    dealii::SolverControl solver_control(
        krylov_parameters.n_max_iterations,
        std::max(trial_microstress_right_hand_side.l2_norm() *
                 krylov_parameters.relative_tolerance,
                 krylov_parameters.absolute_tolerance));

    dealii::TrilinosWrappers::SolverDirect solver(solver_control);

    dealii::LinearAlgebraTrilinos::MPI::PreconditionILU::AdditionalData
      additional_data;

    dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;

    preconditioner.initialize(trial_microstress_matrix, additional_data);

    try
    {
      solver.solve(trial_microstress_matrix,
                  distributed_solution,
                  trial_microstress_right_hand_side);

      solver.solve(trial_microstress_matrix,
                   distributed_solution,
                   trial_microstress_right_hand_side);
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
    }*/
  }

  trial_microstress->get_hanging_node_constraints().distribute(
    distributed_solution);

  trial_microstress->solution = distributed_solution;
}



} // namespace gCP


// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
active_set_algorithm(bool &);

template void gCP::GradientCrystalPlasticitySolver<3>::
active_set_algorithm(bool &);

// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
determine_active_set();

template void gCP::GradientCrystalPlasticitySolver<3>::
determine_active_set();


// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
determine_inactive_set();

template void gCP::GradientCrystalPlasticitySolver<3>::
determine_inactive_set();

// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
reset_inactive_set_values();

template void gCP::GradientCrystalPlasticitySolver<3>::
reset_inactive_set_values();

// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
compute_trial_microstress();

template void gCP::GradientCrystalPlasticitySolver<3>::
compute_trial_microstress();