#include <gCP/gradient_crystal_plasticity.h>



namespace gCP
{


template<int dim>
void GradientCrystalPlasticitySolver<dim>::determine_active_set()
{
  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Active set determination");

  if (crystals_data->get_n_slips() == 0)
  {
    return;
  }

  const RunTimeParameters::HardeningLaw &prm =
    parameters.constitutive_laws_parameters.hardening_law_parameters;

  Assert(dof_mapping.size() != 0,
         dealii::ExcMessage("The degree of freedom mapping is empty"));

  assemble_trial_microstress_right_hand_side();

  compute_trial_microstress();

  active_set.clear();

  dealii::AffineConstraints<double> inactive_set_affine_constraints;

  inactive_set_affine_constraints.clear();
  {
    inactive_set_affine_constraints.reinit(
      fe_field->get_locally_owned_dofs());

    inactive_set_affine_constraints.merge(
      fe_field->get_hanging_node_constraints());

    for (const auto &locally_owned_dof :
          trial_microstress->get_locally_owned_dofs())
    {
      double local_yield_stress = 0.;

      if (prm.flag_perfect_plasticity)
      {
        local_yield_stress = prm.initial_slip_resistance;
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented())
      }

      const dealii::types::global_dof_index dof =
        dof_mapping[locally_owned_dof];

      if (std::abs(trial_microstress->solution[locally_owned_dof]) >
            local_yield_stress)
      {
        // The material can not plastically flow at the Dirichlet
        // boundary
        if (!fe_field->get_affine_constraints().is_constrained(dof))
        {
          active_set.add_index(dof);
        }

        // If a degree of freedom leads to plastic flow at a periodic
        // boundary, the corresponding degree of freedom needs to be
        // also added to the active set
        if (fe_field->get_affine_constraints().is_identity_constrained(dof))
        {
          const std::vector<
            std::pair<dealii::types::global_dof_index, double>>
              *constraint_entries =
                fe_field->get_affine_constraints().
                  get_constraint_entries(dof);

          active_set.add_index(dof);

          active_set.add_index((*constraint_entries)[0].first);
        }
      }

      if (!active_set.is_element(dof))
      {
        inactive_set_affine_constraints.add_line(dof);
      }
    }
    internal_newton_method_constraints.merge(
        inactive_set_affine_constraints,
        dealii::AffineConstraints<double>::
          MergeConflictBehavior::right_object_wins);
  }
  inactive_set_affine_constraints.close();

  active_set.compress();

  /*std::cout
    << "Number of degrees of freedom in the active set: "
    << active_set.n_elements()
    << ", "
    << active_set.size()
    << std::endl;

  active_set.print(std::cout);*/
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::determine_inactive_set()
{
  for (const auto &locally_owned_dof :
        fe_field->get_locally_owned_dofs())
  {
    const bool flag_plastic_slip_dof =
      plastic_slip_dofs_set.is_element(locally_owned_dof);

    const bool flag_no_dirichlet_dof =
      fe_field->get_affine_constraints().is_identity_constrained(
        locally_owned_dof) ||
      !fe_field->get_affine_constraints().is_constrained(
        locally_owned_dof);

    const bool flag_inactive_dof =
      !active_set.is_element(locally_owned_dof);

    if (flag_plastic_slip_dof && flag_no_dirichlet_dof &&
          flag_inactive_dof)
    {
      inactive_set.add_index(locally_owned_dof);
    }
  }

  inactive_set.compress();
}



template <int dim>
void GradientCrystalPlasticitySolver<dim>::reset_inactive_set_values()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_trial_solution;

  distributed_trial_solution.reinit(fe_field->distributed_vector);

  distributed_trial_solution = trial_solution;

  for (const auto &locally_owned_dof : inactive_set)
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
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

  distributed_solution.reinit(trial_microstress->distributed_vector);

  distributed_solution = 0.;

  for (unsigned int entry_id = 0;
       entry_id < trial_microstress_lumped_matrix.size();
       entry_id++)
  {
    if (trial_microstress->get_locally_owned_dofs().
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

  trial_microstress->get_hanging_node_constraints().distribute(
    distributed_solution);

  trial_microstress->solution = distributed_solution;
}



} // namespace gCP


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