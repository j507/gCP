#include <gCP/gradient_crystal_plasticity.h>



namespace gCP
{


template<int dim>
void GradientCrystalPlasticitySolver<dim>::determine_active_set()
{
  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Active set determination");

  const RunTimeParameters::HardeningLaw &prm =
    parameters.constitutive_laws_parameters.hardening_law_parameters;

  Assert(dof_mapping.size() != 0,
         dealii::ExcMessage("The degree of freedom mapping is empty"));

  assemble_trial_microstress_right_hand_side();

  compute_trial_microstress();

  active_set.clear();

  for (const auto &locally_owned_dof :
        trial_microstress.get_locally_owned_dofs())
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

    if (std::abs(trial_microstress.solution[locally_owned_dof]) >
          local_yield_stress)
    {
      const dealii::types::global_dof_index dof =
        dof_mapping[locally_owned_dof];

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
  }

}



template<int dim>
void GradientCrystalPlasticitySolver<dim>::compute_trial_microstress()
{
  dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution;

  distributed_solution.reinit(trial_microstress.distributed_vector);

  distributed_solution = 0.;

  for (unsigned int entry_id = 0;
       entry_id < trial_microstress_lumped_matrix.size();
       entry_id++)
  {
    if (trial_microstress.get_locally_owned_dofs().
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

  trial_microstress.get_hanging_node_constraints().distribute(
    distributed_solution);

  trial_microstress.solution = distributed_solution;
}



} // namespace gCP


// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
determine_active_set();

template void gCP::GradientCrystalPlasticitySolver<3>::
determine_active_set();

// Explicit instantiations
template void gCP::GradientCrystalPlasticitySolver<2>::
compute_trial_microstress();

template void gCP::GradientCrystalPlasticitySolver<3>::
compute_trial_microstress();