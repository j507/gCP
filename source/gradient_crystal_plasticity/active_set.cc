#include <gCP/gradient_crystal_plasticity.h>



namespace gCP
{


template<int dim>
void GradientCrystalPlasticitySolver<dim>::determine_active_set()
{
  dealii::TimerOutput::Scope  t(*timer_output,
                                "Solver: Active set determination");

  Assert(dof_mapping.size() != 0,
         dealii::ExcMessage("The degree of freedom mapping is empty"));

  assemble_trial_microstress_right_hand_side();

  compute_trial_microstress();

  for (const auto &locally_owned_dof :
        trial_microstress.get_locally_owned_dofs())
  {
    double local_yield_stress = 0.;

    if (true)
    {
      local_yield_stress =
        parameters.constitutive_laws_parameters.
        scalar_microstress_law_parameters.initial_slip_resistance;
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented())
    }

    if (trial_microstress.solution[locally_owned_dof] >
          local_yield_stress)
    {
      active_set.add_index(dof_mapping[locally_owned_dof]);
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